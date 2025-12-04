"""
Smooth Reconstruction Algorithm for EIT

Implementation of the Gauss-Newton method with Laplacian regularization
for electrical impedance tomography reconstruction.

Based on Algorithm 1 from Gehre et al., J. Computational and Applied Mathematics (2012):
"Reconstruction algorithm based on smoothness"

The algorithm solves:
    min_sigma 1/2 ||H(sigma - sigma0) - (U_delta - F(sigma0))I||^2 + alpha/2 ||grad(sigma)||^2

where H is the Jacobian, sigma is conductivity, U_delta are measurements,
and alpha is the regularization parameter.

Directory structure for outputs:
    reconstruction_results/
        subject_00/
            3layer/
                sample_1_ischemic/
                    alpha_1e-08/
                        reconstruction.npz
                        reconstruction.png
                        convergence.png
                    alpha_1e-07/
                        ...
                sample_2_ischemic/
                    ...
            6layer/
                sample_2_ischemic/
                    ...
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from pathlib import Path
from typing import Tuple, Optional
from scipy.sparse import csr_matrix, lil_matrix, diags
from scipy.sparse.linalg import spsolve
import time

from eit_forward_skfem import (
    EIT, current_method, load_brainweb_mesh,
    materials_to_conductivity, MeshTri
)


def compute_laplacian_matrix(mesh: MeshTri) -> csr_matrix:
    """
    Compute discrete negative Laplacian matrix (-Delta) for P0 elements.

    For P0 (piecewise constant) elements, we use a finite volume approach:
    -Delta sigma_i = Sum_j (sigma_j - sigma_i) * (edge_length / distance) for neighboring cells

    Args:
        mesh: scikit-fem MeshTri

    Returns:
        Sparse matrix representing -Delta operator (n_elements x n_elements)
    """
    n_elements = mesh.t.shape[1]

    # Build element adjacency through shared edges
    L = lil_matrix((n_elements, n_elements))

    # Find element neighbors through mesh facets
    f2t = mesh.f2t  # (2, n_facets) - two triangles per facet (-1 if boundary)

    for facet_idx in range(mesh.facets.shape[1]):
        t1, t2 = f2t[:, facet_idx]

        # Skip boundary facets
        if t1 < 0 or t2 < 0:
            continue

        # Get facet nodes
        facet_nodes = mesh.facets[:, facet_idx]
        p1 = mesh.p[:, facet_nodes[0]]
        p2 = mesh.p[:, facet_nodes[1]]
        edge_length = np.linalg.norm(p2 - p1)

        # Get centroids of adjacent triangles
        centroid1 = mesh.p[:, mesh.t[:, t1]].mean(axis=1)
        centroid2 = mesh.p[:, mesh.t[:, t2]].mean(axis=1)
        distance = np.linalg.norm(centroid2 - centroid1)

        # Weight: edge_length / distance (like finite volume)
        weight = edge_length / distance if distance > 0 else 0

        # Build Laplacian: -Delta[i,i] = sum of weights, -Delta[i,j] = -weight
        L[t1, t1] += weight
        L[t2, t2] += weight
        L[t1, t2] -= weight
        L[t2, t1] -= weight

    return csr_matrix(L)


def smooth_reconstruction(
    mesh_forward: MeshTri,
    electrode_markers_forward: list,
    materials_forward: np.ndarray,
    mesh_recon: MeshTri,
    electrode_markers_recon: list,
    materials_recon: np.ndarray,
    U_measured: np.ndarray,
    sigma_baseline: np.ndarray,
    alpha: float = 1e-6,
    max_iterations: int = 10,
    tolerance: float = 1e-4,
    n_electrodes: int = 16,
    contact_impedance: float = 0.01,
    current_pattern_method: int = 5,
    verbose: bool = True
) -> Tuple[np.ndarray, dict]:
    """
    Smooth reconstruction algorithm (Gauss-Newton with Laplacian regularization).

    Args:
        mesh_forward: Fine mesh used to generate synthetic data
        electrode_markers_forward: Electrode markers for forward mesh
        materials_forward: Material IDs for forward mesh
        mesh_recon: Coarser mesh for reconstruction (to avoid inverse crime)
        electrode_markers_recon: Electrode markers for reconstruction mesh
        materials_recon: Material IDs for reconstruction mesh
        U_measured: Measured electrode voltages (N_patterns, L)
        sigma_baseline: Baseline conductivity for reconstruction mesh
        alpha: Regularization parameter
        max_iterations: Maximum number of Gauss-Newton iterations
        tolerance: Stopping criterion (relative change)
        n_electrodes: Number of electrodes
        contact_impedance: Contact impedance
        current_pattern_method: Current injection pattern method
        verbose: Print progress

    Returns:
        sigma_final: Reconstructed conductivity
        history: Dictionary with iteration history
    """
    if verbose:
        print("="*70)
        print("SMOOTH RECONSTRUCTION (Gauss-Newton + Laplacian)")
        print("="*70)
        print(f"Forward mesh: {mesh_forward.t.shape[1]} elements")
        print(f"Reconstruction mesh: {mesh_recon.t.shape[1]} elements")
        print(f"Mesh ratio (forward/recon): {mesh_forward.t.shape[1] / mesh_recon.t.shape[1]:.1f}x")
        print(f"Regularization alpha = {alpha:.2e}")
        print(f"Max iterations: {max_iterations}")

    # Create current injection patterns
    L = n_electrodes
    n_patterns = L - 1
    Inj = current_method(L=L, l=n_patterns, method=current_pattern_method, value=1.0)

    # Create EIT solver on reconstruction mesh
    z = np.ones(L) * contact_impedance
    solver_recon = EIT(
        L=L,
        Inj=Inj,
        z=z,
        mesh=mesh_recon,
        electrode_markers=electrode_markers_recon
    )

    # Compute Laplacian matrix for regularization
    if verbose:
        print("\nComputing Laplacian matrix...")
    Laplacian = compute_laplacian_matrix(mesh_recon)

    # Initialize reconstruction
    sigma = sigma_baseline.copy()
    n_elements = len(sigma)

    # History tracking
    history = {
        'sigma': [sigma.copy()],
        'residual_norm': [],
        'relative_change': [],
        'data_misfit': [],
        'regularization': []
    }

    if verbose:
        print("\nStarting Gauss-Newton iterations...")
        print("-"*70)

    for iteration in range(max_iterations):
        iter_start = time.time()

        # Step 1: Solve forward problem with current sigma
        u_all, U_all = solver_recon.forward_solve(sigma)
        U_pred = np.array(U_all)  # (N_patterns, L)

        # Step 2: Compute residual (measurement - prediction)
        residual = U_measured - U_pred
        residual_flat = residual.flatten()
        residual_norm = np.linalg.norm(residual_flat)

        # Step 3: Compute Jacobian H_j = grad_sigma F(sigma^{j-1}) I
        if verbose:
            print(f"Iteration {iteration+1}: Computing Jacobian...", end=" ")

        J = solver_recon.calc_jacobian(sigma, u_all)  # (N_patterns*L, n_elements)

        # Step 4: Solve linearized system
        # (H^T H + alpha(-Delta)) sigma_update = H^T (U_delta - F(sigma^{j-1})I) + H sigma^{j-1}
        # Simplifying: (H^T H + alpha(-Delta)) Delta_sigma = H^T residual
        # where Delta_sigma = sigma_new - sigma_old

        # Convert J to sparse CSR for memory efficiency
        from scipy.sparse import csr_matrix as sparse_csr
        J_sparse = sparse_csr(J)

        HTH = J_sparse.T @ J_sparse
        HT_residual = J_sparse.T @ residual_flat

        # Build system matrix (both are sparse)
        system_matrix = HTH + alpha * Laplacian

        # Build RHS: H^T (U_delta - F(sigma)) (this is the gradient direction)
        rhs = HT_residual

        # Solve for update
        delta_sigma = spsolve(system_matrix.tocsr(), rhs)

        # Step 5: Update conductivity
        sigma_new = sigma + delta_sigma

        # Ensure positivity
        sigma_new = np.maximum(sigma_new, 0.001)

        # Compute relative change
        relative_change = np.linalg.norm(sigma_new - sigma) / np.linalg.norm(sigma)

        # Compute objective function components
        data_misfit = 0.5 * np.linalg.norm(residual_flat)**2
        reg_term = 0.5 * alpha * (Laplacian @ sigma).T @ sigma

        # Update
        sigma = sigma_new

        # Store history
        history['sigma'].append(sigma.copy())
        history['residual_norm'].append(residual_norm)
        history['relative_change'].append(relative_change)
        history['data_misfit'].append(data_misfit)
        history['regularization'].append(reg_term)

        iter_time = time.time() - iter_start

        if verbose:
            print(f"Done ({iter_time:.2f}s)")
            print(f"  Residual norm: {residual_norm:.4e}")
            print(f"  Relative change: {relative_change:.4e}")
            print(f"  Data misfit: {data_misfit:.4e}")
            print(f"  Regularization: {reg_term:.4e}")
            print(f"  sigma range: [{sigma.min():.4f}, {sigma.max():.4f}]")

        # Check stopping criterion
        if relative_change < tolerance:
            if verbose:
                print(f"\nConverged after {iteration+1} iterations!")
            break

    if verbose:
        print("-"*70)
        print(f"Reconstruction complete")
        print("="*70)

    return sigma, history


def plot_reconstruction_results(
    mesh_true: MeshTri,
    sigma_true: np.ndarray,
    mesh_recon: MeshTri,
    sigma_recon: np.ndarray,
    sigma_baseline: np.ndarray,
    history: dict,
    title: str,
    save_path: Optional[str] = None
):
    """
    Visualize reconstruction results.

    Args:
        mesh_true: True (forward) mesh
        sigma_true: True conductivity
        mesh_recon: Reconstruction mesh
        sigma_recon: Reconstructed conductivity
        sigma_baseline: Baseline conductivity
        history: Reconstruction history
        title: Plot title
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(16, 10))

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # True conductivity
    ax1 = fig.add_subplot(gs[0, 0])
    tri_true = Triangulation(mesh_true.p[0], mesh_true.p[1], mesh_true.t.T)
    im1 = ax1.tripcolor(tri_true, sigma_true, shading='flat', cmap='viridis')
    ax1.set_aspect('equal')
    ax1.set_title('True Conductivity\n(Forward Mesh)', fontsize=10)
    plt.colorbar(im1, ax=ax1, label='sigma (S/m)')

    # Baseline conductivity
    ax2 = fig.add_subplot(gs[0, 1])
    tri_recon = Triangulation(mesh_recon.p[0], mesh_recon.p[1], mesh_recon.t.T)
    im2 = ax2.tripcolor(tri_recon, sigma_baseline, shading='flat', cmap='viridis')
    ax2.set_aspect('equal')
    ax2.set_title('Baseline (Initial Guess)\n(Inverse Mesh)', fontsize=10)
    plt.colorbar(im2, ax=ax2, label='sigma (S/m)')

    # Reconstructed conductivity
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.tripcolor(tri_recon, sigma_recon, shading='flat', cmap='viridis')
    ax3.set_aspect('equal')
    ax3.set_title('Reconstructed\n(Inverse Mesh)', fontsize=10)
    plt.colorbar(im3, ax=ax3, label='sigma (S/m)')

    # Difference: Reconstructed - Baseline
    ax4 = fig.add_subplot(gs[1, 0])
    diff = sigma_recon - sigma_baseline
    vmax = max(abs(diff.min()), abs(diff.max()))
    im4 = ax4.tripcolor(tri_recon, diff, shading='flat', cmap='RdBu_r',
                        vmin=-vmax, vmax=vmax)
    ax4.set_aspect('equal')
    ax4.set_title('Difference (Recon - Baseline)', fontsize=10)
    plt.colorbar(im4, ax=ax4, label='Delta_sigma (S/m)')

    # Convergence: Relative change
    ax5 = fig.add_subplot(gs[1, 1])
    if len(history['relative_change']) > 0:
        ax5.semilogy(range(1, len(history['relative_change'])+1),
                     history['relative_change'], 'o-', linewidth=2)
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Relative Change')
        ax5.set_title('Convergence', fontsize=10)
        ax5.grid(True, alpha=0.3)

    # Residual norm
    ax6 = fig.add_subplot(gs[1, 2])
    if len(history['residual_norm']) > 0:
        ax6.semilogy(range(1, len(history['residual_norm'])+1),
                     history['residual_norm'], 'o-', linewidth=2, color='orange')
        ax6.set_xlabel('Iteration')
        ax6.set_ylabel('Residual Norm')
        ax6.set_title('Data Misfit', fontsize=10)
        ax6.grid(True, alpha=0.3)

    # Objective function components
    ax7 = fig.add_subplot(gs[2, :])
    if len(history['data_misfit']) > 0:
        iterations = range(1, len(history['data_misfit'])+1)
        ax7.semilogy(iterations, history['data_misfit'], 'o-',
                     label='Data Misfit', linewidth=2)
        ax7.semilogy(iterations, history['regularization'], 's-',
                     label='Regularization', linewidth=2)
        total = np.array(history['data_misfit']) + np.array(history['regularization'])
        ax7.semilogy(iterations, total, '^-',
                     label='Total Objective', linewidth=2)
        ax7.set_xlabel('Iteration')
        ax7.set_ylabel('Objective Function')
        ax7.set_title('Objective Function Evolution', fontsize=10)
        ax7.legend()
        ax7.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.close()


def reconstruct_sample(
    subject_id: str,
    layer_type: str,
    sample_name: str,
    alphas: list = [1e-8, 1e-7, 1e-6],
    noise_level: float = 0.01,
    max_iterations: int = 10,
    base_dir: str = "/mnt/d/Programming/EIT"
):
    """
    Reconstruct a single stroke sample.

    Args:
        subject_id: Subject ID (e.g., 'subject_00')
        layer_type: '3layer' or '6layer'
        sample_name: Sample filename (e.g., 'subject_00_3layer_sample_1_ischemic.npz')
        alphas: List of regularization parameters to test
        noise_level: Noise level for synthetic measurements
        max_iterations: Maximum Gauss-Newton iterations
        base_dir: Base directory for EIT project
    """
    print("\n" + "="*70)
    print(f"RECONSTRUCTION: {sample_name}")
    print("="*70)

    base_path = Path(base_dir)

    # Paths - using NEW dual mesh structure
    mesh_path_forward = base_path / "brainweb_subjects" / subject_id / f"meshes_{layer_type}" / "forward" / "head_mesh.npz"
    mesh_path_inverse = base_path / "brainweb_subjects" / subject_id / f"meshes_{layer_type}" / "inverse" / "head_mesh.npz"
    sample_path = base_path / "brainweb_stroke_samples" / sample_name

    # Load sample
    sample_data = np.load(sample_path)
    print(f"\nLoaded sample: {sample_name}")
    print(f"  Stroke type: {sample_data['stroke_type']}")
    print(f"  Stroke center: {sample_data['stroke_center']}")
    print(f"  Stroke radius: {sample_data['stroke_radius']:.4f}")

    # Load FORWARD mesh (fine) and conductivity with stroke
    mesh_forward, electrode_markers_forward, materials_forward = load_brainweb_mesh(
        str(mesh_path_forward), n_electrodes=16
    )
    sigma_true = sample_data['conductivity']

    print(f"\nForward mesh: {mesh_forward.t.shape[1]} elements")
    print(f"True conductivity range: [{sigma_true.min():.4f}, {sigma_true.max():.4f}] S/m")

    # Generate synthetic measurements using forward (fine) mesh
    print("\nGenerating synthetic measurements...")
    L = 16
    Inj = current_method(L=L, l=L-1, method=5, value=1.0)
    z = np.ones(L) * 0.01

    solver_forward = EIT(
        L=L, Inj=Inj, z=z,
        mesh=mesh_forward,
        electrode_markers=electrode_markers_forward
    )

    _, U_measured = solver_forward.forward_solve(sigma_true)
    U_measured = np.array(U_measured)

    # Add noise
    noise = np.random.randn(*U_measured.shape) * noise_level * np.abs(U_measured).mean()
    U_measured_noisy = U_measured + noise

    print(f"  Measurement shape: {U_measured.shape}")
    print(f"  Noise level: {noise_level*100}%")

    # Load INVERSE mesh (coarse) for reconstruction
    mesh_recon, electrode_markers_recon, materials_recon = load_brainweb_mesh(
        str(mesh_path_inverse), n_electrodes=16
    )

    # Create baseline conductivity (homogeneous model)
    sigma_baseline = materials_to_conductivity(materials_recon, layer_type=layer_type)

    print(f"\nInverse mesh: {mesh_recon.t.shape[1]} elements")
    print(f"Baseline conductivity range: [{sigma_baseline.min():.4f}, {sigma_baseline.max():.4f}] S/m")

    # Create output directory structure
    sample_basename = Path(sample_name).stem  # e.g., 'subject_00_3layer_sample_1_ischemic'
    output_base = base_path / "reconstruction_results" / subject_id / layer_type / sample_basename
    output_base.mkdir(parents=True, exist_ok=True)

    # Run reconstruction for each alpha
    for alpha in alphas:
        print(f"\n{'='*70}")
        print(f"Reconstruction with alpha = {alpha:.2e}")
        print(f"{'='*70}")

        alpha_dir = output_base / f"alpha_{alpha:.0e}"
        alpha_dir.mkdir(exist_ok=True)

        sigma_recon, history = smooth_reconstruction(
            mesh_forward=mesh_forward,
            electrode_markers_forward=electrode_markers_forward,
            materials_forward=materials_forward,
            mesh_recon=mesh_recon,
            electrode_markers_recon=electrode_markers_recon,
            materials_recon=materials_recon,
            U_measured=U_measured_noisy,
            sigma_baseline=sigma_baseline,
            alpha=alpha,
            max_iterations=max_iterations,
            tolerance=1e-4,
            verbose=True
        )

        # Save results
        np.savez(
            alpha_dir / "reconstruction.npz",
            sigma_reconstructed=sigma_recon,
            sigma_true=sigma_true,
            sigma_baseline=sigma_baseline,
            alpha=alpha,
            history=history,
            sample_name=sample_name,
            stroke_type=str(sample_data['stroke_type'])
        )

        # Plot
        plot_reconstruction_results(
            mesh_true=mesh_forward,
            sigma_true=sigma_true,
            mesh_recon=mesh_recon,
            sigma_recon=sigma_recon,
            sigma_baseline=sigma_baseline,
            history=history,
            title=f"{sample_basename}\nalpha={alpha:.1e}",
            save_path=alpha_dir / "reconstruction.png"
        )

    print(f"\n✓ Results saved to: {output_base}")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    print("="*70)
    print("SMOOTH RECONSTRUCTION FOR SUBJECT 00")
    print("="*70)

    # 3-layer reconstructions
    print("\n\n")
    print("#"*70)
    print("# 3-LAYER RECONSTRUCTIONS")
    print("#"*70)

    reconstruct_sample(
        subject_id="subject_00",
        layer_type="3layer",
        sample_name="subject_00_3layer_sample_1_ischemic.npz",
        alphas=[1e-8, 1e-7, 1e-6]
    )

    # 6-layer reconstructions
    print("\n\n")
    print("#"*70)
    print("# 6-LAYER RECONSTRUCTIONS")
    print("#"*70)

    reconstruct_sample(
        subject_id="subject_00",
        layer_type="6layer",
        sample_name="subject_00_6layer_sample_2_ischemic.npz",
        alphas=[1e-8, 1e-7, 1e-6]
    )

    print("\n" + "="*70)
    print("ALL RECONSTRUCTIONS COMPLETE")
    print("="*70)
    print("\nResults saved to:")
    print("  reconstruction_results/")
    print("    subject_00/")
    print("      3layer/")
    print("        subject_00_3layer_sample_1_ischemic/")
    print("          alpha_1e-08/")
    print("            reconstruction.npz")
    print("            reconstruction.png")
    print("          alpha_1e-07/")
    print("            ...")
    print("      6layer/")
    print("        subject_00_6layer_sample_2_ischemic/")
    print("          ...")
