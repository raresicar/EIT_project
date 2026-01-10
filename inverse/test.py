"""
Gauss-Newton TV Reconstruction Example
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from pathlib import Path
import time
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


from forward.eit_forward_skfem import EIT, current_method, create_eit_from_npz
from gauss_newton_tv import GaussNewtonSolverTV


def create_eit_from_npz(mesh_npz_path, n_electrodes=16, injection_method=2, z=0.01):
    """
    Create EIT solver from NPZ mesh file.
    
    Returns:
        solver, mesh, materials
    """
    from skfem import MeshTri
    from forward.eit_forward_skfem import load_brainweb_mesh
    
    # Load mesh with electrodes
    mesh, electrode_markers, materials = load_brainweb_mesh(
        mesh_npz_path, 
        n_electrodes=n_electrodes
    )
    
    # Create injection pattern
    L = n_electrodes
    Inj = current_method(L=L, l=L-1, method=injection_method, value=1.0)
    
    # Create solver
    solver = EIT(L=L, Inj=Inj, z=z, mesh=mesh, electrode_markers=electrode_markers)
    
    return solver, mesh, materials


def reconstruct_stroke_sample(
    sample_dir='/mnt/d/Programming/EIT/brainweb_strokes/3layer/sample_01_ischemic',
    noise_level=0.01,
    num_steps=8,
    lamb=0.5,
    beta=1e-6
):
    """Run Gauss-Newton TV reconstruction."""
    
    print("="*70)
    print("Gauss-Newton TV Reconstruction")
    print("="*70)
    
    sample_dir = Path(sample_dir)
    
    # ========================================================================
    # LOAD DATA
    # ========================================================================
    print(f"\nLoading: {sample_dir.name}")
    
    data_fwd = np.load(sample_dir / "mesh_forward.npz")
    data_inv = np.load(sample_dir / "mesh_inverse.npz")
    
    sigma_true = data_fwd['conductivity']
    sigma_inv_true = data_inv['conductivity']
    materials_inv = data_inv['materials']
    stroke_type = str(data_fwd['stroke_type'])
    
    print(f"  Stroke: {stroke_type}")
    print(f"  True σ: [{sigma_true.min():.4f}, {sigma_true.max():.4f}] S/m")
    
    # ========================================================================
    # CREATE SOLVERS
    # ========================================================================
    print("\nCreating EIT solvers...")
    
    solver_fwd, mesh_fwd, _ = create_eit_from_npz(sample_dir / "mesh_forward.npz")
    solver_inv, mesh_inv, _ = create_eit_from_npz(sample_dir / "mesh_inverse.npz")
    
    print(f"  Forward: {solver_fwd.dofs:,} DOFs")
    print(f"  Inverse: {solver_inv.dofs:,} DOFs")
    
    # ========================================================================
    # GENERATE MEASUREMENTS
    # ========================================================================
    print("\nGenerating synthetic measurements...")
    
    u_all_fwd, U_all_fwd = solver_fwd.forward_solve(sigma_true)
    U_true = np.array(U_all_fwd)
    
    # Add heteroscedastic noise
    np.random.seed(42)
    delta1, delta2 = noise_level, 0.01 * noise_level
    var_meas = (delta1 * np.abs(U_true) + delta2 * np.max(np.abs(U_true)))**2
    noise = np.random.randn(*U_true.shape) * np.sqrt(var_meas)
    U_meas = U_true + noise
    
    SNR = 20 * np.log10(np.linalg.norm(U_true) / np.linalg.norm(noise))
    print(f"  Noise: {noise_level*100:.1f}%, SNR: {SNR:.1f} dB")
    
    # ========================================================================
    # SETUP GAUSS-NEWTON
    # ========================================================================
    print("\nSetting up Gauss-Newton...")
    
    GammaInv = 1.0 / np.maximum(var_meas.flatten(), 1e-10)
    
    gauss_newton = GaussNewtonSolverTV(
        eit_solver=solver_inv,
        GammaInv=GammaInv,
        num_steps=num_steps,
        lamb=lamb,
        beta=beta,
        clip=[0.001, 3.0]
    )
    
    # Initial guess (mean scalp conductivity)
    backCond = sigma_inv_true[materials_inv == 1].mean()
    print(f"  λ={lamb}, β={beta}, iterations={num_steps}")
    print(f"  Initial σ: {backCond:.4f} S/m")
    
    # ========================================================================
    # RECONSTRUCT
    # ========================================================================
    print("\n" + "="*70)
    print("RECONSTRUCTING")
    print("="*70)
    
    t0 = time.time()
    
    sigma_reco = gauss_newton.forward(
        Umeas=U_meas.flatten(),
        sigma_init=np.ones(len(sigma_inv_true)) * backCond,
        verbose=True
    )
    
    elapsed = time.time() - t0
    
    print(f"\n✓ Reconstruction completed in {elapsed:.2f} s")
    print(f"  Reconstructed: [{sigma_reco.min():.4f}, {sigma_reco.max():.4f}] S/m")
    
    # ========================================================================
    # EVALUATE
    # ========================================================================
    rel_error = np.linalg.norm(sigma_reco - sigma_inv_true) / np.linalg.norm(sigma_inv_true)
    print(f"  Relative error: {rel_error:.2%}")
    
    # ========================================================================
    # VISUALIZE
    # ========================================================================
    print("\nVisualizing...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    tri = Triangulation(mesh_inv.p[0], mesh_inv.p[1], mesh_inv.t.T)
    vmin, vmax = 0.01, 2.0
    
    # True
    ax = axes[0]
    im = ax.tripcolor(tri, sigma_inv_true, cmap='jet', shading='flat', vmin=vmin, vmax=vmax)
    ax.set_aspect('equal')
    ax.set_title(f'True Conductivity\n{stroke_type.capitalize()} Stroke', fontweight='bold', fontsize=14)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='σ (S/m)')
    
    # Reconstructed
    ax = axes[1]
    im = ax.tripcolor(tri, sigma_reco, cmap='jet', shading='flat', vmin=vmin, vmax=vmax)
    ax.set_aspect('equal')
    ax.set_title(f'Gauss-Newton (TV)\nλ={lamb}, {num_steps} iterations', fontweight='bold', fontsize=14)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='σ (S/m)')
    
    # Error
    ax = axes[2]
    error = np.abs(sigma_reco - sigma_inv_true)
    im = ax.tripcolor(tri, error, cmap='Reds', shading='flat')
    ax.set_aspect('equal')
    ax.set_title(f'Absolute Error\nRelative: {rel_error:.1%}', fontweight='bold', fontsize=14)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='|Δσ| (S/m)')
    
    plt.tight_layout()
    
    output_path = sample_dir / "reconstruction.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.show()
    
    print("\n" + "="*70)
    print("✓ COMPLETE!")
    print("="*70)
    
    return sigma_reco, sigma_inv_true, solver_inv


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Gauss-Newton TV reconstruction")
    parser.add_argument('--sample', default='/mnt/d/Programming/EIT/brainweb_strokes/3layer/sample_01_ischemic',
                       help='Sample directory')
    parser.add_argument('--noise', type=float, default=0.01,
                       help='Noise level (default: 1%%)')
    parser.add_argument('--lamb', type=float, default=0.5,
                       help='TV regularization parameter')
    parser.add_argument('--iterations', type=int, default=8,
                       help='Gauss-Newton iterations')
    
    args = parser.parse_args()
    
    reconstruct_stroke_sample(
        sample_dir=args.sample,
        noise_level=args.noise,
        lamb=args.lamb,
        num_steps=args.iterations
    )