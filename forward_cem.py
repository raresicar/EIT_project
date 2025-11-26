"""
Complete EIT Forward Problem Solver
- Electrodes with contact impedances (Complete Electrode Model)
- Current injection patterns
- Voltage measurements
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

print("Importing scikit-fem...")
try:
    from skfem import MeshTri, Basis, ElementTriP1
    from skfem.helpers import grad, dot
    from skfem import asm, condense, solve
    print("✓ scikit-fem imported successfully")
except ImportError as e:
    print(f"✗ Error: {e}")
    print("Install with: pip install scikit-fem[all]")
    exit(1)


# =============================================================================
# CONDUCTIVITY VALUES
# =============================================================================

def get_conductivity_map():
    """Conductivity at 100 kHz (S/m) from Paldanius et al."""
    return {
        0: 0.01,   # Background
        1: 0.36,   # Scalp
        2: 0.02,   # Skull
        3: 2.0,    # CSF
        4: 0.11,   # Grey matter
        5: 0.08,   # White matter
        6: 2.0,    # Ventricles
        7: 0.70,   # Hemorrhage (for perturbation studies)
    }


# =============================================================================
# ELECTRODE PLACEMENT
# =============================================================================

def place_electrodes_on_boundary(mesh, n_electrodes=32):
    """
    Place electrodes uniformly around the boundary
    
    Returns:
        electrode_nodes: List of arrays, each containing node indices for one electrode
        electrode_centers: (n_electrodes, 2) array of electrode center coordinates
    """
    print(f"\nPlacing {n_electrodes} electrodes on boundary...")
    
    # Get boundary nodes
    boundary_facets = mesh.facets[:, mesh.boundary_facets()]
    boundary_nodes = np.unique(boundary_facets.flatten())
    
    # Get boundary node coordinates
    boundary_coords = mesh.p[:, boundary_nodes]
    
    # Calculate angles from center
    center = mesh.p.mean(axis=1)
    angles = np.arctan2(boundary_coords[1] - center[1], 
                       boundary_coords[0] - center[0])
    
    # Sort by angle
    sorted_indices = np.argsort(angles)
    boundary_nodes_sorted = boundary_nodes[sorted_indices]
    angles_sorted = angles[sorted_indices]
    
    # Divide boundary into electrode segments
    n_boundary = len(boundary_nodes_sorted)
    nodes_per_electrode = n_boundary // n_electrodes
    
    electrode_nodes = []
    electrode_centers = []
    
    for i in range(n_electrodes):
        start_idx = i * nodes_per_electrode
        end_idx = start_idx + nodes_per_electrode if i < n_electrodes - 1 else n_boundary
        
        elec_nodes = boundary_nodes_sorted[start_idx:end_idx]
        electrode_nodes.append(elec_nodes)
        
        # Calculate electrode center
        elec_coords = mesh.p[:, elec_nodes]
        center_coord = elec_coords.mean(axis=1)
        electrode_centers.append(center_coord)
    
    electrode_centers = np.array(electrode_centers)
    
    print(f"✓ Placed {n_electrodes} electrodes")
    print(f"  Nodes per electrode: min={min(len(e) for e in electrode_nodes)}, "
          f"max={max(len(e) for e in electrode_nodes)}")
    
    return electrode_nodes, electrode_centers


# =============================================================================
# COMPLETE ELECTRODE MODEL (CEM)
# =============================================================================

def assemble_cem_system(basis, sigma_values, electrode_nodes, contact_impedance=1e-4):
    """
    Assemble Complete Electrode Model system
    
    The CEM equations are:
        ∇·(σ∇u) = 0                           in Ω
        σ ∂u/∂n + (u - U_l)/z_l = 0           on electrode l
        ∫_electrode_l (u - U_l)/z_l dS = I_l  for each electrode
    
    Args:
        basis: FEM basis
        sigma_values: Conductivity array for each element
        electrode_nodes: List of node arrays for each electrode
        contact_impedance: Contact impedance z_l (Ω·m²)
    
    Returns:
        K: Stiffness matrix (including CEM terms)
        n_dofs: Number of DOFs (nodes + electrodes)
        n_nodes: Number of nodes
        n_electrodes: Number of electrodes
    """
    print("\nAssembling CEM system...")
    
    mesh = basis.mesh
    n_nodes = mesh.p.shape[1]
    n_electrodes = len(electrode_nodes)
    n_dofs = n_nodes + n_electrodes

    from skfem import LinearForm
    
    # Standard stiffness matrix: ∫ σ ∇u·∇v dx
    @LinearForm
    def stiffness_form(u, v, w):
        return dot(w['sigma'] * grad(u), grad(v))
    
    # Assemble using bilinear form
    from skfem import BilinearForm
    
    @BilinearForm
    def laplacian(u, v, w):
        return dot(w['sigma'] * grad(u), grad(v))
    
    K_interior = asm(laplacian, basis, sigma=sigma_values)
    
    # Extend matrix for electrode potentials
    K = np.zeros((n_dofs, n_dofs))
    K[:n_nodes, :n_nodes] = K_interior.toarray()
    
    # Add CEM boundary terms for each electrode
    for elec_idx, elec_nodes in enumerate(electrode_nodes):
        # Electrode DOF index
        U_idx = n_nodes + elec_idx
        
        # Calculate electrode area (sum of boundary edge lengths)
        electrode_area = 0.0
        for i in range(len(elec_nodes)):
            n1 = elec_nodes[i]
            n2 = elec_nodes[(i + 1) % len(elec_nodes)]
            edge_length = np.linalg.norm(mesh.p[:, n1] - mesh.p[:, n2])
            electrode_area += edge_length
        
        # Add CEM terms: 1/(z_l * A_l) terms
        # For each node on electrode: K[i,i] += 1/(z*A), K[i,U] -= 1/(z*A)
        cem_coeff = 1.0 / (contact_impedance * electrode_area)
        
        for node in elec_nodes:
            # Node-to-node coupling
            K[node, node] += cem_coeff
            # Node-to-electrode coupling
            K[node, U_idx] -= cem_coeff
            K[U_idx, node] -= cem_coeff
            # Electrode-to-electrode coupling
            K[U_idx, U_idx] += cem_coeff
    
    print(f"✓ System assembled: {n_dofs} DOFs ({n_nodes} nodes + {n_electrodes} electrodes)")
    
    return csr_matrix(K), n_dofs, n_nodes, n_electrodes


# =============================================================================
# CURRENT PATTERNS
# =============================================================================

def create_current_pattern(n_electrodes, pattern_type='adjacent', pattern_idx=0):
    """
    Create current injection pattern
    
    Args:
        n_electrodes: Number of electrodes
        pattern_type: 'adjacent' or 'opposite'
        pattern_idx: Pattern index (which electrode pair to inject)
    
    Returns:
        current_vector: (n_electrodes,) array of injected currents
    """
    I = np.zeros(n_electrodes)
    
    if pattern_type == 'adjacent':
        # Adjacent pattern: inject at electrode i, remove at i+1
        source = pattern_idx % n_electrodes
        sink = (pattern_idx + 1) % n_electrodes
        I[source] = 1.0   # 1 mA injection
        I[sink] = -1.0    # 1 mA removal
        
    elif pattern_type == 'opposite':
        # Opposite pattern: inject at electrode i, remove at i+n/2
        source = pattern_idx % n_electrodes
        sink = (pattern_idx + n_electrodes // 2) % n_electrodes
        I[source] = 1.0
        I[sink] = -1.0
    
    return I


# =============================================================================
# FORWARD SOLVER
# =============================================================================

def solve_eit_forward(mesh, materials, electrode_nodes, current_pattern, 
                      contact_impedance=1e-4):
    """
    Solve EIT forward problem with given current pattern
    
    Returns:
        u_nodes: Potential at mesh nodes
        U_electrodes: Potential at electrodes
    """
    
    # Create FEM basis
    basis = Basis(mesh, ElementTriP1())
    
    # Map materials to conductivity
    conductivity_map = get_conductivity_map()
    sigma_values = np.array([conductivity_map.get(m, 0.1) for m in materials])
    
    # Assemble system
    K, n_dofs, n_nodes, n_electrodes = assemble_cem_system(
        basis, sigma_values, electrode_nodes, contact_impedance
    )
    
    # Build RHS (current injection at electrodes only)
    f = np.zeros(n_dofs)
    f[n_nodes:] = current_pattern  # Current enters at electrode DOFs
    
    # Ground reference: fix one electrode potential to 0
    # This makes the system solvable
    ground_electrode = 0
    ground_dof = n_nodes + ground_electrode
    
    # Solve with grounding constraint
    u_full = np.zeros(n_dofs)
    
    # Remove ground DOF from system
    active_dofs = np.ones(n_dofs, dtype=bool)
    active_dofs[ground_dof] = False
    
    K_reduced = K[active_dofs][:, active_dofs]
    f_reduced = f[active_dofs]
    
    # Solve
    u_reduced = spsolve(K_reduced, f_reduced)
    
    # Put solution back
    u_full[active_dofs] = u_reduced
    u_full[ground_dof] = 0.0
    
    # Extract node and electrode potentials
    u_nodes = u_full[:n_nodes]
    U_electrodes = u_full[n_nodes:]
    
    return u_nodes, U_electrodes


# =============================================================================
# MEASUREMENT SIMULATION
# =============================================================================

def simulate_measurements(mesh, materials, electrode_nodes, 
                         n_patterns=32, pattern_type='adjacent',
                         contact_impedance=1e-4):
    """
    Simulate complete EIT measurement protocol
    
    Returns:
        voltages: (n_patterns, n_electrodes) array of measured voltages
        patterns: (n_patterns, n_electrodes) array of current patterns
    """
    print(f"\nSimulating {n_patterns} measurement patterns...")
    
    n_electrodes = len(electrode_nodes)
    voltages = np.zeros((n_patterns, n_electrodes))
    patterns = np.zeros((n_patterns, n_electrodes))
    
    for i in range(n_patterns):
        print(f"  Pattern {i+1}/{n_patterns}", end='\r')
        
        # Create current pattern
        I = create_current_pattern(n_electrodes, pattern_type, i)
        patterns[i] = I
        
        # Solve forward problem
        u_nodes, U_electrodes = solve_eit_forward(
            mesh, materials, electrode_nodes, I, contact_impedance
        )
        
        # Store electrode voltages
        voltages[i] = U_electrodes
    
    print(f"\n✓ Simulated {n_patterns} patterns")
    
    return voltages, patterns


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_eit_solution(mesh, materials, u_nodes, electrode_centers, 
                      U_electrodes, current_pattern):
    """Plot EIT solution"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    from matplotlib.tri import Triangulation
    tri = Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
    
    # Plot 1: Conductivity
    conductivity_map = get_conductivity_map()
    sigma_vals = np.array([conductivity_map.get(m, 0.1) for m in materials])
    ax = axes[0]
    im1 = ax.tripcolor(tri, sigma_vals, cmap='viridis', shading='flat')
    ax.plot(electrode_centers[:, 0], electrode_centers[:, 1], 'ro', ms=8, label='Electrodes')
    ax.set_title('Conductivity (S/m)', fontweight='bold', fontsize=14)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_aspect('equal')
    ax.legend()
    plt.colorbar(im1, ax=ax)
    
    # Plot 2: Potential distribution
    ax = axes[1]
    im2 = ax.tripcolor(tri, u_nodes, cmap='RdBu_r', shading='gouraud')
    
    # Mark injection electrodes
    inject_idx = np.where(current_pattern > 0)[0]
    remove_idx = np.where(current_pattern < 0)[0]
    if len(inject_idx) > 0:
        ax.plot(electrode_centers[inject_idx, 0], electrode_centers[inject_idx, 1], 
                'g^', ms=12, label='Current +')
    if len(remove_idx) > 0:
        ax.plot(electrode_centers[remove_idx, 0], electrode_centers[remove_idx, 1], 
                'rv', ms=12, label='Current -')
    
    ax.set_title('Potential Distribution (V)', fontweight='bold', fontsize=14)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_aspect('equal')
    ax.legend()
    plt.colorbar(im2, ax=ax)
    
    # Plot 3: Electrode voltages
    ax = axes[2]
    electrode_numbers = np.arange(len(U_electrodes))
    ax.stem(electrode_numbers, U_electrodes, basefmt=' ')
    ax.set_title('Electrode Voltages', fontweight='bold', fontsize=14)
    ax.set_xlabel('Electrode Number')
    ax.set_ylabel('Voltage (V)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('eit_forward_solution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: eit_forward_solution.png")
    
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("EIT Forward Problem with Complete Electrode Model")
    print("="*70)
    
    # Load mesh
    print("\nLoading mesh...")
    data = np.load('brainweb_meshes_correct.npz')
    points = data['points_3layer']
    cells = data['cells_3layer']
    materials = data['materials_3layer']
    
    if points.shape[1] == 3:
        points = points[:, :2]
    
    # Create scikit-fem mesh
    mesh = MeshTri(points.T, cells.T)
    print(f"✓ Mesh loaded: {mesh.p.shape[1]} nodes, {mesh.t.shape[1]} elements")
    
    # Place electrodes
    n_electrodes = 32
    electrode_nodes, electrode_centers = place_electrodes_on_boundary(mesh, n_electrodes)
    
    # Solve single pattern as example
    print("\nSolving forward problem for pattern 0 (adjacent injection)...")
    I_pattern = create_current_pattern(n_electrodes, 'adjacent', 0)
    
    u_nodes, U_electrodes = solve_eit_forward(
        mesh, materials, electrode_nodes, I_pattern, 
        contact_impedance=1e-4
    )
    
    print(f"✓ Solution obtained")
    print(f"  Node potential range: [{u_nodes.min():.3e}, {u_nodes.max():.3e}] V")
    print(f"  Electrode potential range: [{U_electrodes.min():.3e}, {U_electrodes.max():.3e}] V")
    
    # Plot
    fig = plot_eit_solution(mesh, materials, u_nodes, electrode_centers, 
                           U_electrodes, I_pattern)
    plt.show()
    
    # Simulate full measurement protocol
    print("\n" + "="*70)
    print("Simulating complete measurement protocol...")
    print("="*70)
    
    voltages, patterns = simulate_measurements(
        mesh, materials, electrode_nodes,
        n_patterns=n_electrodes,  # One rotation of adjacent patterns
        pattern_type='adjacent',
        contact_impedance=1e-4
    )
    
    print(f"\n✓ Forward problem complete!")
    print(f"  Measurement matrix shape: {voltages.shape}")
    print(f"  Voltage range: [{voltages.min():.3e}, {voltages.max():.3e}] V")
    
    # Save results
    np.savez('eit_forward_data.npz',
             voltages=voltages,
             patterns=patterns,
             electrode_centers=electrode_centers,
             mesh_points=mesh.p,
             mesh_cells=mesh.t,
             materials=materials)
    print("\n✓ Saved: eit_forward_data.npz")
    
    print("\n" + "="*70)
    print("✓ COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()