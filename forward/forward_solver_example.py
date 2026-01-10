"""
EIT Forward Solver Example

Demonstrates forward solve with BrainWeb mesh.
Prints detailed diagnostics including voltage statistics.
"""

from eit_forward_skfem import (
    EIT, current_method, load_brainweb_mesh, 
    materials_to_conductivity, plot_conductivity
)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import time


def forward_solve_example(
    mesh_path='/mnt/d/Programming/EIT/brainweb_meshes/subject_00/6layer/forward/head_mesh.npz',
    layer_type='6layer',
    n_electrodes=16,
    injection_method=2,
    contact_impedance=0.01,
    visualize=True
):
    """
    Run forward solve example with diagnostics.
    
    Args:
        mesh_path: Path to mesh NPZ file
        layer_type: '3layer' or '6layer'
        n_electrodes: Number of electrodes
        injection_method: Current injection pattern (1-5)
        contact_impedance: Contact impedance (Ohm·m²)
        visualize: Create visualization
    """
    print("="*70)
    print("EIT Forward Solver Example")
    print("="*70)
    
    # Load mesh
    print(f"\nLoading mesh: {mesh_path}")
    mesh, electrode_markers, materials = load_brainweb_mesh(
        mesh_path,
        n_electrodes=n_electrodes
    )
    
    print(f"  Nodes: {mesh.p.shape[1]:,}")
    print(f"  Elements: {mesh.t.shape[1]:,}")
    print(f"  Electrodes: {n_electrodes}")
    
    # Create injection pattern
    L = n_electrodes
    Inj = current_method(L=L, l=L-1, method=injection_method, value=1.0)
    print(f"  Injection patterns: {Inj.shape[0]} (method {injection_method})")
    
    # Create solver
    print("\nInitializing EIT solver...")
    solver = EIT(
        L=L, 
        Inj=Inj, 
        z=contact_impedance, 
        mesh=mesh, 
        electrode_markers=electrode_markers
    )
    
    print(f"  DOFs: {solver.dofs:,}")
    print(f"  System size: {solver.dofs + L + 1:,}")
    print(f"  Ground DOF: {solver.ground_dof}")
    
    # Get conductivity
    sigma = materials_to_conductivity(materials, layer_type)
    print(f"\nConductivity distribution ({layer_type}):")
    print(f"  Range: [{sigma.min():.4f}, {sigma.max():.4f}] S/m")
    print(f"  Unique materials: {len(np.unique(materials))}")
    
    # Forward solve
    print("\n" + "-"*70)
    print("FORWARD SOLVE")
    print("-"*70)
    t0 = time.time()
    u_all, U_all = solver.forward_solve(sigma)
    elapsed = time.time() - t0
    
    print(f"✓ Forward solve completed in {elapsed:.3f} s")
    print(f"  Patterns solved: {len(u_all)}")
    
    # Convert to array
    U = np.array(U_all)
    
    # Voltage statistics
    print("\n" + "-"*70)
    print("VOLTAGE STATISTICS")
    print("-"*70)
    print(f"Electrode voltages shape: {U.shape}")
    print(f"  (patterns × electrodes) = ({U.shape[0]} × {U.shape[1]})")
    
    print(f"\nGlobal statistics:")
    print(f"  Min voltage:  {U.min():.6e} V")
    print(f"  Max voltage:  {U.max():.6e} V")
    print(f"  Mean voltage: {U.mean():.6e} V  (should be ≈0 due to grounding)")
    print(f"  Std voltage:  {U.std():.6e} V")
    
    # Per-pattern statistics
    print(f"\nPer-pattern statistics:")
    mean_per_pattern = U.mean(axis=1)
    print(f"  Mean per pattern range: [{mean_per_pattern.min():.6e}, {mean_per_pattern.max():.6e}]")
    print(f"  Max |mean|: {np.abs(mean_per_pattern).max():.6e}  (grounding check)")
    
    # Check grounding (mean should be ~machine precision)
    max_mean = np.abs(mean_per_pattern).max()
    if max_mean < 1e-10:
        print(f"  ✓ Grounding constraint satisfied ({max_mean:.2e} < 1e-10)")
    else:
        print(f"  ⚠ Grounding constraint may be violated ({max_mean:.2e} > 1e-10)")
    
    # Example pattern
    print(f"\nExample pattern 0:")
    print(f"  Voltages: {U[0]}")
    print(f"  Sum: {U[0].sum():.6e}  (should be ≈0)")
    
    # Jacobian
    print("\n" + "-"*70)
    print("JACOBIAN COMPUTATION")
    print("-"*70)
    t0 = time.time()
    J = solver.calc_jacobian(sigma, u_all)
    elapsed = time.time() - t0
    
    print(f"✓ Jacobian computed in {elapsed:.3f} s")
    print(f"  Jacobian shape: {J.shape}")
    print(f"    (measurements × elements) = ({J.shape[0]} × {J.shape[1]})")
    print(f"  J range: [{J.min():.6e}, {J.max():.6e}]")
    print(f"  J mean: {J.mean():.6e}")
    print(f"  J std: {J.std():.6e}")
    
    # Visualization
    if visualize:
        print("\n" + "-"*70)
        print("CREATING VISUALIZATION")
        print("-"*70)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        tri = Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
        
        # 1. Mesh with electrodes
        ax = axes[0, 0]
        ax.triplot(tri, 'k-', linewidth=0.2, alpha=0.3)
        
        colors = plt.cm.tab20(np.linspace(0, 1, L))
        for i, facets in enumerate(electrode_markers):
            for facet_idx in facets:
                nodes = mesh.facets[:, facet_idx]
                coords = mesh.p[:, nodes]
                ax.plot(coords[0], coords[1], '-', color=colors[i], linewidth=3)
            
            # Label
            mid_facet = facets[len(facets)//2]
            mid_nodes = mesh.facets[:, mid_facet]
            mid = mesh.p[:, mid_nodes].mean(axis=1)
            ax.text(mid[0]*1.1, mid[1]*1.1, str(i+1), 
                   ha='center', va='center', fontsize=8, fontweight='bold')
        
        ax.set_aspect('equal')
        ax.set_title(f'Mesh with {L} Electrodes\n'
                    f'{mesh.p.shape[1]:,} nodes, {mesh.t.shape[1]:,} elements',
                    fontweight='bold')
        ax.axis('off')
        
        # 2. Potential field (first pattern)
        ax = axes[0, 1]
        im = ax.tripcolor(tri, u_all[0], cmap='RdBu_r', shading='gouraud')
        plt.colorbar(im, ax=ax, label='Potential (V)')
        ax.set_aspect('equal')
        ax.set_title('Potential Field (Pattern 0)\n'
                    f'Range: [{u_all[0].min():.3e}, {u_all[0].max():.3e}]',
                    fontweight='bold')
        ax.axis('off')
        
        # 3. Electrode voltages (all patterns)
        ax = axes[1, 0]
        im = ax.imshow(U, aspect='auto', cmap='RdBu_r', interpolation='nearest')
        plt.colorbar(im, ax=ax, label='Voltage (V)')
        ax.set_xlabel('Electrode', fontweight='bold')
        ax.set_ylabel('Pattern', fontweight='bold')
        ax.set_title(f'Electrode Potentials U\n'
                    f'Range: [{U.min():.3e}, {U.max():.3e}], Mean: {U.mean():.3e}',
                    fontweight='bold')
        
        # 4. Jacobian sensitivity
        ax = axes[1, 1]
        sens = np.abs(J).mean(axis=0)
        im = ax.tripcolor(tri, sens, cmap='hot', shading='flat')
        plt.colorbar(im, ax=ax, label='Mean |J|')
        ax.set_aspect('equal')
        ax.set_title('Jacobian Sensitivity\n'
                    f'Range: [{sens.min():.3e}, {sens.max():.3e}]',
                    fontweight='bold')
        ax.axis('off')
        
        plt.tight_layout()
        
        output_path = 'forward_solve_example.png'
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()
    
    print("\n" + "="*70)
    print("✓ Example Complete!")
    print("="*70)
    
    return solver, sigma, u_all, U_all, J


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="EIT forward solver example")
    parser.add_argument('--mesh', default='/mnt/d/Programming/EIT/brainweb_meshes/subject_00/6layer/forward/head_mesh.npz',
                       help='Path to mesh NPZ file')
    parser.add_argument('--layer-type', default='6layer', choices=['3layer', '6layer'],
                       help='Layer type')
    parser.add_argument('--electrodes', type=int, default=16,
                       help='Number of electrodes')
    parser.add_argument('--injection', type=int, default=2, choices=[1, 2, 3, 4, 5],
                       help='Injection pattern method')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization')
    
    args = parser.parse_args()
    
    forward_solve_example(
        mesh_path=args.mesh,
        layer_type=args.layer_type,
        n_electrodes=args.electrodes,
        injection_method=args.injection,
        visualize=not args.no_viz
    )