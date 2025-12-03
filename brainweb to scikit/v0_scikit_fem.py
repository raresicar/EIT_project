"""
Test loading BrainWeb mesh into scikit-fem - Clean visualization
"""

import numpy as np
import matplotlib.pyplot as plt

print("Importing scikit-fem...")
try:
    from skfem import MeshTri
    print("✓ scikit-fem imported successfully")
except ImportError as e:
    print(f"✗ Error importing scikit-fem: {e}")
    print("Install with: pip install scikit-fem[all]")
    exit(1)


def load_and_visualize(npz_file='brainweb_meshes_correct.npz', model='3layer'):
    """Load mesh from NumPy and create scikit-fem mesh"""
    
    print(f"\nLoading {model} mesh from {npz_file}...")
    data = np.load(npz_file)
    
    if model == '6layer':
        points = data['points_6layer']
        cells = data['cells_6layer']
        materials = data['materials_6layer']
    else:
        points = data['points_3layer']
        cells = data['cells_3layer']
        materials = data['materials_3layer']
    
    print(f"  Nodes: {len(points)}")
    print(f"  Triangles: {len(cells)}")
    print(f"  Materials: {np.unique(materials)}")
    
    # Extract 2D coordinates if needed
    if points.shape[1] == 3:
        points_2d = points[:, :2]
    else:
        points_2d = points
    
    print(f"\nCreating scikit-fem mesh...")
    print(f"  Points shape: {points_2d.shape}")
    print(f"  Cells shape: {cells.shape}")
    
    # Create mesh - scikit-fem uses column format
    mesh = MeshTri(
        points_2d.T,      # (N, 2) -> (2, N)
        cells.T           # (M, 3) -> (3, M)
    )
    
    print(f"✓ Mesh created successfully!")
    print(f"  Number of vertices: {mesh.p.shape[1]}")
    print(f"  Number of elements: {mesh.t.shape[1]}")
    print(f"  Boundary edges: {mesh.facets.shape[1]}")
    
    # Validate mesh
    print(f"\nMesh validation:")
    print(f"  Points range: x=[{mesh.p[0].min():.1f}, {mesh.p[0].max():.1f}], "
          f"y=[{mesh.p[1].min():.1f}, {mesh.p[1].max():.1f}]")
    print(f"  Elements range: [{mesh.t.min()}, {mesh.t.max()}]")
    
    # Check triangle areas
    areas = []
    for i in range(min(1000, mesh.t.shape[1])):
        tri = mesh.t[:, i]
        p0, p1, p2 = mesh.p[:, tri[0]], mesh.p[:, tri[1]], mesh.p[:, tri[2]]
        area = 0.5 * abs((p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1]))
        areas.append(area)
    
    print(f"  Triangle areas (first 1000):")
    print(f"    min={min(areas):.2e}, max={max(areas):.2e}, mean={np.mean(areas):.2e}")
    
    if min(areas) < 1e-10:
        print(f"  ⚠️ WARNING: Found near-degenerate triangles (area < 1e-10)")
    
    # Plot the mesh
    print("\nPlotting...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    from matplotlib.tri import Triangulation
    tri = Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
    
    # Plot 1: Wireframe (zoomed out)
    ax = axes[0, 0]
    ax.triplot(tri, 'k-', linewidth=0.2, alpha=0.5)
    ax.set_title('Full Mesh - Wireframe', fontweight='bold', fontsize=14)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Zoomed in region to see triangles clearly
    ax = axes[0, 1]
    ax.triplot(tri, 'k-', linewidth=0.5)
    ax.set_title('Zoomed View - Triangle Mesh', fontweight='bold', fontsize=14)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_aspect('equal')
    # Zoom to a small region
    cx, cy = mesh.p[0].mean(), mesh.p[1].mean()
    zoom = 50
    ax.set_xlim(cx - zoom, cx + zoom)
    ax.set_ylim(cy - zoom, cy + zoom)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Filled triangles
    ax = axes[1, 0]
    ax.tripcolor(tri, np.ones(mesh.t.shape[1]), cmap='Blues', 
                 edgecolors='k', linewidth=0.05, alpha=0.7)
    ax.set_title('Filled Mesh', fontweight='bold', fontsize=14)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_aspect('equal')
    
    # Plot 4: Material regions
    ax = axes[1, 1]
    im = ax.tripcolor(tri, materials, cmap='tab10', shading='flat', 
                     edgecolors='k', linewidth=0.1, alpha=0.8)
    ax.set_title('Material Regions', fontweight='bold', fontsize=14)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_aspect('equal')
    
    # Add colorbar
    material_names = {1: 'Scalp', 2: 'Skull', 3: 'CSF', 
                     4: 'Grey', 5: 'White', 6: 'Ventricles'}
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Material ID', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('skfem_mesh_test.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: skfem_mesh_test.png")
    
    plt.show()
    
    return mesh, materials


def main():
    print("="*70)
    print("scikit-fem Mesh Loading Test")
    print("="*70)
    
    # Test with 3-layer model
    mesh, materials = load_and_visualize(model='6layer')
    
    print("\n" + "="*70)
    print("✓ SUCCESS! Mesh loaded into scikit-fem")
    print("="*70)
    print("\nThe mesh HAS triangles - the first plot shows boundary wireframe,")
    print("but the zoomed view (top right) shows the actual triangle mesh!")


if __name__ == "__main__":
    main()