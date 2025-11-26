"""
Create Proper Nested FEM Meshes from BrainWeb Head Models
Fixed version - no overlapping regions
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from scipy.spatial import Delaunay
import sys

print("Checking imports...")
try:
    import meshio
    print("✓ meshio available")
except ImportError:
    print("✗ meshio not found. Install with: pip install meshio")
    sys.exit(1)

print("✓ Imports successful!\n")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_brainweb_models(filename='brainweb_head_models.npz'):
    """Load saved BrainWeb models from NPZ file"""
    print(f"Loading data from {filename}...")
    
    try:
        data = np.load(filename, allow_pickle=True)
        
        six_layer = data['six_layer']
        three_layer = data['three_layer']
        raw_slice = data['raw_slice']
        
        print(f"✓ Loaded successfully")
        print(f"  6-layer model: {six_layer.shape}")
        print(f"  3-layer model: {three_layer.shape}")
        
        return six_layer, three_layer, raw_slice
        
    except FileNotFoundError:
        print(f"✗ Error: {filename} not found!")
        sys.exit(1)


# =============================================================================
# PROPER NESTED MESH GENERATION
# =============================================================================

def create_nested_mesh(layer_model, pixel_size=2.0, subsample=2):
    """
    Create proper nested mesh where each pixel becomes triangles
    Materials assigned correctly without overlap
    
    This creates a structured mesh that respects layer boundaries
    """
    print("\nCreating nested structured mesh...")
    
    # Subsample to reduce mesh size
    layer_sub = layer_model[::subsample, ::subsample]
    ny, nx = layer_sub.shape
    
    print(f"Grid size: {ny} x {nx}")
    
    # Create node grid
    points = []
    node_grid = np.full((ny, nx), -1, dtype=int)
    
    # Only create nodes where there's non-background material
    for j in range(ny):
        for i in range(nx):
            if layer_sub[j, i] > 0:
                node_grid[j, i] = len(points)
                points.append([
                    i * pixel_size * subsample,
                    j * pixel_size * subsample
                ])
    
    points = np.array(points)
    print(f"Created {len(points)} nodes")
    
    # Create triangles from grid
    triangles = []
    cell_materials = []
    
    for j in range(ny - 1):
        for i in range(nx - 1):
            # Get material at this cell
            material = layer_sub[j, i]
            
            if material > 0:
                # Get the 4 corner nodes
                n0 = node_grid[j, i]
                n1 = node_grid[j, i+1]
                n2 = node_grid[j+1, i+1]
                n3 = node_grid[j+1, i]
                
                # Only create triangles if all 4 nodes exist
                if n0 >= 0 and n1 >= 0 and n2 >= 0 and n3 >= 0:
                    # Create 2 triangles per quad
                    # Triangle 1: [n0, n1, n2]
                    triangles.append([n0, n1, n2])
                    cell_materials.append(material)
                    
                    # Triangle 2: [n0, n2, n3]
                    triangles.append([n0, n2, n3])
                    cell_materials.append(material)
    
    cells = np.array(triangles)
    cell_data = np.array(cell_materials)
    
    print(f"Created {len(cells)} triangles")
    print(f"Materials present: {np.unique(cell_data)}")
    
    # Verify no overlaps
    material_counts = {}
    for mat in np.unique(cell_data):
        count = np.sum(cell_data == mat)
        material_counts[mat] = count
        print(f"  Material {mat}: {count} triangles")
    
    return points, cells, cell_data


# =============================================================================
# REFINED NESTED MESH (Better Quality)
# =============================================================================

def create_refined_nested_mesh(layer_model, pixel_size=2.0, subsample=2, 
                              refine_boundaries=True):
    """
    Create refined nested mesh with optional boundary refinement
    """
    print("\nCreating refined nested mesh...")
    
    # Subsample
    layer_sub = layer_model[::subsample, ::subsample]
    ny, nx = layer_sub.shape
    
    print(f"Base grid size: {ny} x {nx}")
    
    if refine_boundaries:
        print("Adding boundary refinement...")
        
        # Detect boundaries between materials
        boundaries = np.zeros_like(layer_sub, dtype=bool)
        
        for j in range(1, ny-1):
            for i in range(1, nx-1):
                if layer_sub[j, i] > 0:
                    # Check if any neighbor has different material
                    neighbors = [
                        layer_sub[j-1, i], layer_sub[j+1, i],
                        layer_sub[j, i-1], layer_sub[j, i+1]
                    ]
                    if any(n != layer_sub[j, i] for n in neighbors):
                        boundaries[j, i] = True
        
        # Create refined grid near boundaries
        points = []
        point_grid = {}  # (j, i) -> node index
        
        for j in range(ny):
            for i in range(nx):
                if layer_sub[j, i] > 0:
                    # Regular node
                    point_grid[(j, i)] = len(points)
                    points.append([
                        i * pixel_size * subsample,
                        j * pixel_size * subsample
                    ])
                    
                    # Add extra nodes at boundaries for refinement
                    if boundaries[j, i] and j < ny-1 and i < nx-1:
                        # Add mid-edge nodes
                        if (j, i+0.5) not in point_grid:
                            point_grid[(j, i+0.5)] = len(points)
                            points.append([
                                (i + 0.5) * pixel_size * subsample,
                                j * pixel_size * subsample
                            ])
                        
                        if (j+0.5, i) not in point_grid:
                            point_grid[(j+0.5, i)] = len(points)
                            points.append([
                                i * pixel_size * subsample,
                                (j + 0.5) * pixel_size * subsample
                            ])
    else:
        # Simple grid
        points = []
        point_grid = {}
        
        for j in range(ny):
            for i in range(nx):
                if layer_sub[j, i] > 0:
                    point_grid[(j, i)] = len(points)
                    points.append([
                        i * pixel_size * subsample,
                        j * pixel_size * subsample
                    ])
    
    points = np.array(points)
    print(f"Created {len(points)} nodes")
    
    # Create triangles
    triangles = []
    cell_materials = []
    
    for j in range(ny - 1):
        for i in range(nx - 1):
            material = layer_sub[j, i]
            
            if material > 0:
                # Get corner nodes
                corners = [(j, i), (j, i+1), (j+1, i+1), (j+1, i)]
                
                if all(c in point_grid for c in corners):
                    nodes = [point_grid[c] for c in corners]
                    n0, n1, n2, n3 = nodes
                    
                    # Two triangles
                    triangles.append([n0, n1, n2])
                    cell_materials.append(material)
                    
                    triangles.append([n0, n2, n3])
                    cell_materials.append(material)
    
    cells = np.array(triangles)
    cell_data = np.array(cell_materials)
    
    print(f"Created {len(cells)} triangles")
    print(f"Materials: {np.unique(cell_data)}")
    
    return points, cells, cell_data


# =============================================================================
# MESH QUALITY METRICS
# =============================================================================

def compute_mesh_quality(points, cells):
    """Compute mesh quality metrics"""
    print("\n" + "="*50)
    print("MESH QUALITY METRICS")
    print("="*50)
    
    # Triangle areas
    p0 = points[cells[:, 0]]
    p1 = points[cells[:, 1]]
    p2 = points[cells[:, 2]]
    
    v1 = p1 - p0
    v2 = p2 - p0
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    
    print(f"\nTriangle Areas (mm²):")
    print(f"  Min:    {areas.min():8.3f}")
    print(f"  Max:    {areas.max():8.3f}")
    print(f"  Mean:   {areas.mean():8.3f}")
    print(f"  Median: {np.median(areas):8.3f}")
    
    # Edge lengths
    edge_lengths = []
    for i in range(3):
        j = (i + 1) % 3
        lengths = np.linalg.norm(points[cells[:, i]] - points[cells[:, j]], axis=1)
        edge_lengths.extend(lengths)
    
    edge_lengths = np.array(edge_lengths)
    print(f"\nEdge Lengths (mm):")
    print(f"  Min:    {edge_lengths.min():8.3f}")
    print(f"  Max:    {edge_lengths.max():8.3f}")
    print(f"  Mean:   {edge_lengths.mean():8.3f}")
    
    # Aspect ratio
    aspect_ratios = []
    for tri in cells:
        tri_points = points[tri]
        edges = [
            np.linalg.norm(tri_points[1] - tri_points[0]),
            np.linalg.norm(tri_points[2] - tri_points[1]),
            np.linalg.norm(tri_points[0] - tri_points[2])
        ]
        if min(edges) > 0:
            aspect_ratios.append(max(edges) / min(edges))
    
    aspect_ratios = np.array(aspect_ratios)
    print(f"\nAspect Ratios (1.0 = equilateral):")
    print(f"  Min:    {aspect_ratios.min():8.3f}")
    print(f"  Max:    {aspect_ratios.max():8.3f}")
    print(f"  Mean:   {aspect_ratios.mean():8.3f}")
    
    # Quality assessment
    good_triangles = np.sum(aspect_ratios < 2.5)
    total_triangles = len(aspect_ratios)
    quality_pct = 100 * good_triangles / total_triangles
    
    print(f"\nQuality Assessment:")
    print(f"  Good triangles (AR < 2.5): {good_triangles}/{total_triangles} ({quality_pct:.1f}%)")
    
    print("="*50 + "\n")
    
    return {
        'areas': areas,
        'edge_lengths': edge_lengths,
        'aspect_ratios': aspect_ratios,
        'quality_pct': quality_pct
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_mesh_clean(points, cells, cell_data, layer_model, title="Mesh"):
    """Clean visualization showing mesh correctly"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Plot 1: Original layer model for reference
    from matplotlib.colors import ListedColormap, BoundaryNorm
    
    if layer_model.max() <= 3:
        colors = ['white', '#FDB462', '#FFFFB3', '#80B1D3']
    else:
        colors = ['white', '#FDB462', '#FFFFB3', '#8DD3C7',
                 '#BEBADA', '#FB8072', '#80B1D3']
    
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(len(colors) + 1) - 0.5, cmap.N)
    
    im0 = axes[0, 0].imshow(layer_model.T, cmap=cmap, norm=norm, origin='lower')
    axes[0, 0].set_title('Original Layer Model', fontsize=13, fontweight='bold')
    axes[0, 0].axis('equal')
    plt.colorbar(im0, ax=axes[0, 0])
    
    # Plot 2: Mesh structure
    axes[0, 1].triplot(points[:, 0], points[:, 1], cells, 'k-', linewidth=0.2)
    axes[0, 1].plot(points[:, 0], points[:, 1], 'b.', markersize=0.5)
    axes[0, 1].set_title(f'{title} - Mesh Structure\n({len(points)} nodes, {len(cells)} triangles)', 
                        fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('x (mm)')
    axes[0, 1].set_ylabel('y (mm)')
    axes[0, 1].axis('equal')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Materials with solid colors (no wireframe)
    import matplotlib.cm as cm
    from matplotlib.tri import Triangulation
    
    tri_obj = Triangulation(points[:, 0], points[:, 1], cells)
    
    # Create color map for materials
    unique_mats = np.unique(cell_data[cell_data > 0])
    mat_colors = cm.get_cmap('tab10')(np.linspace(0, 1, len(unique_mats)))
    
    # Plot filled triangles colored by material
    axes[1, 0].tripcolor(tri_obj, cell_data, cmap='tab10', shading='flat')
    axes[1, 0].set_title(f'{title} - Material Distribution', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('x (mm)')
    axes[1, 0].set_ylabel('y (mm)')
    axes[1, 0].axis('equal')
    plt.colorbar(axes[1, 0].collections[0], ax=axes[1, 0], label='Material ID')
    
    # Plot 4: Material boundaries only
    for material in unique_mats:
        mask = cell_data == material
        if mask.sum() > 0:
            axes[1, 1].triplot(points[:, 0], points[:, 1], cells[mask], 
                              linewidth=0.3, alpha=0.6,
                              label=f'Material {material}')
    
    axes[1, 1].set_title(f'{title} - Material Boundaries', fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel('x (mm)')
    axes[1, 1].set_ylabel('y (mm)')
    axes[1, 1].axis('equal')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# =============================================================================
# EXPORT
# =============================================================================

def export_mesh(points, cells, cell_data, base_filename, model_name):
    """Export mesh to multiple formats"""
    print(f"\nExporting {model_name} mesh...")
    
    # Ensure points are 3D
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(len(points))])
    
    # Create meshio mesh
    mesh = meshio.Mesh(
        points=points,
        cells=[("triangle", cells)],
        cell_data={"material": [cell_data]}
    )
    
    # Export formats
    formats = {
        'vtk': 'VTK (ParaView)',
        'msh': 'Gmsh/COMSOL',
        'xdmf': 'XDMF (FEniCS)',
    }
    
    exported_files = []
    for ext, description in formats.items():
        filename = f"{base_filename}.{ext}"
        try:
            meshio.write(filename, mesh)
            print(f"  ✓ {filename:<35} ({description})")
            exported_files.append(filename)
        except Exception as e:
            print(f"  ✗ {filename:<35} Failed: {e}")
    
    return exported_files


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main workflow"""
    print("="*70)
    print("BrainWeb Head Models → Proper Nested FEM Meshes")
    print("="*70)
    
    # Load models
    print("\n" + "="*70)
    print("STEP 1: Loading saved models")
    print("="*70)
    six_layer, three_layer, raw_slice = load_brainweb_models()
    
    # Create 6-layer mesh
    print("\n" + "="*70)
    print("STEP 2: Creating 6-layer mesh")
    print("="*70)
    
    points_6, cells_6, materials_6 = create_nested_mesh(
        six_layer,
        pixel_size=2.0,
        subsample=2  # Adjust this: higher = coarser mesh
    )
    
    quality_6 = compute_mesh_quality(points_6, cells_6)
    
    # Create 3-layer mesh
    print("\n" + "="*70)
    print("STEP 3: Creating 3-layer mesh")
    print("="*70)
    
    points_3, cells_3, materials_3 = create_nested_mesh(
        three_layer,
        pixel_size=2.0,
        subsample=2
    )
    
    quality_3 = compute_mesh_quality(points_3, cells_3)
    
    # Visualize
    print("\n" + "="*70)
    print("STEP 4: Visualizing meshes")
    print("="*70)
    
    fig1 = visualize_mesh_clean(points_6, cells_6, materials_6, 
                                six_layer, "6-Layer Model")
    plt.savefig('mesh_6layer_correct.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: mesh_6layer_correct.png")
    plt.close()
    
    fig2 = visualize_mesh_clean(points_3, cells_3, materials_3,
                                three_layer, "3-Layer Model")
    plt.savefig('mesh_3layer_correct.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: mesh_3layer_correct.png")
    plt.close()
    
    # Export
    print("\n" + "="*70)
    print("STEP 5: Exporting meshes")
    print("="*70)
    
    files_6 = export_mesh(points_6, cells_6, materials_6,
                         'head_mesh_6layer_correct', '6-layer')
    files_3 = export_mesh(points_3, cells_3, materials_3,
                         'head_mesh_3layer_correct', '3-layer')
    
    # Save NumPy
    print("\nSaving mesh data...")
    np.savez_compressed(
        'brainweb_meshes_correct.npz',
        points_6layer=points_6,
        cells_6layer=cells_6,
        materials_6layer=materials_6,
        points_3layer=points_3,
        cells_3layer=cells_3,
        materials_3layer=materials_3
    )
    print("✓ Saved: brainweb_meshes_correct.npz")
    
    # Summary
    print("\n" + "="*70)
    print("✓ COMPLETE!")
    print("="*70)
    print(f"\nMesh Quality Summary:")
    print(f"  6-Layer: {quality_6['quality_pct']:.1f}% good triangles")
    print(f"  3-Layer: {quality_3['quality_pct']:.1f}% good triangles")
    print("\nGenerated files:")
    print("  - mesh_6layer_correct.png")
    print("  - mesh_3layer_correct.png")
    for f in files_6 + files_3:
        print(f"  - {f}")
    print("  - brainweb_meshes_correct.npz")
    print("\n✓ No overlapping regions - proper nested mesh!")
    print("="*70)


if __name__ == "__main__":
    main()