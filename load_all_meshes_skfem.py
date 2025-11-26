"""
Load and Verify All Stroke Sample Meshes in scikit-fem
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm

print("Importing scikit-fem...")
try:
    from skfem import MeshTri
    print("✓ scikit-fem imported successfully")
except ImportError as e:
    print(f"✗ Error importing scikit-fem: {e}")
    print("Install with: pip install scikit-fem[all]")
    exit(1)


# =============================================================================
# MESH LOADING
# =============================================================================

def load_mesh_to_skfem(npz_path):
    """
    Load mesh from NPZ file into scikit-fem
    
    Args:
        npz_path: Path to mesh NPZ file
    
    Returns:
        mesh: scikit-fem MeshTri object
        materials: Material array
        metadata: Metadata dictionary
    """
    data = np.load(npz_path, allow_pickle=True)
    
    points = data['points']
    cells = data['cells']
    materials = data['materials']
    metadata = data['metadata'].item() if 'metadata' in data else {}
    
    # Extract 2D coordinates if needed
    if points.shape[1] == 3:
        points_2d = points[:, :2]
    else:
        points_2d = points
    
    # Create scikit-fem mesh
    # scikit-fem uses column format: (2, N) for points, (3, M) for cells
    mesh = MeshTri(
        points_2d.T,  # (N, 2) -> (2, N)
        cells.T       # (M, 3) -> (3, M)
    )
    
    return mesh, materials, metadata


# =============================================================================
# MESH VALIDATION
# =============================================================================

def validate_mesh(mesh, materials, mesh_name="mesh"):
    """
    Validate mesh quality
    
    Returns:
        dict with validation results
    """
    # Triangle areas
    areas = []
    for i in range(mesh.t.shape[1]):
        tri = mesh.t[:, i]
        p0, p1, p2 = mesh.p[:, tri[0]], mesh.p[:, tri[1]], mesh.p[:, tri[2]]
        area = 0.5 * abs((p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1]))
        areas.append(area)
    
    areas = np.array(areas)
    
    # Check for degenerate triangles
    degenerate = areas < 1e-10
    n_degenerate = degenerate.sum()
    
    # Material distribution
    unique_materials = np.unique(materials)
    material_counts = {int(m): int((materials == m).sum()) for m in unique_materials}
    
    results = {
        'name': mesh_name,
        'n_vertices': int(mesh.p.shape[1]),
        'n_triangles': int(mesh.t.shape[1]),
        'n_boundary_edges': int(mesh.facets.shape[1]),
        'area_min': float(areas.min()),
        'area_max': float(areas.max()),
        'area_mean': float(areas.mean()),
        'n_degenerate': int(n_degenerate),
        'materials': material_counts,
        'valid': bool(n_degenerate == 0)  # Convert numpy.bool_ to Python bool
    }
    
    return results


# =============================================================================
# BATCH LOADING
# =============================================================================

def load_all_meshes(mesh_dir, max_meshes=None):
    """
    Load all meshes from a directory
    
    Args:
        mesh_dir: Directory containing mesh NPZ files
        max_meshes: Maximum number to load (None = all)
    
    Returns:
        list of dicts with mesh info
    """
    print("="*70)
    print("Loading Stroke Sample Meshes into scikit-fem")
    print("="*70)
    
    mesh_dir = Path(mesh_dir)
    
    if not mesh_dir.exists():
        raise ValueError(f"Mesh directory not found: {mesh_dir}")
    
    # Find all mesh NPZ files
    mesh_files = sorted(mesh_dir.glob("sample_*.npz"))
    
    if len(mesh_files) == 0:
        print(f"\n✗ No mesh files found in {mesh_dir}")
        return []
    
    if max_meshes is not None:
        mesh_files = mesh_files[:max_meshes]
    
    print(f"\nFound {len(mesh_files)} mesh files")
    print(f"Directory: {mesh_dir}\n")
    
    # Load all meshes
    loaded_meshes = []
    validation_results = []
    
    print("Loading meshes...")
    for mesh_path in tqdm(mesh_files, desc="Loading"):
        try:
            # Load mesh
            mesh, materials, metadata = load_mesh_to_skfem(mesh_path)
            
            # Validate
            validation = validate_mesh(mesh, materials, mesh_path.stem)
            
            loaded_meshes.append({
                'path': mesh_path,
                'mesh': mesh,
                'materials': materials,
                'metadata': metadata,
                'validation': validation
            })
            
            validation_results.append(validation)
            
        except Exception as e:
            print(f"\n✗ Failed to load {mesh_path.name}: {e}")
            continue
    
    print(f"\n✓ Successfully loaded {len(loaded_meshes)}/{len(mesh_files)} meshes")
    
    # Summary statistics
    print("\n" + "="*70)
    print("Mesh Statistics")
    print("="*70)
    
    if len(validation_results) > 0:
        n_vertices = [r['n_vertices'] for r in validation_results]
        n_triangles = [r['n_triangles'] for r in validation_results]
        n_degenerate = [r['n_degenerate'] for r in validation_results]
        
        print(f"\nVertices per mesh:")
        print(f"  Min:    {min(n_vertices):,}")
        print(f"  Max:    {max(n_vertices):,}")
        print(f"  Mean:   {np.mean(n_vertices):,.0f}")
        print(f"  Median: {np.median(n_vertices):,.0f}")
        
        print(f"\nTriangles per mesh:")
        print(f"  Min:    {min(n_triangles):,}")
        print(f"  Max:    {max(n_triangles):,}")
        print(f"  Mean:   {np.mean(n_triangles):,.0f}")
        print(f"  Median: {np.median(n_triangles):,.0f}")
        
        print(f"\nMesh quality:")
        n_valid = sum(r['valid'] for r in validation_results)
        print(f"  Valid meshes: {n_valid}/{len(validation_results)}")
        print(f"  Degenerate triangles: {sum(n_degenerate)} total")
        
        if sum(n_degenerate) > 0:
            print(f"  ⚠️ WARNING: Some meshes have degenerate triangles")
        
        # Material distribution across all meshes
        all_materials = set()
        for r in validation_results:
            all_materials.update(r['materials'].keys())
        
        print(f"\nMaterials found across all meshes: {sorted(all_materials)}")
        
        # Stroke type distribution
        stroke_types = {}
        stroke_sizes = {}
        
        for item in loaded_meshes:
            metadata = item['metadata']
            if 'stroke_info' in metadata:
                info = metadata['stroke_info']
                stype = info['type']
                ssize = info['size_category']
                
                stroke_types[stype] = stroke_types.get(stype, 0) + 1
                stroke_sizes[ssize] = stroke_sizes.get(ssize, 0) + 1
        
        if stroke_types:
            print(f"\nStroke types:")
            for stype, count in sorted(stroke_types.items()):
                print(f"  {stype}: {count} ({100*count/len(loaded_meshes):.1f}%)")
        
        if stroke_sizes:
            print(f"\nStroke sizes:")
            for ssize, count in sorted(stroke_sizes.items()):
                print(f"  {ssize}: {count} ({100*count/len(loaded_meshes):.1f}%)")
    
    # Save validation report
    report_path = mesh_dir / "mesh_validation_report.json"
    with open(report_path, 'w') as f:
        json.dump({
            'n_meshes_loaded': len(loaded_meshes),
            'n_meshes_total': len(mesh_files),
            'validation_results': validation_results
        }, f, indent=2)
    
    print(f"\n✓ Saved validation report: {report_path}")
    
    return loaded_meshes


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_sample_meshes(loaded_meshes, output_dir, n_display=9):
    """
    Visualize random sample meshes
    
    Args:
        loaded_meshes: List of loaded mesh dicts
        output_dir: Directory to save visualization
        n_display: Number of meshes to display
    """
    if len(loaded_meshes) == 0:
        return
    
    print(f"\nCreating visualization of {n_display} random meshes...")
    
    # Select random samples
    indices = np.random.choice(len(loaded_meshes), 
                              min(n_display, len(loaded_meshes)), 
                              replace=False)
    
    n_cols = 3
    n_rows = (len(indices) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_display > 1 else [axes]
    
    from matplotlib.tri import Triangulation
    
    for idx, sample_idx in enumerate(indices):
        item = loaded_meshes[sample_idx]
        mesh = item['mesh']
        materials = item['materials']
        metadata = item['metadata']
        
        ax = axes[idx]
        
        # Create triangulation
        tri = Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
        
        # Plot with material colors
        ax.tripcolor(tri, materials, cmap='tab10', shading='flat',
                    edgecolors='k', linewidth=0.1, alpha=0.8)
        
        # Title
        if 'stroke_info' in metadata:
            info = metadata['stroke_info']
            title = f"{item['path'].stem}\n{info['type']} - {info['size_category']}"
        else:
            title = item['path'].stem
        
        validation = item['validation']
        title += f"\n{validation['n_vertices']} nodes, {validation['n_triangles']} tris"
        
        ax.set_title(title, fontsize=9)
        ax.set_aspect('equal')
        ax.axis('off')
    
    # Hide unused axes
    for idx in range(len(indices), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    viz_path = Path(output_dir) / "skfem_loaded_meshes.png"
    plt.savefig(viz_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved visualization: {viz_path}")
    plt.close()


def visualize_detailed_mesh(loaded_mesh, output_path):
    """
    Create detailed visualization of a single mesh
    
    Args:
        loaded_mesh: Dict with mesh info
        output_path: Path to save figure
    """
    mesh = loaded_mesh['mesh']
    materials = loaded_mesh['materials']
    metadata = loaded_mesh['metadata']
    validation = loaded_mesh['validation']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    from matplotlib.tri import Triangulation
    tri = Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
    
    # Plot 1: Wireframe
    ax = axes[0, 0]
    ax.triplot(tri, 'k-', linewidth=0.2, alpha=0.5)
    ax.set_title('Full Mesh - Wireframe', fontweight='bold', fontsize=14)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Zoomed view
    ax = axes[0, 1]
    ax.triplot(tri, 'k-', linewidth=0.5)
    cx, cy = mesh.p[0].mean(), mesh.p[1].mean()
    zoom = 50
    ax.set_xlim(cx - zoom, cx + zoom)
    ax.set_ylim(cy - zoom, cy + zoom)
    ax.set_title('Zoomed View', fontweight='bold', fontsize=14)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Filled mesh
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
    
    # Highlight stroke if present
    if 'stroke_info' in metadata:
        info = metadata['stroke_info']
        cy, cx = info['center_y'], info['center_x']
        ax.plot(cy, cx, 'w+', markersize=20, markeredgewidth=3)
        title = f"Material Regions\n{info['type']} stroke ({info['size_category']})"
    else:
        title = "Material Regions"
    
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_aspect('equal')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Material ID', fontweight='bold')
    
    # Add info text
    info_text = (
        f"Mesh: {loaded_mesh['path'].name}\n"
        f"Vertices: {validation['n_vertices']:,}\n"
        f"Triangles: {validation['n_triangles']:,}\n"
        f"Materials: {list(validation['materials'].keys())}\n"
        f"Valid: {validation['valid']}"
    )
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved detailed view: {output_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Load and verify stroke sample meshes in scikit-fem"
    )
    parser.add_argument(
        'mesh_dir',
        type=str,
        nargs='?',
        help='Directory containing mesh files (e.g., stroke_samples_6layer/meshes)'
    )
    parser.add_argument(
        '--max-meshes',
        type=int,
        default=None,
        help='Maximum number of meshes to load (default: all)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create visualizations'
    )
    parser.add_argument(
        '--detailed',
        type=int,
        default=None,
        help='Create detailed visualization for mesh N'
    )
    
    args = parser.parse_args()
    
    # Find mesh directories if not specified
    if args.mesh_dir is None:
        mesh_dirs = []
        for sample_dir in Path('.').iterdir():
            if sample_dir.is_dir() and sample_dir.name.startswith('stroke_samples_'):
                mesh_subdir = sample_dir / 'meshes'
                if mesh_subdir.exists():
                    mesh_dirs.append(mesh_subdir)
        
        if len(mesh_dirs) == 0:
            print("✗ No mesh directories found")
            print("  Looking for: stroke_samples_*/meshes/")
            sys.exit(1)
        
        print(f"Found {len(mesh_dirs)} mesh directories:")
        for i, d in enumerate(mesh_dirs, 1):
            n_meshes = len(list(d.glob("sample_*.npz")))
            print(f"  {i}. {d} ({n_meshes} meshes)")
        
        choice = input(f"\nSelect directory (1-{len(mesh_dirs)}): ").strip()
        try:
            idx = int(choice) - 1
            mesh_dir = mesh_dirs[idx]
        except (ValueError, IndexError):
            print("Invalid choice")
            sys.exit(1)
    else:
        mesh_dir = Path(args.mesh_dir)
    
    # Load all meshes
    loaded_meshes = load_all_meshes(mesh_dir, max_meshes=args.max_meshes)
    
    if len(loaded_meshes) == 0:
        print("\n✗ No meshes loaded successfully")
        return
    
    # Visualizations
    if args.visualize:
        visualize_sample_meshes(loaded_meshes, mesh_dir)
    
    if args.detailed is not None:
        if 0 <= args.detailed < len(loaded_meshes):
            output_path = mesh_dir / f"detailed_mesh_{args.detailed:04d}.png"
            visualize_detailed_mesh(loaded_meshes[args.detailed], output_path)
        else:
            print(f"✗ Invalid mesh index: {args.detailed}")
    
    print("\n" + "="*70)
    print("✓ COMPLETE!")
    print("="*70)
    print(f"\nAll {len(loaded_meshes)} meshes successfully loaded into scikit-fem!")
    print("Ready for EIT forward problem simulation.")
    
    return loaded_meshes


if __name__ == "__main__":
    main()