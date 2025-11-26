"""
Batch Mesh Generation for Stroke Samples
Reads stroke sample folders and generates FEM meshes for each sample
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
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
# MESH GENERATION (from the nested mesh script)
# =============================================================================

def create_nested_mesh(layer_model, pixel_size=2.0, subsample=2):
    """
    Create proper nested mesh where each pixel becomes triangles
    Materials assigned correctly without overlap
    """
    # Subsample to reduce mesh size
    layer_sub = layer_model[::subsample, ::subsample]
    ny, nx = layer_sub.shape
    
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
    
    # Create triangles from grid
    triangles = []
    cell_materials = []
    
    for j in range(ny - 1):
        for i in range(nx - 1):
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
                    triangles.append([n0, n1, n2])
                    cell_materials.append(material)
                    
                    triangles.append([n0, n2, n3])
                    cell_materials.append(material)
    
    cells = np.array(triangles)
    cell_data = np.array(cell_materials)
    
    return points, cells, cell_data


# =============================================================================
# EXPORT
# =============================================================================

def export_mesh(points, cells, cell_data, output_path):
    """Export mesh to multiple formats"""
    
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
    formats = ['vtk', 'msh', 'xdmf']
    exported_files = []
    
    base_path = Path(output_path).with_suffix('')
    
    for ext in formats:
        filename = f"{base_path}.{ext}"
        try:
            meshio.write(filename, mesh)
            exported_files.append(filename)
        except Exception as e:
            print(f"    ✗ Failed to export {ext}: {e}")
    
    return exported_files


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_sample(sample_path, output_dir, pixel_size=2.0, subsample=2):
    """
    Process a single stroke sample and generate mesh
    
    Args:
        sample_path: Path to sample .npz file
        output_dir: Directory to save mesh files
        pixel_size: Physical size of each pixel (mm)
        subsample: Subsampling factor
    
    Returns:
        dict with mesh info and paths
    """
    # Load sample
    data = np.load(sample_path, allow_pickle=True)
    model = data['model']
    metadata = data['metadata'].item() if 'metadata' in data else {}
    
    # Generate mesh
    points, cells, cell_data = create_nested_mesh(
        model,
        pixel_size=pixel_size,
        subsample=subsample
    )
    
    # Create output filename
    sample_name = Path(sample_path).stem
    mesh_base_path = os.path.join(output_dir, sample_name)
    
    # Export mesh
    exported_files = export_mesh(points, cells, cell_data, mesh_base_path)
    
    # Save mesh data as NPZ too
    npz_path = f"{mesh_base_path}.npz"
    np.savez_compressed(
        npz_path,
        points=points,
        cells=cells,
        materials=cell_data,
        metadata=metadata
    )
    exported_files.append(npz_path)
    
    return {
        'sample_name': sample_name,
        'n_nodes': len(points),
        'n_triangles': len(cells),
        'materials': np.unique(cell_data).tolist(),
        'files': exported_files,
        'metadata': metadata
    }


def batch_process_stroke_samples(sample_dir, mesh_output_dir=None,
                                 pixel_size=2.0, subsample=2,
                                 max_samples=None):
    """
    Batch process all samples in a directory
    
    Args:
        sample_dir: Directory containing stroke samples
        mesh_output_dir: Output directory for meshes (default: sample_dir/meshes)
        pixel_size: Physical size of each pixel (mm)
        subsample: Subsampling factor
        max_samples: Maximum number of samples to process (None = all)
    
    Returns:
        List of processed sample info dictionaries
    """
    print("="*70)
    print(f"Batch Mesh Generation from Stroke Samples")
    print("="*70)
    
    # Setup directories
    sample_dir = Path(sample_dir)
    if not sample_dir.exists():
        raise ValueError(f"Sample directory not found: {sample_dir}")
    
    if mesh_output_dir is None:
        mesh_output_dir = sample_dir / "meshes"
    else:
        mesh_output_dir = Path(mesh_output_dir)
    
    mesh_output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nInput directory:  {sample_dir}")
    print(f"Output directory: {mesh_output_dir}")
    
    # Find all sample files
    sample_files = sorted(sample_dir.glob("sample_*.npz"))
    
    if len(sample_files) == 0:
        print(f"\n✗ No sample files found in {sample_dir}")
        return []
    
    if max_samples is not None:
        sample_files = sample_files[:max_samples]
    
    print(f"\nFound {len(sample_files)} samples to process")
    print(f"Mesh parameters:")
    print(f"  Pixel size: {pixel_size} mm")
    print(f"  Subsample:  {subsample}x")
    
    # Process each sample
    print("\nProcessing samples...")
    results = []
    
    for sample_path in tqdm(sample_files, desc="Generating meshes"):
        try:
            result = process_sample(
                sample_path,
                mesh_output_dir,
                pixel_size=pixel_size,
                subsample=subsample
            )
            results.append(result)
        except Exception as e:
            print(f"\n✗ Failed to process {sample_path.name}: {e}")
            continue
    
    print(f"\n✓ Successfully processed {len(results)}/{len(sample_files)} samples")
    
    # Save batch summary
    summary = {
        'n_samples_processed': len(results),
        'n_samples_total': len(sample_files),
        'pixel_size_mm': pixel_size,
        'subsample_factor': subsample,
        'mesh_output_dir': str(mesh_output_dir),
        'samples': results
    }
    
    summary_path = mesh_output_dir / "mesh_generation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Saved summary: {summary_path}")
    
    # Statistics
    print("\n" + "="*70)
    print("Mesh Statistics")
    print("="*70)
    
    if len(results) > 0:
        n_nodes = [r['n_nodes'] for r in results]
        n_tris = [r['n_triangles'] for r in results]
        
        print(f"\nNodes per mesh:")
        print(f"  Min:    {min(n_nodes):,}")
        print(f"  Max:    {max(n_nodes):,}")
        print(f"  Mean:   {np.mean(n_nodes):,.0f}")
        print(f"  Median: {np.median(n_nodes):,.0f}")
        
        print(f"\nTriangles per mesh:")
        print(f"  Min:    {min(n_tris):,}")
        print(f"  Max:    {max(n_tris):,}")
        print(f"  Mean:   {np.mean(n_tris):,.0f}")
        print(f"  Median: {np.median(n_tris):,.0f}")
        
        # Material distribution
        all_materials = set()
        for r in results:
            all_materials.update(r['materials'])
        
        print(f"\nMaterials found: {sorted(all_materials)}")
        
        # Stroke statistics
        stroke_types = {}
        stroke_sizes = {}
        
        for r in results:
            if 'stroke_info' in r['metadata']:
                info = r['metadata']['stroke_info']
                stype = info['type']
                ssize = info['size_category']
                
                stroke_types[stype] = stroke_types.get(stype, 0) + 1
                stroke_sizes[ssize] = stroke_sizes.get(ssize, 0) + 1
        
        if stroke_types:
            print(f"\nStroke type distribution:")
            for stype, count in sorted(stroke_types.items()):
                print(f"  {stype}: {count} samples ({100*count/len(results):.1f}%)")
        
        if stroke_sizes:
            print(f"\nStroke size distribution:")
            for ssize, count in sorted(stroke_sizes.items()):
                print(f"  {ssize}: {count} samples ({100*count/len(results):.1f}%)")
    
    print("\n" + "="*70)
    print("✓ COMPLETE!")
    print("="*70)
    print(f"\nGenerated files in: {mesh_output_dir}/")
    print(f"  - {len(results)} mesh sets (VTK, MSH, XDMF, NPZ)")
    print(f"  - mesh_generation_summary.json")
    
    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_random_meshes(results, mesh_dir, n_display=9):
    """Create visualization of random mesh samples"""
    
    if len(results) == 0:
        return
    
    print(f"\nCreating visualization of {n_display} random meshes...")
    
    # Select random samples
    indices = np.random.choice(len(results), min(n_display, len(results)), replace=False)
    
    n_cols = 3
    n_rows = (len(indices) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    axes = axes.flatten() if n_display > 1 else [axes]
    
    for idx, sample_idx in enumerate(indices):
        result = results[sample_idx]
        ax = axes[idx]
        
        # Load mesh
        npz_path = Path(mesh_dir) / f"{result['sample_name']}.npz"
        data = np.load(npz_path)
        points = data['points']
        cells = data['cells']
        materials = data['materials']
        
        # Plot
        from matplotlib.tri import Triangulation
        tri = Triangulation(points[:, 0], points[:, 1], cells)
        ax.tripcolor(tri, materials, cmap='tab10', shading='flat', edgecolors='k', linewidth=0.1)
        
        # Title
        metadata = result.get('metadata', {})
        if 'stroke_info' in metadata:
            info = metadata['stroke_info']
            title = f"{result['sample_name']}\n{info['type']} ({info['size_category']})"
        else:
            title = result['sample_name']
        
        ax.set_title(title, fontsize=10)
        ax.set_aspect('equal')
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(len(indices), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    viz_path = Path(mesh_dir) / "mesh_samples_overview.png"
    plt.savefig(viz_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved visualization: {viz_path}")
    plt.close()


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate FEM meshes from stroke sample folders"
    )
    parser.add_argument(
        'sample_dir',
        type=str,
        help='Directory containing stroke samples (e.g., stroke_samples_6layer)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for meshes (default: sample_dir/meshes)'
    )
    parser.add_argument(
        '--pixel-size',
        type=float,
        default=2.0,
        help='Physical size of each pixel in mm (default: 2.0)'
    )
    parser.add_argument(
        '--subsample',
        type=int,
        default=2,
        help='Subsampling factor (default: 2, higher = coarser)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (default: all)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create visualization of random mesh samples'
    )
    
    args = parser.parse_args()
    
    # Process samples
    results = batch_process_stroke_samples(
        sample_dir=args.sample_dir,
        mesh_output_dir=args.output_dir,
        pixel_size=args.pixel_size,
        subsample=args.subsample,
        max_samples=args.max_samples
    )
    
    # Create visualization
    if args.visualize and len(results) > 0:
        mesh_dir = args.output_dir if args.output_dir else Path(args.sample_dir) / "meshes"
        visualize_random_meshes(results, mesh_dir)
    
    return results


if __name__ == "__main__":
    # Check if command line args provided
    if len(sys.argv) > 1:
        main()
    else:
        # Interactive mode
        print("="*70)
        print("Interactive Mesh Generation")
        print("="*70)
        
        # Find sample directories
        sample_dirs = [d for d in Path('.').iterdir() 
                      if d.is_dir() and d.name.startswith('stroke_samples_')]
        
        if len(sample_dirs) == 0:
            print("\n✗ No stroke sample directories found in current directory")
            print("  Looking for directories named: stroke_samples_*")
            sys.exit(1)
        
        print(f"\nFound {len(sample_dirs)} sample directories:")
        for i, d in enumerate(sample_dirs, 1):
            n_samples = len(list(d.glob("sample_*.npz")))
            print(f"  {i}. {d.name} ({n_samples} samples)")
        
        # Get user choice
        choice = input(f"\nSelect directory (1-{len(sample_dirs)}): ").strip()
        try:
            idx = int(choice) - 1
            sample_dir = sample_dirs[idx]
        except (ValueError, IndexError):
            print("Invalid choice")
            sys.exit(1)
        
        # Process
        results = batch_process_stroke_samples(
            sample_dir=sample_dir,
            pixel_size=2.0,
            subsample=2
        )
        
        # Visualize
        if len(results) > 0:
            mesh_dir = sample_dir / "meshes"
            visualize_random_meshes(results, mesh_dir)