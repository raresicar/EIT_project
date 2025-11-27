"""
Batch Mesh Generation for All BrainWeb Subjects
Generates FEM meshes for 3-layer, 6-layer, and 9-layer models
Organized in subject subdirectories
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
# MESH GENERATION
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
            exported_files.append(str(filename))
        except Exception as e:
            print(f"    ✗ Failed to export {ext}: {e}")
    
    return exported_files


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_mesh(points, cells, materials, output_path, title="Mesh"):
    """Create visualization of a single mesh"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    from matplotlib.tri import Triangulation
    tri = Triangulation(points[:, 0], points[:, 1], cells)
    
    # Plot 1: Wireframe
    ax = axes[0]
    ax.triplot(tri, 'k-', linewidth=0.2, alpha=0.5)
    ax.set_title(f'{title} - Structure\n({len(points)} nodes, {len(cells)} tris)', 
                fontweight='bold', fontsize=11)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Material regions
    ax = axes[1]
    im = ax.tripcolor(tri, materials, cmap='tab10', shading='flat',
                     edgecolors='k', linewidth=0.1, alpha=0.8)
    ax.set_title(f'{title} - Materials', fontweight='bold', fontsize=11)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_aspect('equal')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Material ID', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


# =============================================================================
# PROCESS SINGLE SUBJECT
# =============================================================================

def process_subject_layer(model, layer_type, subject_dir, pixel_size=2.0, subsample=2):
    """
    Process one layer type for one subject
    
    Args:
        model: 2D array with layer labels
        layer_type: '3layer', '6layer', or '9layer'
        subject_dir: Path to subject directory
        pixel_size: Physical size of pixel (mm)
        subsample: Subsampling factor
    
    Returns:
        dict with processing results
    """
    # Create mesh subdirectory
    mesh_dir = subject_dir / f"meshes_{layer_type}"
    mesh_dir.mkdir(exist_ok=True)
    
    # Generate mesh
    points, cells, cell_data = create_nested_mesh(
        model,
        pixel_size=pixel_size,
        subsample=subsample
    )
    
    # Export mesh files
    mesh_base = mesh_dir / "head_mesh"
    exported_files = export_mesh(points, cells, cell_data, mesh_base)
    
    # Save NPZ
    npz_path = mesh_dir / "head_mesh.npz"
    np.savez_compressed(
        npz_path,
        points=points,
        cells=cells,
        materials=cell_data
    )
    exported_files.append(str(npz_path))
    
    # Create visualization
    viz_path = mesh_dir / "mesh_visualization.png"
    visualize_mesh(points, cells, cell_data, viz_path, 
                  title=f"{subject_dir.name} - {layer_type}")
    
    # Statistics
    stats = {
        'layer_type': layer_type,
        'n_nodes': int(len(points)),
        'n_triangles': int(len(cells)),
        'materials': np.unique(cell_data).tolist(),
        'pixel_size_mm': pixel_size,
        'subsample_factor': subsample,
        'files': exported_files
    }
    
    return stats


def process_single_subject(subject_dir, pixel_size=2.0, subsample=2):
    """
    Process all layer types for a single subject
    
    Args:
        subject_dir: Path to subject directory
        pixel_size: Physical size of pixel (mm)
        subsample: Subsampling factor
    
    Returns:
        dict with all processing results
    """
    subject_name = subject_dir.name
    
    # Load head models
    models_path = subject_dir / "head_models.npz"
    if not models_path.exists():
        raise FileNotFoundError(f"head_models.npz not found in {subject_dir}")
    
    data = np.load(models_path)
    
    models = {
        '3layer': data['model_3layer'],
        '6layer': data['model_6layer'],
        '9layer': data['model_9layer']
    }
    
    # Process each layer type
    results = {}
    
    for layer_type, model in models.items():
        try:
            stats = process_subject_layer(
                model, 
                layer_type, 
                subject_dir,
                pixel_size=pixel_size,
                subsample=subsample
            )
            results[layer_type] = stats
        except Exception as e:
            print(f"\n  ✗ Failed {layer_type}: {e}")
            results[layer_type] = {'error': str(e)}
    
    return {
        'subject_name': subject_name,
        'layer_results': results
    }


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_all_subjects(subjects_base_dir='brainweb_subjects', 
                        pixel_size=2.0, 
                        subsample=2,
                        max_subjects=None):
    """
    Process all subjects and generate meshes for all layer types
    
    Args:
        subjects_base_dir: Base directory containing subject folders
        pixel_size: Physical size of pixel (mm)
        subsample: Subsampling factor
        max_subjects: Limit number of subjects (None = all)
    
    Returns:
        List of processing results
    """
    print("="*70)
    print("Batch Mesh Generation for All BrainWeb Subjects")
    print("="*70)
    
    subjects_base_dir = Path(subjects_base_dir)
    
    if not subjects_base_dir.exists():
        raise ValueError(f"Subjects directory not found: {subjects_base_dir}")
    
    # Find all subject directories
    subject_dirs = sorted([d for d in subjects_base_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('subject_')])
    
    if len(subject_dirs) == 0:
        print(f"\n✗ No subject directories found in {subjects_base_dir}")
        return []
    
    if max_subjects is not None:
        subject_dirs = subject_dirs[:max_subjects]
    
    print(f"\nFound {len(subject_dirs)} subjects")
    print(f"Mesh parameters:")
    print(f"  Pixel size: {pixel_size} mm")
    print(f"  Subsample:  {subsample}x")
    print(f"\nGenerating meshes for each subject (3 layer types each)...\n")
    
    # Process each subject
    all_results = []
    
    for subject_dir in tqdm(subject_dirs, desc="Processing subjects"):
        try:
            result = process_single_subject(
                subject_dir,
                pixel_size=pixel_size,
                subsample=subsample
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n✗ Failed to process {subject_dir.name}: {e}")
            continue
    
    print(f"\n✓ Successfully processed {len(all_results)}/{len(subject_dirs)} subjects")
    
    # Save summary
    summary = {
        'n_subjects_processed': len(all_results),
        'n_subjects_total': len(subject_dirs),
        'pixel_size_mm': pixel_size,
        'subsample_factor': subsample,
        'subjects': all_results
    }
    
    summary_path = subjects_base_dir / "mesh_generation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Saved summary: {summary_path}")
    
    # Statistics
    print("\n" + "="*70)
    print("Mesh Statistics")
    print("="*70)
    
    if len(all_results) > 0:
        for layer_type in ['3layer', '6layer', '9layer']:
            layer_stats = []
            for result in all_results:
                if layer_type in result['layer_results'] and 'n_nodes' in result['layer_results'][layer_type]:
                    layer_stats.append(result['layer_results'][layer_type])
            
            if layer_stats:
                n_nodes = [s['n_nodes'] for s in layer_stats]
                n_tris = [s['n_triangles'] for s in layer_stats]
                
                print(f"\n{layer_type.upper()}:")
                print(f"  Successfully generated: {len(layer_stats)}/{len(all_results)} subjects")
                print(f"  Nodes:     min={min(n_nodes):,}, max={max(n_nodes):,}, mean={np.mean(n_nodes):,.0f}")
                print(f"  Triangles: min={min(n_tris):,}, max={max(n_tris):,}, mean={np.mean(n_tris):,.0f}")
    
    print("\n" + "="*70)
    print("✓ COMPLETE!")
    print("="*70)
    print(f"\nGenerated mesh directories for each subject:")
    print(f"  - meshes_3layer/ (VTK, MSH, XDMF, NPZ + visualization)")
    print(f"  - meshes_6layer/ (VTK, MSH, XDMF, NPZ + visualization)")
    print(f"  - meshes_9layer/ (VTK, MSH, XDMF, NPZ + visualization)")
    print(f"\nDirectory structure:")
    print(f"  {subjects_base_dir}/")
    print(f"    subject_00/")
    print(f"      head_models.npz")
    print(f"      models_visualization.png")
    print(f"      meshes_3layer/")
    print(f"        head_mesh.vtk, .msh, .xdmf, .npz")
    print(f"        mesh_visualization.png")
    print(f"      meshes_6layer/")
    print(f"        head_mesh.vtk, .msh, .xdmf, .npz")
    print(f"        mesh_visualization.png")
    print(f"      meshes_9layer/")
    print(f"        head_mesh.vtk, .msh, .xdmf, .npz")
    print(f"        mesh_visualization.png")
    print(f"    subject_01/")
    print(f"      ...")
    
    return all_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate FEM meshes for all BrainWeb subjects (all layer types)"
    )
    parser.add_argument(
        '--subjects-dir',
        type=str,
        default='brainweb_subjects',
        help='Base directory containing subject folders (default: brainweb_subjects)'
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
        '--max-subjects',
        type=int,
        default=None,
        help='Maximum number of subjects to process (default: all)'
    )
    
    args = parser.parse_args()
    
    # Process all subjects
    results = process_all_subjects(
        subjects_base_dir=args.subjects_dir,
        pixel_size=args.pixel_size,
        subsample=args.subsample,
        max_subjects=args.max_subjects
    )
    
    return results


if __name__ == "__main__":
    main()