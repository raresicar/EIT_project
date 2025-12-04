# -*- coding: utf-8 -*-
"""
Dual Mesh Generation for EIT Forward and Inverse Problems

Generates two meshes for each subject and layer type:
- Fine mesh (subsample=1): For forward problem (generating synthetic data)
- Coarse mesh (subsample=3): For inverse problem (reconstruction)

This avoids the inverse crime where the same discretization is used
for both forward and inverse problems.

Directory structure:
    brainweb_subjects/
        subject_00/
            head_models.npz
            models_visualization.png
            meshes_3layer/
                forward/  (fine mesh, subsample=1)
                    head_mesh.npz, .vtk, .msh, .xdmf
                    mesh_visualization.png
                inverse/  (coarse mesh, subsample=3)
                    head_mesh.npz, .vtk, .msh, .xdmf
                    mesh_visualization.png
            meshes_6layer/
                forward/
                inverse/
            meshes_9layer/
                forward/
                inverse/
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
    Create proper nested mesh where each pixel becomes triangles.

    Args:
        layer_model: 2D array with material IDs
        pixel_size: Physical size of pixel (mm)
        subsample: Subsampling factor (higher = coarser mesh)

    Returns:
        points: Node coordinates (N, 2)
        cells: Triangle connectivity (M, 3)
        cell_data: Material ID per triangle (M,)
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
    """Export mesh to multiple formats."""

    # Ensure points are 3D for meshio
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
    """Create visualization of a single mesh."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    from matplotlib.tri import Triangulation
    tri = Triangulation(points[:, 0], points[:, 1], cells)

    # Plot 1: Wireframe
    ax = axes[0]
    ax.triplot(tri, 'k-', linewidth=0.2, alpha=0.5)
    ax.set_title(f'{title} - Structure\n({len(points)} nodes, {len(cells)} triangles)',
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

def process_subject_layer_dual(model, layer_type, subject_dir, pixel_size=2.0,
                               subsample_forward=1, subsample_inverse=3):
    """
    Process one layer type for one subject, creating both forward and inverse meshes.

    Args:
        model: 2D array with layer labels
        layer_type: '3layer', '6layer', or '9layer'
        subject_dir: Path to subject directory
        pixel_size: Physical size of pixel (mm)
        subsample_forward: Subsampling for forward (fine) mesh
        subsample_inverse: Subsampling for inverse (coarse) mesh

    Returns:
        dict with processing results
    """
    # Create mesh subdirectory
    mesh_dir = subject_dir / f"meshes_{layer_type}"
    mesh_dir.mkdir(exist_ok=True)

    results = {}

    # Generate FORWARD mesh (fine)
    forward_dir = mesh_dir / "forward"
    forward_dir.mkdir(exist_ok=True)

    print(f"    Generating forward mesh (subsample={subsample_forward})...", end=" ")
    points_fwd, cells_fwd, materials_fwd = create_nested_mesh(
        model, pixel_size=pixel_size, subsample=subsample_forward
    )

    # Export forward mesh
    mesh_base_fwd = forward_dir / "head_mesh"
    exported_fwd = export_mesh(points_fwd, cells_fwd, materials_fwd, mesh_base_fwd)

    # Save NPZ
    npz_path_fwd = forward_dir / "head_mesh.npz"
    np.savez_compressed(npz_path_fwd, points=points_fwd, cells=cells_fwd, materials=materials_fwd)
    exported_fwd.append(str(npz_path_fwd))

    # Visualize
    viz_path_fwd = forward_dir / "mesh_visualization.png"
    visualize_mesh(points_fwd, cells_fwd, materials_fwd, viz_path_fwd,
                  title=f"{subject_dir.name} - {layer_type} - FORWARD")

    print(f"Done ({len(points_fwd):,} nodes, {len(cells_fwd):,} tris)")

    results['forward'] = {
        'n_nodes': int(len(points_fwd)),
        'n_triangles': int(len(cells_fwd)),
        'subsample': subsample_forward,
        'files': exported_fwd
    }

    # Generate INVERSE mesh (coarse)
    inverse_dir = mesh_dir / "inverse"
    inverse_dir.mkdir(exist_ok=True)

    print(f"    Generating inverse mesh (subsample={subsample_inverse})...", end=" ")
    points_inv, cells_inv, materials_inv = create_nested_mesh(
        model, pixel_size=pixel_size, subsample=subsample_inverse
    )

    # Export inverse mesh
    mesh_base_inv = inverse_dir / "head_mesh"
    exported_inv = export_mesh(points_inv, cells_inv, materials_inv, mesh_base_inv)

    # Save NPZ
    npz_path_inv = inverse_dir / "head_mesh.npz"
    np.savez_compressed(npz_path_inv, points=points_inv, cells=cells_inv, materials=materials_inv)
    exported_inv.append(str(npz_path_inv))

    # Visualize
    viz_path_inv = inverse_dir / "mesh_visualization.png"
    visualize_mesh(points_inv, cells_inv, materials_inv, viz_path_inv,
                  title=f"{subject_dir.name} - {layer_type} - INVERSE")

    print(f"Done ({len(points_inv):,} nodes, {len(cells_inv):,} tris)")

    results['inverse'] = {
        'n_nodes': int(len(points_inv)),
        'n_triangles': int(len(cells_inv)),
        'subsample': subsample_inverse,
        'files': exported_inv
    }

    # Statistics
    results['layer_type'] = layer_type
    results['materials'] = np.unique(materials_fwd).tolist()
    results['pixel_size_mm'] = pixel_size

    return results


def process_single_subject_dual(subject_dir, pixel_size=2.0,
                                subsample_forward=1, subsample_inverse=3):
    """
    Process all layer types for a single subject (dual meshes).

    Args:
        subject_dir: Path to subject directory
        pixel_size: Physical size of pixel (mm)
        subsample_forward: Subsampling for forward mesh (1 = full resolution)
        subsample_inverse: Subsampling for inverse mesh (3 = 3x coarser)

    Returns:
        dict with all processing results
    """
    subject_name = subject_dir.name
    print(f"\nProcessing {subject_name}...")

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
        print(f"  {layer_type}:")
        try:
            stats = process_subject_layer_dual(
                model, layer_type, subject_dir,
                pixel_size=pixel_size,
                subsample_forward=subsample_forward,
                subsample_inverse=subsample_inverse
            )
            results[layer_type] = stats
        except Exception as e:
            print(f"    ✗ Failed {layer_type}: {e}")
            results[layer_type] = {'error': str(e)}

    return {
        'subject_name': subject_name,
        'layer_results': results
    }


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_all_subjects_dual(subjects_base_dir='brainweb_subjects',
                              pixel_size=2.0,
                              subsample_forward=1,
                              subsample_inverse=3,
                              max_subjects=None):
    """
    Process all subjects and generate dual meshes (forward + inverse).

    Args:
        subjects_base_dir: Base directory containing subject folders
        pixel_size: Physical size of pixel (mm)
        subsample_forward: Subsampling for forward mesh (1 = full resolution)
        subsample_inverse: Subsampling for inverse mesh (3 = 3x coarser)
        max_subjects: Limit number of subjects (None = all)

    Returns:
        List of processing results
    """
    print("="*70)
    print("DUAL MESH GENERATION FOR EIT")
    print("="*70)
    print("\nGenerating both forward (fine) and inverse (coarse) meshes")
    print("to avoid inverse crime in reconstruction.\n")

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

    print(f"Found {len(subject_dirs)} subjects")
    print(f"\nMesh parameters:")
    print(f"  Pixel size:           {pixel_size} mm")
    print(f"  Forward subsample:    {subsample_forward}x (fine mesh)")
    print(f"  Inverse subsample:    {subsample_inverse}x (coarse mesh)")
    print(f"  Mesh ratio:           ~{subsample_inverse**2}x fewer elements for inverse")
    print("="*70)

    # Process each subject
    all_results = []

    for subject_dir in subject_dirs:
        try:
            result = process_single_subject_dual(
                subject_dir,
                pixel_size=pixel_size,
                subsample_forward=subsample_forward,
                subsample_inverse=subsample_inverse
            )
            all_results.append(result)
        except Exception as e:
            print(f"\n✗ Failed to process {subject_dir.name}: {e}")
            continue

    print(f"\n{'='*70}")
    print(f"✓ Successfully processed {len(all_results)}/{len(subject_dirs)} subjects")

    # Save summary
    summary = {
        'n_subjects_processed': len(all_results),
        'n_subjects_total': len(subject_dirs),
        'pixel_size_mm': pixel_size,
        'subsample_forward': subsample_forward,
        'subsample_inverse': subsample_inverse,
        'subjects': all_results
    }

    summary_path = subjects_base_dir / "dual_mesh_generation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Saved summary: {summary_path}")

    # Statistics
    print("\n" + "="*70)
    print("MESH STATISTICS")
    print("="*70)

    if len(all_results) > 0:
        for layer_type in ['3layer', '6layer', '9layer']:
            print(f"\n{layer_type.upper()}:")

            fwd_stats = []
            inv_stats = []

            for result in all_results:
                if layer_type in result['layer_results']:
                    lr = result['layer_results'][layer_type]
                    if 'forward' in lr and 'n_nodes' in lr['forward']:
                        fwd_stats.append(lr['forward'])
                    if 'inverse' in lr and 'n_nodes' in lr['inverse']:
                        inv_stats.append(lr['inverse'])

            if fwd_stats:
                n_nodes_fwd = [s['n_nodes'] for s in fwd_stats]
                n_tris_fwd = [s['n_triangles'] for s in fwd_stats]
                print(f"  FORWARD (fine mesh):")
                print(f"    Subjects:  {len(fwd_stats)}/{len(all_results)}")
                print(f"    Nodes:     min={min(n_nodes_fwd):,}, max={max(n_nodes_fwd):,}, mean={np.mean(n_nodes_fwd):,.0f}")
                print(f"    Triangles: min={min(n_tris_fwd):,}, max={max(n_tris_fwd):,}, mean={np.mean(n_tris_fwd):,.0f}")

            if inv_stats:
                n_nodes_inv = [s['n_nodes'] for s in inv_stats]
                n_tris_inv = [s['n_triangles'] for s in inv_stats]
                print(f"  INVERSE (coarse mesh):")
                print(f"    Subjects:  {len(inv_stats)}/{len(all_results)}")
                print(f"    Nodes:     min={min(n_nodes_inv):,}, max={max(n_nodes_inv):,}, mean={np.mean(n_nodes_inv):,.0f}")
                print(f"    Triangles: min={min(n_tris_inv):,}, max={max(n_tris_inv):,}, mean={np.mean(n_tris_inv):,.0f}")

            if fwd_stats and inv_stats:
                ratio = np.mean(n_tris_fwd) / np.mean(n_tris_inv)
                print(f"  Mesh ratio (forward/inverse): {ratio:.1f}x")

    print("\n" + "="*70)
    print("✓ COMPLETE!")
    print("="*70)
    print(f"\nGenerated dual mesh directories for each subject:")
    print(f"  - meshes_3layer/forward/  (fine mesh for forward problem)")
    print(f"  - meshes_3layer/inverse/  (coarse mesh for inverse problem)")
    print(f"  - meshes_6layer/forward/  (fine mesh for forward problem)")
    print(f"  - meshes_6layer/inverse/  (coarse mesh for inverse problem)")
    print(f"  - meshes_9layer/forward/  (fine mesh for forward problem)")
    print(f"  - meshes_9layer/inverse/  (coarse mesh for inverse problem)")

    return all_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate dual FEM meshes (forward/inverse) for EIT reconstruction"
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
        '--subsample-forward',
        type=int,
        default=1,
        help='Subsampling for forward mesh (default: 1 = full resolution)'
    )
    parser.add_argument(
        '--subsample-inverse',
        type=int,
        default=3,
        help='Subsampling for inverse mesh (default: 3 = 3x coarser)'
    )
    parser.add_argument(
        '--max-subjects',
        type=int,
        default=None,
        help='Maximum number of subjects to process (default: all)'
    )

    args = parser.parse_args()

    # Process all subjects
    results = process_all_subjects_dual(
        subjects_base_dir=args.subjects_dir,
        pixel_size=args.pixel_size,
        subsample_forward=args.subsample_forward,
        subsample_inverse=args.subsample_inverse,
        max_subjects=args.max_subjects
    )

    return results


if __name__ == "__main__":
    main()
