"""
Dual Mesh Generation for EIT Forward and Inverse Problems

Generates two meshes for each subject and layer type:
- Fine mesh (subsample=1): For forward problem (generating synthetic data)
- Coarse mesh (subsample=4): For inverse problem (reconstruction)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm
import sys

# --- Project-relative imports via eit_config ---
_dir = Path(__file__).resolve().parent
while not (_dir / "eit_config.py").exists():
    _dir = _dir.parent
sys.path.insert(0, str(_dir))
from eit_config import *

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

def create_mesh_from_layers(layer_model, pixel_size=2.0, subsample=1):
    """
    Create triangular mesh from layer model.
    """
    layer_sub = layer_model[::subsample, ::subsample]
    ny, nx = layer_sub.shape
    
    points = []
    node_grid = np.full((ny, nx), -1, dtype=int)
    
    for j in range(ny):
        for i in range(nx):
            if layer_sub[j, i] > 0:
                node_grid[j, i] = len(points)
                points.append([
                    i * pixel_size * subsample,
                    j * pixel_size * subsample
                ])
    
    points = np.array(points)
    
    triangles = []
    cell_materials = []
    
    for j in range(ny - 1):
        for i in range(nx - 1):
            material = layer_sub[j, i]
            
            if material > 0:
                n0 = node_grid[j, i]
                n1 = node_grid[j, i+1]
                n2 = node_grid[j+1, i+1]
                n3 = node_grid[j+1, i]
                
                if n0 >= 0 and n1 >= 0 and n2 >= 0 and n3 >= 0:
                    triangles.append([n0, n1, n2])
                    cell_materials.append(material)
                    
                    triangles.append([n0, n2, n3])
                    cell_materials.append(material)
    
    cells = np.array(triangles)
    materials = np.array(cell_materials)
    
    return points, cells, materials


# =============================================================================
# EXPORT
# =============================================================================

def export_mesh(points, cells, materials, output_dir):
    """Export mesh to multiple formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if points.shape[1] == 2:
        points_3d = np.column_stack([points, np.zeros(len(points))])
    else:
        points_3d = points
    
    mesh = meshio.Mesh(
        points=points_3d,
        cells=[("triangle", cells)],
        cell_data={"material": [materials]}
    )
    
    exported = []
    formats = ['vtk', 'msh', 'xdmf']
    
    for ext in formats:
        filepath = output_dir / f"head_mesh.{ext}"
        try:
            meshio.write(filepath, mesh)
            exported.append(str(filepath))
        except Exception as e:
            print(f"    ✗ Failed {ext}: {e}")
    
    npz_path = output_dir / "head_mesh.npz"
    np.savez_compressed(npz_path, points=points, cells=cells, materials=materials)
    exported.append(str(npz_path))
    
    return exported


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_mesh(points, cells, materials, output_path, title="Mesh"):
    """Create and save mesh visualization."""
    from matplotlib.tri import Triangulation
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    tri = Triangulation(points[:, 0], points[:, 1], cells)
    
    ax = axes[0]
    ax.triplot(tri, 'k-', linewidth=0.2, alpha=0.5)
    ax.set_title(f'{title} - Structure\n{len(points):,} nodes, {len(cells):,} triangles',
                fontweight='bold', fontsize=11)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
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

def process_subject(subject_dir_input, subject_dir_output, pixel_size=2.0,
                    subsample_forward=1, subsample_inverse=4):
    """Process one subject: generate all meshes."""
    subject_name = subject_dir_input.name
    print(f"\nProcessing {subject_name}...")
    
    models_path = subject_dir_input / "head_models.npz"
    if not models_path.exists():
        raise FileNotFoundError(f"head_models.npz not found in {subject_dir_input}")
    
    data = np.load(models_path)
    models = {
        '3layer': data['model_3layer'],
        '6layer': data['model_6layer'],
        '9layer': data['model_9layer']
    }
    
    results = {'subject_name': subject_name, 'layers': {}}
    
    for layer_type, model in models.items():
        print(f"  {layer_type}:")
        layer_results = {}
        layer_dir = subject_dir_output / layer_type
        
        # --- FORWARD MESH (fine) ---
        print(f"    Generating forward mesh (subsample={subsample_forward})...", end=" ")
        fwd_dir = layer_dir / "forward"
        points_fwd, cells_fwd, materials_fwd = create_mesh_from_layers(
            model, pixel_size=pixel_size, subsample=subsample_forward)
        files_fwd = export_mesh(points_fwd, cells_fwd, materials_fwd, fwd_dir)
        viz_fwd = fwd_dir / "mesh_viz.png"
        visualize_mesh(points_fwd, cells_fwd, materials_fwd, viz_fwd,
                      title=f"{subject_name} {layer_type} FORWARD")
        print(f"✓ {len(points_fwd):,} nodes, {len(cells_fwd):,} tris")
        
        layer_results['forward'] = {
            'n_nodes': int(len(points_fwd)),
            'n_triangles': int(len(cells_fwd)),
            'subsample': subsample_forward,
            'files': files_fwd + [str(viz_fwd)]
        }
        
        # --- INVERSE MESH (coarse) ---
        print(f"    Generating inverse mesh (subsample={subsample_inverse})...", end=" ")
        inv_dir = layer_dir / "inverse"
        points_inv, cells_inv, materials_inv = create_mesh_from_layers(
            model, pixel_size=pixel_size, subsample=subsample_inverse)
        files_inv = export_mesh(points_inv, cells_inv, materials_inv, inv_dir)
        viz_inv = inv_dir / "mesh_viz.png"
        visualize_mesh(points_inv, cells_inv, materials_inv, viz_inv,
                      title=f"{subject_name} {layer_type} INVERSE")
        print(f"✓ {len(points_inv):,} nodes, {len(cells_inv):,} tris")
        
        layer_results['inverse'] = {
            'n_nodes': int(len(points_inv)),
            'n_triangles': int(len(cells_inv)),
            'subsample': subsample_inverse,
            'files': files_inv + [str(viz_inv)]
        }
        
        layer_results['materials'] = np.unique(materials_fwd).tolist()
        results['layers'][layer_type] = layer_results
    
    return results


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_all_subjects(
    input_dir=None,
    output_dir=None,
    pixel_size=2.0,
    subsample_forward=1,
    subsample_inverse=4,
    max_subjects=None,
    batch_start=0
):
    """Process all subjects."""
    if input_dir is None:
        input_dir = str(BRAINWEB_SUBJECTS_DIR)
    if output_dir is None:
        output_dir = str(BRAINWEB_MESHES_DIR)

    print("="*70)
    print("DUAL MESH GENERATION FOR EIT")
    print("="*70)
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    subject_dirs = sorted([d for d in input_dir.iterdir()
                          if d.is_dir() and d.name.startswith('subject_')])
    
    if len(subject_dirs) == 0:
        print(f"\n✗ No subjects found in {input_dir}")
        return []

    if batch_start > 0:
        subject_dirs = subject_dirs[batch_start:]
    
    if max_subjects:
        subject_dirs = subject_dirs[:max_subjects]
    
    print(f"\nFound {len(subject_dirs)} subjects")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"\nMesh parameters:")
    print(f"  Pixel size:        {pixel_size} mm")
    print(f"  Forward subsample: {subsample_forward}x (fine)")
    print(f"  Inverse subsample: {subsample_inverse}x (coarse)")
    print(f"  Mesh ratio:        ~{subsample_inverse**2}x fewer elements for inverse")
    print("="*70)
    
    all_results = []
    
    for subject_dir_input in subject_dirs:
        subject_dir_output = output_dir / subject_dir_input.name
        try:
            result = process_subject(
                subject_dir_input, subject_dir_output,
                pixel_size=pixel_size,
                subsample_forward=subsample_forward,
                subsample_inverse=subsample_inverse)
            all_results.append(result)
        except Exception as e:
            print(f"\n✗ Failed {subject_dir_input.name}: {e}")
            continue
    
    print(f"\n{'='*70}")
    print(f"✓ Processed {len(all_results)}/{len(subject_dirs)} subjects")
    
    summary = {
        'n_subjects': len(all_results),
        'pixel_size_mm': pixel_size,
        'subsample_forward': subsample_forward,
        'subsample_inverse': subsample_inverse,
        'input_directory': str(input_dir),
        'output_directory': str(output_dir),
        'subjects': all_results
    }
    
    summary_path = output_dir / "mesh_generation_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Summary: {summary_path}")
    
    print("\n" + "="*70)
    print("STATISTICS")
    print("="*70)
    
    for layer in ['3layer', '6layer', '9layer']:
        print(f"\n{layer.upper()}:")
        fwd = [s['layers'][layer]['forward'] for s in all_results if layer in s['layers']]
        inv = [s['layers'][layer]['inverse'] for s in all_results if layer in s['layers']]
        if fwd:
            n_fwd = [f['n_nodes'] for f in fwd]
            t_fwd = [f['n_triangles'] for f in fwd]
            print(f"  FORWARD: {len(fwd)} subjects")
            print(f"    Nodes:     {min(n_fwd):,} - {max(n_fwd):,} (mean: {np.mean(n_fwd):,.0f})")
            print(f"    Triangles: {min(t_fwd):,} - {max(t_fwd):,} (mean: {np.mean(t_fwd):,.0f})")
        if inv:
            n_inv = [i['n_nodes'] for i in inv]
            t_inv = [i['n_triangles'] for i in inv]
            print(f"  INVERSE: {len(inv)} subjects")
            print(f"    Nodes:     {min(n_inv):,} - {max(n_inv):,} (mean: {np.mean(n_inv):,.0f})")
            print(f"    Triangles: {min(t_inv):,} - {max(t_inv):,} (mean: {np.mean(t_inv):,.0f})")
        if fwd and inv:
            ratio = np.mean(t_fwd) / np.mean(t_inv)
            print(f"  Ratio (forward/inverse): {ratio:.1f}x")
    
    print("\n" + "="*70)
    print("✓ COMPLETE!")
    print("="*70)
    
    return all_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate dual meshes for EIT")
    parser.add_argument('--input-dir', default=str(BRAINWEB_SUBJECTS_DIR))
    parser.add_argument('--output-dir', default=str(BRAINWEB_MESHES_DIR))
    parser.add_argument('--pixel-size', type=float, default=2.0)
    parser.add_argument('--subsample-forward', type=int, default=2)
    parser.add_argument('--subsample-inverse', type=int, default=5)
    parser.add_argument('--max-subjects', type=int, default=20)
    parser.add_argument('--batch-start', type=int, default=0)
    
    args = parser.parse_args()
    
    process_all_subjects(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        pixel_size=args.pixel_size,
        subsample_forward=args.subsample_forward,
        subsample_inverse=args.subsample_inverse,
        max_subjects=args.max_subjects,
        batch_start=args.batch_start
    )


if __name__ == "__main__":
    main()