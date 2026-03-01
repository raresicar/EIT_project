"""
Load and Verify All BrainWeb Meshes (Forward + Inverse)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

# --- Project-relative imports via eit_config ---
_dir = Path(__file__).resolve().parent
while not (_dir / "eit_config.py").exists():
    _dir = _dir.parent
sys.path.insert(0, str(_dir))
from eit_config import *

print("Importing scikit-fem...")
try:
    from skfem import MeshTri
    print("✓ scikit-fem imported successfully")
except ImportError as e:
    print(f"✗ Error importing scikit-fem: {e}")
    print("Install with: pip install scikit-fem")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("⚠ tqdm not available, using simple progress")
    def tqdm(iterable, desc="Processing"):
        print(f"{desc}...")
        return iterable


def load_mesh_to_skfem(npz_path):
    data = np.load(npz_path)
    points = data['points']
    cells = data['cells']
    materials = data['materials']
    if points.shape[1] == 3:
        points_2d = points[:, :2]
    else:
        points_2d = points
    mesh = MeshTri(points_2d.T, cells.T)
    return mesh, materials


def validate_mesh(mesh, materials, mesh_name="mesh"):
    areas = []
    for i in range(mesh.t.shape[1]):
        tri = mesh.t[:, i]
        p0, p1, p2 = mesh.p[:, tri[0]], mesh.p[:, tri[1]], mesh.p[:, tri[2]]
        area = 0.5 * abs((p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1]))
        areas.append(area)
    areas = np.array(areas)
    degenerate = areas < 1e-10
    n_degenerate = degenerate.sum()
    unique_materials = np.unique(materials)
    material_counts = {int(m): int((materials == m).sum()) for m in unique_materials}
    return {
        'name': mesh_name,
        'n_vertices': int(mesh.p.shape[1]),
        'n_triangles': int(mesh.t.shape[1]),
        'n_boundary_edges': int(mesh.facets.shape[1]),
        'area_min': float(areas.min()),
        'area_max': float(areas.max()),
        'area_mean': float(areas.mean()),
        'n_degenerate': int(n_degenerate),
        'materials': material_counts,
        'valid': bool(n_degenerate == 0)
    }


def load_subject_meshes(subject_dir):
    subject_name = subject_dir.name
    results = {}
    for layer_type in ['3layer', '6layer', '9layer']:
        results[layer_type] = {}
        for mesh_type in ['forward', 'inverse']:
            mesh_file = subject_dir / layer_type / mesh_type / "head_mesh.npz"
            if not mesh_file.exists():
                results[layer_type][mesh_type] = {'success': False, 'error': f"Not found: {mesh_file}"}
                continue
            try:
                mesh, materials = load_mesh_to_skfem(mesh_file)
                validation = validate_mesh(mesh, materials, f"{subject_name}_{layer_type}_{mesh_type}")
                results[layer_type][mesh_type] = {
                    'success': True, 'mesh': mesh, 'materials': materials,
                    'validation': validation, 'mesh_file': str(mesh_file)
                }
            except Exception as e:
                results[layer_type][mesh_type] = {'success': False, 'error': str(e)}
    return {'subject_name': subject_name, 'meshes': results}


def load_all_subject_meshes(meshes_dir=None, max_subjects=None):
    if meshes_dir is None:
        meshes_dir = str(BRAINWEB_MESHES_DIR)

    print("="*70)
    print("Loading All BrainWeb Meshes (Forward + Inverse)")
    print("="*70)
    meshes_dir = Path(meshes_dir)
    if not meshes_dir.exists():
        raise ValueError(f"Meshes directory not found: {meshes_dir}")
    subject_dirs = sorted([d for d in meshes_dir.iterdir() if d.is_dir() and d.name.startswith('subject_')])
    if len(subject_dirs) == 0:
        print(f"\n✗ No subject directories found in {meshes_dir}")
        return []
    if max_subjects is not None:
        subject_dirs = subject_dirs[:max_subjects]
    print(f"\nFound {len(subject_dirs)} subjects")
    print(f"Loading meshes (forward + inverse for 3 layer types each)...\n")
    
    all_subjects = []
    for subject_dir in tqdm(subject_dirs, desc="Loading subjects"):
        try:
            subject_data = load_subject_meshes(subject_dir)
            all_subjects.append(subject_data)
        except Exception as e:
            print(f"\n✗ Failed to load {subject_dir.name}: {e}")
            continue
    
    print(f"\n✓ Successfully loaded {len(all_subjects)}/{len(subject_dirs)} subjects")
    
    print("\n" + "="*70)
    print("Loading Statistics")
    print("="*70)
    total_meshes_loaded = 0
    total_meshes_expected = len(all_subjects) * 3 * 2
    for layer_type in ['3layer', '6layer', '9layer']:
        print(f"\n{layer_type.upper()}:")
        for mesh_type in ['forward', 'inverse']:
            successful = sum(1 for s in all_subjects if s['meshes'][layer_type][mesh_type]['success'])
            total_meshes_loaded += successful
            print(f"  {mesh_type.upper()}: {successful}/{len(all_subjects)} loaded")
            if successful > 0:
                validations = [s['meshes'][layer_type][mesh_type]['validation'] for s in all_subjects if s['meshes'][layer_type][mesh_type]['success']]
                n_vertices = [v['n_vertices'] for v in validations]
                n_triangles = [v['n_triangles'] for v in validations]
                n_degenerate = [v['n_degenerate'] for v in validations]
                print(f"    Vertices:  {min(n_vertices):,} - {max(n_vertices):,} (mean: {np.mean(n_vertices):,.0f})")
                print(f"    Triangles: {min(n_triangles):,} - {max(n_triangles):,} (mean: {np.mean(n_triangles):,.0f})")
                n_valid = sum(1 for v in validations if v['valid'])
                print(f"    Valid: {n_valid}/{successful}")
                if sum(n_degenerate) > 0:
                    print(f"    ⚠️ Degenerate triangles: {sum(n_degenerate)} total")
    
    print(f"\nTotal meshes loaded: {total_meshes_loaded}/{total_meshes_expected}")
    
    print("\n" + "="*70)
    print("Forward vs Inverse Mesh Ratios")
    print("="*70)
    for layer_type in ['3layer', '6layer', '9layer']:
        fwd_tris = []
        inv_tris = []
        for s in all_subjects:
            if s['meshes'][layer_type]['forward']['success'] and s['meshes'][layer_type]['inverse']['success']:
                fwd_tris.append(s['meshes'][layer_type]['forward']['validation']['n_triangles'])
                inv_tris.append(s['meshes'][layer_type]['inverse']['validation']['n_triangles'])
        if fwd_tris and inv_tris:
            ratio = np.mean(fwd_tris) / np.mean(inv_tris)
            print(f"{layer_type.upper()}: {ratio:.1f}x (forward has {ratio:.1f}× more elements)")
    
    report = {'n_subjects': len(all_subjects), 'n_meshes_loaded': total_meshes_loaded, 'n_meshes_expected': total_meshes_expected, 'subjects': []}
    for subject in all_subjects:
        subject_report = {'subject_name': subject['subject_name'], 'layers': {}}
        for layer_type in ['3layer', '6layer', '9layer']:
            subject_report['layers'][layer_type] = {}
            for mesh_type in ['forward', 'inverse']:
                mesh_data = subject['meshes'][layer_type][mesh_type]
                if mesh_data['success']:
                    subject_report['layers'][layer_type][mesh_type] = mesh_data['validation']
                else:
                    subject_report['layers'][layer_type][mesh_type] = {'error': mesh_data.get('error', 'Unknown error')}
        report['subjects'].append(subject_report)
    report_path = meshes_dir / "mesh_loading_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✓ Saved loading report: {report_path}")
    return all_subjects


def visualize_subject_meshes(subject_data, output_dir):
    subject_name = subject_data['subject_name']
    available_layers = [layer for layer in ['3layer', '6layer', '9layer']
                       if (subject_data['meshes'][layer]['forward']['success'] and
                           subject_data['meshes'][layer]['inverse']['success'])]
    if len(available_layers) == 0:
        print(f"  No complete meshes available for {subject_name}")
        return
    n_layers = len(available_layers)
    fig, axes = plt.subplots(n_layers, 2, figsize=(12, 6*n_layers))
    if n_layers == 1:
        axes = axes.reshape(1, -1)
    from matplotlib.tri import Triangulation
    for row_idx, layer_type in enumerate(available_layers):
        mesh_fwd_data = subject_data['meshes'][layer_type]['forward']
        mesh_fwd = mesh_fwd_data['mesh']
        materials_fwd = mesh_fwd_data['materials']
        val_fwd = mesh_fwd_data['validation']
        ax = axes[row_idx, 0]
        tri = Triangulation(mesh_fwd.p[0], mesh_fwd.p[1], mesh_fwd.t.T)
        ax.tripcolor(tri, materials_fwd, cmap='tab10', shading='flat', edgecolors='k', linewidth=0.05, alpha=0.8)
        ax.set_title(f"{layer_type.upper()} FORWARD\n{val_fwd['n_vertices']:,} nodes, {val_fwd['n_triangles']:,} tris", fontsize=11, fontweight='bold')
        ax.set_aspect('equal'); ax.axis('off')
        mesh_inv_data = subject_data['meshes'][layer_type]['inverse']
        mesh_inv = mesh_inv_data['mesh']
        materials_inv = mesh_inv_data['materials']
        val_inv = mesh_inv_data['validation']
        ax = axes[row_idx, 1]
        tri = Triangulation(mesh_inv.p[0], mesh_inv.p[1], mesh_inv.t.T)
        ax.tripcolor(tri, materials_inv, cmap='tab10', shading='flat', edgecolors='k', linewidth=0.1, alpha=0.8)
        ax.set_title(f"{layer_type.upper()} INVERSE\n{val_inv['n_vertices']:,} nodes, {val_inv['n_triangles']:,} tris", fontsize=11, fontweight='bold')
        ax.set_aspect('equal'); ax.axis('off')
    fig.suptitle(f"{subject_name} - Forward vs Inverse Meshes", fontsize=14, fontweight='bold')
    plt.tight_layout()
    viz_path = Path(output_dir) / f"{subject_name}_meshes_comparison.png"
    plt.savefig(viz_path, dpi=200, bbox_inches='tight'); plt.close()
    print(f"✓ Saved: {viz_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Load and verify all BrainWeb meshes (forward + inverse)")
    parser.add_argument('--meshes-dir', type=str, default=str(BRAINWEB_MESHES_DIR))
    parser.add_argument('--max-subjects', type=int, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--visualize-subject', type=str, default=None)
    args = parser.parse_args()
    meshes_dir = Path(args.meshes_dir)
    if not meshes_dir.exists():
        print(f"✗ Meshes directory not found: {meshes_dir}"); sys.exit(1)
    all_subjects = load_all_subject_meshes(meshes_dir=args.meshes_dir, max_subjects=args.max_subjects)
    if len(all_subjects) == 0:
        print("\n✗ No subjects loaded successfully"); return
    if args.visualize:
        from load_meshes_skfem import create_overview_visualization
        create_overview_visualization(all_subjects, meshes_dir)
    if args.visualize_subject:
        subject_data = None
        for s in all_subjects:
            if s['subject_name'] == args.visualize_subject:
                subject_data = s; break
        if subject_data:
            visualize_subject_meshes(subject_data, meshes_dir)
        else:
            print(f"✗ Subject not found: {args.visualize_subject}")
    print("\n" + "="*70); print("✓ COMPLETE!"); print("="*70)
    print(f"\nSuccessfully loaded meshes for {len(all_subjects)} subjects")
    return all_subjects


if __name__ == "__main__":
    main()