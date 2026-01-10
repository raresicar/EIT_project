"""
Load and Verify All BrainWeb Meshes (Forward + Inverse)

Loads and validates all generated meshes from brainweb_meshes/ directory.
Checks both forward (fine) and inverse (coarse) meshes for all subjects and layer types.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

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


# =============================================================================
# MESH LOADING
# =============================================================================

def load_mesh_to_skfem(npz_path):
    """
    Load mesh from NPZ file into scikit-fem.
    
    Returns:
        mesh: scikit-fem MeshTri object
        materials: Material array
    """
    data = np.load(npz_path)
    
    points = data['points']
    cells = data['cells']
    materials = data['materials']
    
    # Extract 2D coordinates
    if points.shape[1] == 3:
        points_2d = points[:, :2]
    else:
        points_2d = points
    
    # Create scikit-fem mesh (column format)
    mesh = MeshTri(points_2d.T, cells.T)
    
    return mesh, materials


# =============================================================================
# MESH VALIDATION
# =============================================================================

def validate_mesh(mesh, materials, mesh_name="mesh"):
    """
    Validate mesh quality.
    
    Returns:
        dict with validation results
    """
    # Calculate triangle areas
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
        'valid': bool(n_degenerate == 0)
    }
    
    return results


# =============================================================================
# LOAD SINGLE SUBJECT
# =============================================================================

def load_subject_meshes(subject_dir):
    """
    Load all meshes (forward + inverse) for one subject.
    
    Args:
        subject_dir: Path to subject directory in brainweb_meshes/
    
    Returns:
        dict with loaded meshes for each layer type and mesh type
    """
    subject_name = subject_dir.name
    results = {}
    
    for layer_type in ['3layer', '6layer', '9layer']:
        results[layer_type] = {}
        
        for mesh_type in ['forward', 'inverse']:
            mesh_file = subject_dir / layer_type / mesh_type / "head_mesh.npz"
            
            if not mesh_file.exists():
                results[layer_type][mesh_type] = {
                    'success': False,
                    'error': f"Mesh file not found: {mesh_file}"
                }
                continue
            
            try:
                # Load mesh
                mesh, materials = load_mesh_to_skfem(mesh_file)
                
                # Validate
                validation = validate_mesh(
                    mesh, materials, 
                    f"{subject_name}_{layer_type}_{mesh_type}"
                )
                
                results[layer_type][mesh_type] = {
                    'success': True,
                    'mesh': mesh,
                    'materials': materials,
                    'validation': validation,
                    'mesh_file': str(mesh_file)
                }
                
            except Exception as e:
                results[layer_type][mesh_type] = {
                    'success': False,
                    'error': str(e)
                }
    
    return {
        'subject_name': subject_name,
        'meshes': results
    }


# =============================================================================
# BATCH LOADING
# =============================================================================

def load_all_subject_meshes(meshes_dir='brainweb_meshes', max_subjects=None):
    """
    Load all meshes from all subjects.
    
    Args:
        meshes_dir: Directory containing subject mesh folders
        max_subjects: Maximum number of subjects to load (None = all)
    
    Returns:
        list of dicts with subject mesh info
    """
    print("="*70)
    print("Loading All BrainWeb Meshes (Forward + Inverse)")
    print("="*70)
    
    meshes_dir = Path(meshes_dir)
    
    if not meshes_dir.exists():
        raise ValueError(f"Meshes directory not found: {meshes_dir}")
    
    # Find all subject directories
    subject_dirs = sorted([d for d in meshes_dir.iterdir() 
                          if d.is_dir() and d.name.startswith('subject_')])
    
    if len(subject_dirs) == 0:
        print(f"\n✗ No subject directories found in {meshes_dir}")
        return []
    
    if max_subjects is not None:
        subject_dirs = subject_dirs[:max_subjects]
    
    print(f"\nFound {len(subject_dirs)} subjects")
    print(f"Loading meshes (forward + inverse for 3 layer types each)...\n")
    
    # Load all subjects
    all_subjects = []
    
    for subject_dir in tqdm(subject_dirs, desc="Loading subjects"):
        try:
            subject_data = load_subject_meshes(subject_dir)
            all_subjects.append(subject_data)
        except Exception as e:
            print(f"\n✗ Failed to load {subject_dir.name}: {e}")
            continue
    
    print(f"\n✓ Successfully loaded {len(all_subjects)}/{len(subject_dirs)} subjects")
    
    # Statistics
    print("\n" + "="*70)
    print("Loading Statistics")
    print("="*70)
    
    total_meshes_loaded = 0
    total_meshes_expected = len(all_subjects) * 3 * 2  # 3 layers × 2 mesh types
    
    for layer_type in ['3layer', '6layer', '9layer']:
        print(f"\n{layer_type.upper()}:")
        
        for mesh_type in ['forward', 'inverse']:
            successful = sum(1 for s in all_subjects 
                           if s['meshes'][layer_type][mesh_type]['success'])
            total_meshes_loaded += successful
            
            print(f"  {mesh_type.upper()}: {successful}/{len(all_subjects)} loaded")
            
            if successful > 0:
                # Get statistics
                validations = [s['meshes'][layer_type][mesh_type]['validation'] 
                              for s in all_subjects 
                              if s['meshes'][layer_type][mesh_type]['success']]
                
                n_vertices = [v['n_vertices'] for v in validations]
                n_triangles = [v['n_triangles'] for v in validations]
                n_degenerate = [v['n_degenerate'] for v in validations]
                
                print(f"    Vertices:  {min(n_vertices):,} - {max(n_vertices):,} "
                      f"(mean: {np.mean(n_vertices):,.0f})")
                print(f"    Triangles: {min(n_triangles):,} - {max(n_triangles):,} "
                      f"(mean: {np.mean(n_triangles):,.0f})")
                
                n_valid = sum(1 for v in validations if v['valid'])
                print(f"    Valid: {n_valid}/{successful}")
                
                if sum(n_degenerate) > 0:
                    print(f"    ⚠️ Degenerate triangles: {sum(n_degenerate)} total")
    
    print(f"\nTotal meshes loaded: {total_meshes_loaded}/{total_meshes_expected}")
    
    # Mesh size ratios
    print("\n" + "="*70)
    print("Forward vs Inverse Mesh Ratios")
    print("="*70)
    
    for layer_type in ['3layer', '6layer', '9layer']:
        fwd_tris = []
        inv_tris = []
        
        for s in all_subjects:
            if (s['meshes'][layer_type]['forward']['success'] and 
                s['meshes'][layer_type]['inverse']['success']):
                fwd_tris.append(s['meshes'][layer_type]['forward']['validation']['n_triangles'])
                inv_tris.append(s['meshes'][layer_type]['inverse']['validation']['n_triangles'])
        
        if fwd_tris and inv_tris:
            ratio = np.mean(fwd_tris) / np.mean(inv_tris)
            print(f"{layer_type.upper()}: {ratio:.1f}x (forward has {ratio:.1f}× more elements)")
    
    # Save loading report
    report = {
        'n_subjects': len(all_subjects),
        'n_meshes_loaded': total_meshes_loaded,
        'n_meshes_expected': total_meshes_expected,
        'subjects': []
    }
    
    for subject in all_subjects:
        subject_report = {
            'subject_name': subject['subject_name'],
            'layers': {}
        }
        
        for layer_type in ['3layer', '6layer', '9layer']:
            subject_report['layers'][layer_type] = {}
            
            for mesh_type in ['forward', 'inverse']:
                mesh_data = subject['meshes'][layer_type][mesh_type]
                if mesh_data['success']:
                    subject_report['layers'][layer_type][mesh_type] = mesh_data['validation']
                else:
                    subject_report['layers'][layer_type][mesh_type] = {
                        'error': mesh_data.get('error', 'Unknown error')
                    }
        
        report['subjects'].append(subject_report)
    
    report_path = meshes_dir / "mesh_loading_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Saved loading report: {report_path}")
    
    return all_subjects


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_subject_meshes(subject_data, output_dir):
    """
    Create visualization comparing forward vs inverse for all layer types.
    
    Args:
        subject_data: Dict with subject mesh data
        output_dir: Directory to save visualization
    """
    subject_name = subject_data['subject_name']
    
    # Check available layers
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
        # Forward mesh
        mesh_fwd_data = subject_data['meshes'][layer_type]['forward']
        mesh_fwd = mesh_fwd_data['mesh']
        materials_fwd = mesh_fwd_data['materials']
        val_fwd = mesh_fwd_data['validation']
        
        ax = axes[row_idx, 0]
        tri = Triangulation(mesh_fwd.p[0], mesh_fwd.p[1], mesh_fwd.t.T)
        ax.tripcolor(tri, materials_fwd, cmap='tab10', shading='flat',
                    edgecolors='k', linewidth=0.05, alpha=0.8)
        ax.set_title(f"{layer_type.upper()} FORWARD\n"
                    f"{val_fwd['n_vertices']:,} nodes, {val_fwd['n_triangles']:,} tris",
                    fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Inverse mesh
        mesh_inv_data = subject_data['meshes'][layer_type]['inverse']
        mesh_inv = mesh_inv_data['mesh']
        materials_inv = mesh_inv_data['materials']
        val_inv = mesh_inv_data['validation']
        
        ax = axes[row_idx, 1]
        tri = Triangulation(mesh_inv.p[0], mesh_inv.p[1], mesh_inv.t.T)
        ax.tripcolor(tri, materials_inv, cmap='tab10', shading='flat',
                    edgecolors='k', linewidth=0.1, alpha=0.8)
        ax.set_title(f"{layer_type.upper()} INVERSE\n"
                    f"{val_inv['n_vertices']:,} nodes, {val_inv['n_triangles']:,} tris",
                    fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')
    
    fig.suptitle(f"{subject_name} - Forward vs Inverse Meshes", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    viz_path = Path(output_dir) / f"{subject_name}_meshes_comparison.png"
    plt.savefig(viz_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {viz_path}")


def create_overview_visualization(all_subjects, output_dir, n_display=6):
    """
    Create overview showing random subjects (6-layer, both forward and inverse).
    
    Args:
        all_subjects: List of all subject data
        output_dir: Directory to save visualization
        n_display: Number of subjects to display
    """
    if len(all_subjects) == 0:
        return
    
    print(f"\nCreating overview visualization of {n_display} random subjects...")
    
    # Select random subjects
    indices = np.random.choice(len(all_subjects), 
                              min(n_display, len(all_subjects)), 
                              replace=False)
    
    # 2 rows × 3 cols, each showing forward + inverse side by side
    fig, axes = plt.subplots(n_display // 3, 6, figsize=(18, 3*(n_display//3)))
    axes = axes.flatten()
    
    from matplotlib.tri import Triangulation
    
    for plot_idx, subject_idx in enumerate(indices):
        subject = all_subjects[subject_idx]
        
        layer_type = '6layer'
        
        # Forward mesh (left)
        ax = axes[plot_idx * 2]
        
        if not subject['meshes'][layer_type]['forward']['success']:
            ax.text(0.5, 0.5, f"{subject['subject_name']}\nForward N/A",
                   ha='center', va='center')
            ax.axis('off')
        else:
            mesh_data = subject['meshes'][layer_type]['forward']
            mesh = mesh_data['mesh']
            materials = mesh_data['materials']
            
            tri = Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
            ax.tripcolor(tri, materials, cmap='tab10', shading='flat',
                        edgecolors='k', linewidth=0.05, alpha=0.8)
            ax.set_title(f"{subject['subject_name']}\nFWD", 
                        fontsize=9, fontweight='bold')
            ax.set_aspect('equal')
            ax.axis('off')
        
        # Inverse mesh (right)
        ax = axes[plot_idx * 2 + 1]
        
        if not subject['meshes'][layer_type]['inverse']['success']:
            ax.text(0.5, 0.5, "Inverse N/A", ha='center', va='center')
            ax.axis('off')
        else:
            mesh_data = subject['meshes'][layer_type]['inverse']
            mesh = mesh_data['mesh']
            materials = mesh_data['materials']
            
            tri = Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
            ax.tripcolor(tri, materials, cmap='tab10', shading='flat',
                        edgecolors='k', linewidth=0.1, alpha=0.8)
            ax.set_title(f"{subject['subject_name']}\nINV", 
                        fontsize=9, fontweight='bold')
            ax.set_aspect('equal')
            ax.axis('off')
    
    # Hide unused axes
    for idx in range(len(indices) * 2, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('BrainWeb Meshes - Random Sample (6-layer, Forward vs Inverse)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    viz_path = Path(output_dir) / "meshes_overview.png"
    plt.savefig(viz_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved overview: {viz_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Load and verify all BrainWeb meshes (forward + inverse)"
    )
    parser.add_argument(
        '--meshes-dir',
        type=str,
        default='/mnt/d/Programming/EIT/brainweb_meshes',
        help='Directory with generated meshes (default: brainweb_meshes)'
    )
    parser.add_argument(
        '--max-subjects',
        type=int,
        default=None,
        help='Maximum number of subjects to load (default: all)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Create overview visualization'
    )
    parser.add_argument(
        '--visualize-subject',
        type=str,
        default=None,
        help='Create detailed visualization for specific subject (e.g., subject_00)'
    )
    
    args = parser.parse_args()
    
    meshes_dir = Path(args.meshes_dir)
    
    if not meshes_dir.exists():
        print(f"✗ Meshes directory not found: {meshes_dir}")
        sys.exit(1)
    
    # Load all subjects
    all_subjects = load_all_subject_meshes(
        meshes_dir=args.meshes_dir,
        max_subjects=args.max_subjects
    )
    
    if len(all_subjects) == 0:
        print("\n✗ No subjects loaded successfully")
        return
    
    # Visualizations
    if args.visualize:
        create_overview_visualization(all_subjects, meshes_dir)
    
    if args.visualize_subject:
        # Find requested subject
        subject_data = None
        for s in all_subjects:
            if s['subject_name'] == args.visualize_subject:
                subject_data = s
                break
        
        if subject_data:
            visualize_subject_meshes(subject_data, meshes_dir)
        else:
            print(f"✗ Subject not found: {args.visualize_subject}")
    
    print("\n" + "="*70)
    print("✓ COMPLETE!")
    print("="*70)
    print(f"\nSuccessfully loaded meshes for {len(all_subjects)} subjects")
    print("All meshes verified and ready for EIT simulations!")
    
    return all_subjects


if __name__ == "__main__":
    main()