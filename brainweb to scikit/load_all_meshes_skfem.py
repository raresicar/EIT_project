"""
Load and Verify All BrainWeb Subject Meshes in scikit-fem
Loads meshes for all subjects and all layer types (3, 6, 9)
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
    print("Install with: pip install scikit-fem[all]")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    print("⚠ tqdm not available, using simple progress")
    # Simple fallback if tqdm not installed
    def tqdm(iterable, desc="Processing"):
        print(f"{desc}...")
        return iterable


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
    """
    data = np.load(npz_path, allow_pickle=True)
    
    points = data['points']
    cells = data['cells']
    materials = data['materials']
    
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
    
    return mesh, materials


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
        'valid': bool(n_degenerate == 0)
    }
    
    return results


# =============================================================================
# LOAD SINGLE SUBJECT
# =============================================================================

def load_subject_meshes(subject_dir):
    """
    Load all mesh types for a single subject
    
    Args:
        subject_dir: Path to subject directory
    
    Returns:
        dict with loaded meshes for each layer type
    """
    subject_name = subject_dir.name
    results = {}
    
    for layer_type in ['3layer', '6layer', '9layer']:
        mesh_dir = subject_dir / f"meshes_{layer_type}"
        mesh_file = mesh_dir / "head_mesh.npz"
        
        if not mesh_file.exists():
            results[layer_type] = {
                'success': False,
                'error': f"Mesh file not found: {mesh_file}"
            }
            continue
        
        try:
            # Load mesh
            mesh, materials = load_mesh_to_skfem(mesh_file)
            
            # Validate
            validation = validate_mesh(mesh, materials, f"{subject_name}_{layer_type}")
            
            results[layer_type] = {
                'success': True,
                'mesh': mesh,
                'materials': materials,
                'validation': validation,
                'mesh_file': str(mesh_file)
            }
            
        except Exception as e:
            results[layer_type] = {
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

def load_all_subject_meshes(subjects_base_dir='brainweb_subjects', max_subjects=None):
    """
    Load all meshes from all subjects
    
    Args:
        subjects_base_dir: Base directory containing subject folders
        max_subjects: Maximum number of subjects to load (None = all)
    
    Returns:
        list of dicts with subject mesh info
    """
    print("="*70)
    print("Loading All BrainWeb Subject Meshes into scikit-fem")
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
    print(f"Loading meshes for each subject (3 layer types each)...\n")
    
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
    total_meshes_expected = len(all_subjects) * 3
    
    for layer_type in ['3layer', '6layer', '9layer']:
        successful = sum(1 for s in all_subjects 
                        if s['meshes'][layer_type]['success'])
        total_meshes_loaded += successful
        print(f"\n{layer_type.upper()}:")
        print(f"  Successfully loaded: {successful}/{len(all_subjects)}")
        
        if successful > 0:
            # Get statistics for successfully loaded meshes
            validations = [s['meshes'][layer_type]['validation'] 
                          for s in all_subjects 
                          if s['meshes'][layer_type]['success']]
            
            n_vertices = [v['n_vertices'] for v in validations]
            n_triangles = [v['n_triangles'] for v in validations]
            n_degenerate = [v['n_degenerate'] for v in validations]
            
            print(f"  Vertices:  min={min(n_vertices):,}, max={max(n_vertices):,}, mean={np.mean(n_vertices):,.0f}")
            print(f"  Triangles: min={min(n_triangles):,}, max={max(n_triangles):,}, mean={np.mean(n_triangles):,.0f}")
            
            n_valid = sum(1 for v in validations if v['valid'])
            print(f"  Valid meshes: {n_valid}/{successful}")
            
            if sum(n_degenerate) > 0:
                print(f"  ⚠️ Degenerate triangles: {sum(n_degenerate)} total")
    
    print(f"\nTotal meshes loaded: {total_meshes_loaded}/{total_meshes_expected}")
    
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
            mesh_data = subject['meshes'][layer_type]
            if mesh_data['success']:
                subject_report['layers'][layer_type] = mesh_data['validation']
            else:
                subject_report['layers'][layer_type] = {
                    'error': mesh_data['error']
                }
        
        report['subjects'].append(subject_report)
    
    report_path = subjects_base_dir / "mesh_loading_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Saved loading report: {report_path}")
    
    return all_subjects


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_subject_meshes(subject_data, output_dir):
    """
    Create visualization comparing all layer types for one subject
    
    Args:
        subject_data: Dict with subject mesh data
        output_dir: Directory to save visualization
    """
    subject_name = subject_data['subject_name']
    
    # Check which meshes are available
    available_layers = [layer for layer in ['3layer', '6layer', '9layer']
                       if subject_data['meshes'][layer]['success']]
    
    if len(available_layers) == 0:
        print(f"  No meshes available for {subject_name}")
        return
    
    n_plots = len(available_layers)
    fig, axes = plt.subplots(1, n_plots, figsize=(6*n_plots, 6))
    
    if n_plots == 1:
        axes = [axes]
    
    from matplotlib.tri import Triangulation
    
    for idx, layer_type in enumerate(available_layers):
        mesh_data = subject_data['meshes'][layer_type]
        mesh = mesh_data['mesh']
        materials = mesh_data['materials']
        validation = mesh_data['validation']
        
        ax = axes[idx]
        
        # Create triangulation
        tri = Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
        
        # Plot with material colors
        ax.tripcolor(tri, materials, cmap='tab10', shading='flat',
                    edgecolors='k', linewidth=0.1, alpha=0.8)
        
        # Title
        title = f"{layer_type.upper()}\n{validation['n_vertices']:,} nodes, {validation['n_triangles']:,} tris"
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')
    
    fig.suptitle(f"{subject_name} - All Layer Types", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    viz_path = Path(output_dir) / f"{subject_name}_meshes_comparison.png"
    plt.savefig(viz_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {viz_path}")


def create_overview_visualization(all_subjects, output_dir, n_display=6):
    """
    Create overview visualization showing random subjects
    
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
    
    # Create figure (2 rows x 3 cols for 6 subjects)
    n_rows = 2
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    axes = axes.flatten()
    
    from matplotlib.tri import Triangulation
    
    for plot_idx, subject_idx in enumerate(indices):
        subject = all_subjects[subject_idx]
        ax = axes[plot_idx]
        
        # Use 6-layer mesh for overview
        layer_type = '6layer'
        
        if not subject['meshes'][layer_type]['success']:
            ax.text(0.5, 0.5, f"{subject['subject_name']}\nMesh not available",
                   ha='center', va='center')
            ax.axis('off')
            continue
        
        mesh_data = subject['meshes'][layer_type]
        mesh = mesh_data['mesh']
        materials = mesh_data['materials']
        
        # Plot
        tri = Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
        ax.tripcolor(tri, materials, cmap='tab10', shading='flat',
                    edgecolors='k', linewidth=0.1, alpha=0.8)
        
        ax.set_title(f"{subject['subject_name']}", fontsize=10, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')
    
    # Hide unused axes
    for idx in range(len(indices), len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('BrainWeb Subject Meshes - Random Sample (6-layer)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    viz_path = Path(output_dir) / "subjects_overview.png"
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
        description="Load and verify all BrainWeb subject meshes in scikit-fem"
    )
    parser.add_argument(
        '--subjects-dir',
        type=str,
        default='brainweb_subjects',
        help='Base directory containing subject folders (default: brainweb_subjects)'
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
        help='Create visualizations'
    )
    parser.add_argument(
        '--visualize-subject',
        type=str,
        default=None,
        help='Create detailed visualization for specific subject (e.g., subject_00)'
    )
    
    args = parser.parse_args()
    
    subjects_dir = Path(args.subjects_dir)
    
    if not subjects_dir.exists():
        print(f"✗ Subjects directory not found: {subjects_dir}")
        sys.exit(1)
    
    # Load all subjects
    all_subjects = load_all_subject_meshes(
        subjects_base_dir=args.subjects_dir,
        max_subjects=args.max_subjects
    )
    
    if len(all_subjects) == 0:
        print("\n✗ No subjects loaded successfully")
        return
    
    # Visualizations
    if args.visualize:
        create_overview_visualization(all_subjects, subjects_dir)
    
    if args.visualize_subject:
        # Find the requested subject
        subject_data = None
        for s in all_subjects:
            if s['subject_name'] == args.visualize_subject:
                subject_data = s
                break
        
        if subject_data:
            visualize_subject_meshes(subject_data, subjects_dir)
        else:
            print(f"✗ Subject not found: {args.visualize_subject}")
    
    print("\n" + "="*70)
    print("✓ COMPLETE!")
    print("="*70)
    print(f"\nSuccessfully loaded meshes for {len(all_subjects)} subjects")
    print("All meshes are now in scikit-fem format and ready for EIT simulations!")
    
    return all_subjects


if __name__ == "__main__":
    main()