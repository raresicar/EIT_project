"""
Generate Stroke Samples for EIT Reconstruction Testing
(Legacy — superseded by monitoring_data_generator.py for training data)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from pathlib import Path
import sys

# --- Project-relative imports via eit_config ---
_dir = Path(__file__).resolve().parent
while not (_dir / "eit_config.py").exists():
    _dir = _dir.parent
sys.path.insert(0, str(_dir))
from eit_config import *

try:
    from skfem import MeshTri
    print("✓ scikit-fem available")
except ImportError:
    print("✗ scikit-fem not found. Install with: pip install scikit-fem")
    sys.exit(1)


# Conductivity distribution parameters (normalized to scalp = 1.0)
CONDUCTIVITY_PARAMS = {
    'scalp': (1.0, 0.0333),
    'skull': (0.0625, 0.0021),
    'csf': (6.25, 0.2083),
    'grey_matter': (0.3063, 0.0102),
    'white_matter': (0.1938, 0.0065),
    'ischemic_stroke': (0.0938, 0.0031),
    'hemorrhagic_stroke': (2.1875, 0.0729),
}

SCALP_CONDUCTIVITY = 0.36  # S/m at 100 kHz


def sample_conductivity(param_name, n_samples=1):
    mean, std = CONDUCTIVITY_PARAMS[param_name]
    samples = np.random.normal(mean, std, n_samples)
    return np.maximum(samples, 0.001)


def load_mesh(npz_path):
    data = np.load(npz_path)
    points = data['points']
    cells = data['cells']
    materials = data['materials']
    mesh = MeshTri(points.T, cells.T)
    return mesh, materials


def get_brain_region_size(mesh, materials, brain_material_ids):
    brain_mask = np.isin(materials, brain_material_ids)
    brain_elements = np.where(brain_mask)[0]
    if len(brain_elements) == 0:
        raise ValueError("No brain elements found!")
    centroids = []
    for elem_idx in brain_elements:
        nodes = mesh.t[:, elem_idx]
        coords = mesh.p[:, nodes]
        centroid = coords.mean(axis=1)
        centroids.append(centroid)
    centroids = np.array(centroids)
    brain_min = centroids.min(axis=0)
    brain_max = centroids.max(axis=0)
    brain_size = np.linalg.norm(brain_max - brain_min)
    return brain_size, centroids


def find_stroke_location(mesh, materials, brain_material_ids, brain_size):
    brain_mask = np.isin(materials, brain_material_ids)
    brain_elements = np.where(brain_mask)[0]
    centroids = []
    for elem_idx in brain_elements:
        nodes = mesh.t[:, elem_idx]
        coords = mesh.p[:, nodes]
        centroid = coords.mean(axis=1)
        centroids.append(centroid)
    centroids = np.array(centroids)
    brain_center = centroids.mean(axis=0)
    brain_std = centroids.std(axis=0)
    center = brain_center + np.random.randn(2) * brain_std * 0.3
    radius = brain_size * np.random.uniform(0.02, 0.04)
    return center, radius


def create_stroke_in_mesh(mesh, materials, stroke_center, stroke_radius,
                         stroke_type, brain_material_ids):
    materials_stroke = materials.copy()
    n_elements = mesh.t.shape[1]
    centroids = np.zeros((n_elements, 2))
    for elem_idx in range(n_elements):
        nodes = mesh.t[:, elem_idx]
        coords = mesh.p[:, nodes]
        centroids[elem_idx] = coords.mean(axis=1)
    distances = np.linalg.norm(centroids - stroke_center, axis=1)
    stroke_mask = distances <= stroke_radius
    brain_mask = np.isin(materials, brain_material_ids)
    stroke_mask = stroke_mask & brain_mask
    stroke_id = 10 if stroke_type == 'ischemic' else 11
    materials_stroke[stroke_mask] = stroke_id
    n_affected = stroke_mask.sum()
    return materials_stroke, n_affected


def create_conductivity_3layer(materials_modified):
    n_elements = len(materials_modified)
    sigma = np.zeros(n_elements)
    scalp_cond = sample_conductivity('scalp', 1)[0]
    skull_cond = sample_conductivity('skull', 1)[0]
    grey_cond = sample_conductivity('grey_matter', 1)[0]
    white_cond = sample_conductivity('white_matter', 1)[0]
    brain_cond = (grey_cond + white_cond) / 2
    ischemic_cond = sample_conductivity('ischemic_stroke', 1)[0]
    hemorrhagic_cond = sample_conductivity('hemorrhagic_stroke', 1)[0]
    for elem_idx in range(n_elements):
        mat_id = materials_modified[elem_idx]
        if mat_id == 0:     sigma[elem_idx] = 0.01
        elif mat_id == 1:   sigma[elem_idx] = scalp_cond * SCALP_CONDUCTIVITY
        elif mat_id == 2:   sigma[elem_idx] = skull_cond * SCALP_CONDUCTIVITY
        elif mat_id == 3:   sigma[elem_idx] = brain_cond * SCALP_CONDUCTIVITY
        elif mat_id == 10:  sigma[elem_idx] = ischemic_cond * SCALP_CONDUCTIVITY
        elif mat_id == 11:  sigma[elem_idx] = hemorrhagic_cond * SCALP_CONDUCTIVITY
        else:               sigma[elem_idx] = 0.1
    return sigma


def create_conductivity_6layer(materials_modified):
    n_elements = len(materials_modified)
    sigma = np.zeros(n_elements)
    scalp_cond = sample_conductivity('scalp', 1)[0]
    skull_cond = sample_conductivity('skull', 1)[0]
    csf_cond = sample_conductivity('csf', 1)[0]
    grey_cond = sample_conductivity('grey_matter', 1)[0]
    white_cond = sample_conductivity('white_matter', 1)[0]
    ischemic_cond = sample_conductivity('ischemic_stroke', 1)[0]
    hemorrhagic_cond = sample_conductivity('hemorrhagic_stroke', 1)[0]
    for elem_idx in range(n_elements):
        mat_id = materials_modified[elem_idx]
        if mat_id == 0:     sigma[elem_idx] = 0.01
        elif mat_id == 1:   sigma[elem_idx] = scalp_cond * SCALP_CONDUCTIVITY
        elif mat_id == 2:   sigma[elem_idx] = skull_cond * SCALP_CONDUCTIVITY
        elif mat_id == 3:   sigma[elem_idx] = csf_cond * SCALP_CONDUCTIVITY
        elif mat_id == 4:   sigma[elem_idx] = grey_cond * SCALP_CONDUCTIVITY
        elif mat_id == 5:   sigma[elem_idx] = white_cond * SCALP_CONDUCTIVITY
        elif mat_id == 6:   sigma[elem_idx] = csf_cond * SCALP_CONDUCTIVITY
        elif mat_id == 10:  sigma[elem_idx] = ischemic_cond * SCALP_CONDUCTIVITY
        elif mat_id == 11:  sigma[elem_idx] = hemorrhagic_cond * SCALP_CONDUCTIVITY
        else:               sigma[elem_idx] = 0.1
    return sigma


def visualize_sample(mesh, sigma, materials_stroke, title, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    tri = Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
    ax = axes[0]
    im = ax.tripcolor(tri, sigma, shading='flat', cmap='viridis')
    ax.set_aspect('equal'); ax.set_title('Conductivity (S/m)', fontweight='bold', fontsize=12)
    ax.axis('off'); plt.colorbar(im, ax=ax, label='σ (S/m)')
    ax = axes[1]
    im = ax.tripcolor(tri, materials_stroke, shading='flat', cmap='tab10')
    ax.set_aspect('equal'); ax.set_title('Materials (stroke highlighted)', fontweight='bold', fontsize=12)
    ax.axis('off'); plt.colorbar(im, ax=ax, label='Material ID')
    ax = axes[2]
    stroke_mask = (materials_stroke == 10) | (materials_stroke == 11)
    stroke_viz = stroke_mask.astype(float)
    im = ax.tripcolor(tri, stroke_viz, shading='flat', cmap='RdYlBu_r', vmin=0, vmax=1)
    ax.set_aspect('equal'); ax.set_title('Stroke Region', fontweight='bold', fontsize=12)
    ax.axis('off'); plt.colorbar(im, ax=ax, label='Stroke')
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(save_path, dpi=200, bbox_inches='tight'); plt.close()


def generate_sample(mesh_fwd, materials_fwd, mesh_inv, materials_inv,
                    sample_idx, stroke_type, output_dir, layer_type, brain_material_ids):
    sample_name = f"sample_{sample_idx:02d}_{stroke_type}"
    sample_dir = Path(output_dir) / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Generating {sample_name}...")
    brain_size, _ = get_brain_region_size(mesh_fwd, materials_fwd, brain_material_ids)
    stroke_center, stroke_radius = find_stroke_location(mesh_fwd, materials_fwd, brain_material_ids, brain_size)
    print(f"    Stroke center: ({stroke_center[0]:.2f}, {stroke_center[1]:.2f})")
    print(f"    Stroke radius: {stroke_radius:.2f} mm ({stroke_radius/brain_size*100:.1f}% of brain)")
    materials_fwd_stroke, n_affected_fwd = create_stroke_in_mesh(mesh_fwd, materials_fwd, stroke_center, stroke_radius, stroke_type, brain_material_ids)
    print(f"    Forward mesh: {n_affected_fwd} elements affected")
    materials_inv_stroke, n_affected_inv = create_stroke_in_mesh(mesh_inv, materials_inv, stroke_center, stroke_radius, stroke_type, brain_material_ids)
    print(f"    Inverse mesh: {n_affected_inv} elements affected")
    if layer_type == '3layer':
        sigma_fwd = create_conductivity_3layer(materials_fwd_stroke)
        sigma_inv = create_conductivity_3layer(materials_inv_stroke)
    else:
        sigma_fwd = create_conductivity_6layer(materials_fwd_stroke)
        sigma_inv = create_conductivity_6layer(materials_inv_stroke)
    print(f"    Conductivity range: [{sigma_fwd.min():.4f}, {sigma_fwd.max():.4f}] S/m")
    np.savez_compressed(sample_dir / "mesh_forward.npz", points=mesh_fwd.p.T, cells=mesh_fwd.t.T, materials=materials_fwd_stroke, conductivity=sigma_fwd, stroke_center=stroke_center, stroke_radius=stroke_radius, stroke_type=stroke_type)
    np.savez_compressed(sample_dir / "mesh_inverse.npz", points=mesh_inv.p.T, cells=mesh_inv.t.T, materials=materials_inv_stroke, conductivity=sigma_inv, stroke_center=stroke_center, stroke_radius=stroke_radius, stroke_type=stroke_type)
    np.savez_compressed(sample_dir / "conductivity.npz", sigma_forward=sigma_fwd, sigma_inverse=sigma_inv, materials_forward=materials_fwd_stroke, materials_inverse=materials_inv_stroke, stroke_center=stroke_center, stroke_radius=stroke_radius, stroke_type=stroke_type)
    viz_path = sample_dir / "visualization.png"
    visualize_sample(mesh_fwd, sigma_fwd, materials_fwd_stroke, f"{layer_type.upper()} - {sample_name}", viz_path)
    print(f"    ✓ Saved to {sample_dir}")


def generate_all_samples(subject='subject_00', mesh_base_dir=None, output_dir=None, n_samples_per_layer=5):
    if mesh_base_dir is None:
        mesh_base_dir = str(BRAINWEB_MESHES_DIR)
    if output_dir is None:
        output_dir = str(STROKE_DATA_DIR)

    print("="*70)
    print(f"Generating Stroke Samples for {subject}")
    print("="*70)
    np.random.seed(42)
    mesh_base_dir = Path(mesh_base_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 3-layer
    print("\n" + "="*70); print("3-LAYER MODEL"); print("="*70)
    layer_dir = output_dir / "3layer"; layer_dir.mkdir(exist_ok=True)
    mesh_fwd_path = mesh_base_dir / subject / "3layer" / "forward" / "head_mesh.npz"
    mesh_inv_path = mesh_base_dir / subject / "3layer" / "inverse" / "head_mesh.npz"
    print(f"\nLoading meshes from {mesh_base_dir / subject / '3layer'}...")
    mesh_fwd, materials_fwd = load_mesh(mesh_fwd_path)
    mesh_inv, materials_inv = load_mesh(mesh_inv_path)
    print(f"Forward mesh: {mesh_fwd.p.shape[1]} nodes, {mesh_fwd.t.shape[1]} elements")
    print(f"Inverse mesh: {mesh_inv.p.shape[1]} nodes, {mesh_inv.t.shape[1]} elements")
    brain_material_ids = [3]
    stroke_types = ['ischemic', 'hemorrhagic', 'ischemic', 'hemorrhagic', 'ischemic']
    for i in range(n_samples_per_layer):
        generate_sample(mesh_fwd, materials_fwd, mesh_inv, materials_inv, i + 1, stroke_types[i], layer_dir, '3layer', brain_material_ids)
    
    # 6-layer
    print("\n" + "="*70); print("6-LAYER MODEL"); print("="*70)
    layer_dir = output_dir / "6layer"; layer_dir.mkdir(exist_ok=True)
    mesh_fwd_path = mesh_base_dir / subject / "6layer" / "forward" / "head_mesh.npz"
    mesh_inv_path = mesh_base_dir / subject / "6layer" / "inverse" / "head_mesh.npz"
    print(f"\nLoading meshes from {mesh_base_dir / subject / '6layer'}...")
    mesh_fwd, materials_fwd = load_mesh(mesh_fwd_path)
    mesh_inv, materials_inv = load_mesh(mesh_inv_path)
    print(f"Forward mesh: {mesh_fwd.p.shape[1]} nodes, {mesh_fwd.t.shape[1]} elements")
    print(f"Inverse mesh: {mesh_inv.p.shape[1]} nodes, {mesh_inv.t.shape[1]} elements")
    brain_material_ids = [4, 5]
    for i in range(n_samples_per_layer):
        generate_sample(mesh_fwd, materials_fwd, mesh_inv, materials_inv, i + 1, stroke_types[i], layer_dir, '6layer', brain_material_ids)
    
    print("\n" + "="*70); print("✓ COMPLETE!"); print("="*70)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate stroke samples for EIT testing")
    parser.add_argument('--subject', default='subject_00')
    parser.add_argument('--mesh-dir', default=str(BRAINWEB_MESHES_DIR))
    parser.add_argument('--output-dir', default=str(STROKE_DATA_DIR))
    parser.add_argument('--n-samples', type=int, default=5)
    args = parser.parse_args()
    generate_all_samples(subject=args.subject, mesh_base_dir=args.mesh_dir, output_dir=args.output_dir, n_samples_per_layer=args.n_samples)


if __name__ == "__main__":
    main()