"""
Generate multiple conductivity distributions with strokes for subject_00.

This script creates 5 samples each for 3-layer and 6-layer head models:
- Each sample has randomized conductivity values based on the provided table
- Each sample includes a circular stroke (ischemic or hemorrhagic)
- For 6-layer models, strokes are placed in white matter (material ID 5)
- All samples are exported with conductivity visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from pathlib import Path
import time

from eit_forward_skfem import (
    EIT, current_method, load_brainweb_mesh,
    materials_to_conductivity, MeshTri
)


# Conductivity distribution parameters from Table 1
# Format: (mean, std_dev) for normal distribution N(μ, σ²)
CONDUCTIVITY_PARAMS = {
    'scalp': (1.0, 0.0333),           # Normalized to scalp = 1.0
    'skull': (0.0625, 0.0021),        # ~N(0.0625, 0.0021)
    'csf': (6.25, 0.2083),            # ~N(6.25, 0.2083)
    'grey_matter': (0.3063, 0.0102),  # ~N(0.3063, 0.0102)
    'white_matter': (0.1938, 0.0065), # ~N(0.1938, 0.0065)
    'ischemic_stroke': (0.0938, 0.0031),   # ~N(0.0938, 0.0031)
    'hemorrhagic_stroke': (2.1875, 0.0729), # ~N(2.1875, 0.0729)
}


def sample_conductivity(param_name, n_samples=1):
    """Sample conductivity from normal distribution."""
    mean, std = CONDUCTIVITY_PARAMS[param_name]
    samples = np.random.normal(mean, std, n_samples)
    # Ensure positive values
    samples = np.maximum(samples, 0.001)
    return samples


def create_circular_stroke(mesh, materials, stroke_center, stroke_radius, stroke_type='ischemic', target_materials=None):
    """
    Create a circular stroke in the mesh.

    Args:
        mesh: scikit-fem MeshTri
        materials: Material IDs per element
        stroke_center: (x, y) coordinates of stroke center
        stroke_radius: Radius of stroke
        stroke_type: 'ischemic' or 'hemorrhagic'
        target_materials: List of material IDs where stroke can be placed (None = all materials)

    Returns:
        Modified materials array with stroke
    """
    materials_stroke = materials.copy()

    # Calculate element centroids
    n_elements = mesh.t.shape[1]
    centroids = np.zeros((n_elements, 2))

    for elem_idx in range(n_elements):
        nodes = mesh.t[:, elem_idx]
        coords = mesh.p[:, nodes]
        centroids[elem_idx] = coords.mean(axis=1)

    # Find elements within stroke radius
    distances = np.linalg.norm(centroids - stroke_center, axis=1)
    stroke_mask = distances <= stroke_radius

    # If target_materials specified, only affect those elements
    if target_materials is not None:
        material_mask = np.isin(materials, target_materials)
        stroke_mask = stroke_mask & material_mask

    # Assign stroke marker (use material ID 10 for ischemic, 11 for hemorrhagic)
    if stroke_type == 'ischemic':
        materials_stroke[stroke_mask] = 10
    else:  # hemorrhagic
        materials_stroke[stroke_mask] = 11

    n_affected = stroke_mask.sum()
    print(f"  Stroke affects {n_affected} elements")

    return materials_stroke


def find_stroke_location_6layer(mesh, materials, brain_material_ids=[4, 5]):
    """
    Find a suitable location for stroke in brain tissue (6-layer model).

    Args:
        brain_material_ids: List of material IDs for grey (4) and white (5) matter

    Returns:
        center: (x, y) coordinates
        radius: suitable radius
    """
    # Get all elements with grey or white matter
    brain_mask = np.isin(materials, brain_material_ids)
    brain_elements = np.where(brain_mask)[0]

    if len(brain_elements) == 0:
        raise ValueError("No brain tissue elements found!")

    # Calculate centroids of brain elements
    centroids = []
    for elem_idx in brain_elements:
        nodes = mesh.t[:, elem_idx]
        coords = mesh.p[:, nodes]
        centroid = coords.mean(axis=1)
        centroids.append(centroid)

    centroids = np.array(centroids)

    # Choose a random location in brain, slightly biased toward center
    brain_center = centroids.mean(axis=0)
    brain_std = centroids.std(axis=0)

    # Sample location with some randomness
    center = brain_center + np.random.randn(2) * brain_std * 0.3

    # Calculate appropriate radius (small stroke, ~2-5% of brain size)
    brain_size = np.linalg.norm(centroids.max(axis=0) - centroids.min(axis=0))
    radius = brain_size * np.random.uniform(0.02, 0.05)

    return center, radius


def find_stroke_location_3layer(mesh, materials, brain_id=3):
    """
    Find a suitable location for stroke in brain (3-layer model).

    Returns:
        center: (x, y) coordinates
        radius: suitable radius
    """
    # Get all elements with brain tissue
    brain_mask = materials == brain_id
    brain_elements = np.where(brain_mask)[0]

    if len(brain_elements) == 0:
        raise ValueError("No brain elements found!")

    # Calculate centroids of brain elements
    centroids = []
    for elem_idx in brain_elements:
        nodes = mesh.t[:, elem_idx]
        coords = mesh.p[:, nodes]
        centroid = coords.mean(axis=1)
        centroids.append(centroid)

    centroids = np.array(centroids)

    # Choose a random location in brain
    brain_center = centroids.mean(axis=0)
    brain_std = centroids.std(axis=0)

    center = brain_center + np.random.randn(2) * brain_std * 0.3

    # Calculate appropriate radius
    brain_size = np.linalg.norm(centroids.max(axis=0) - centroids.min(axis=0))
    radius = brain_size * np.random.uniform(0.02, 0.05)

    return center, radius


def create_conductivity_distribution_6layer(materials_modified):
    """
    Create conductivity distribution for 6-layer model with stroke.

    Material IDs:
        0: Background
        1: Scalp
        2: Skull
        3: CSF
        4: Grey matter
        5: White matter
        6: Ventricles (use CSF conductivity)
        10: Ischemic stroke
        11: Hemorrhagic stroke
    """
    n_elements = len(materials_modified)
    sigma = np.zeros(n_elements)

    # Sample conductivities for this distribution
    scalp_cond = sample_conductivity('scalp', 1)[0]
    skull_cond = sample_conductivity('skull', 1)[0]
    csf_cond = sample_conductivity('csf', 1)[0]
    grey_cond = sample_conductivity('grey_matter', 1)[0]
    white_cond = sample_conductivity('white_matter', 1)[0]
    ischemic_cond = sample_conductivity('ischemic_stroke', 1)[0]
    hemorrhagic_cond = sample_conductivity('hemorrhagic_stroke', 1)[0]

    # Assign conductivities (multiply by 0.36 to get actual S/m values)
    for elem_idx in range(n_elements):
        mat_id = materials_modified[elem_idx]
        if mat_id == 0:  # Background
            sigma[elem_idx] = 0.01
        elif mat_id == 1:  # Scalp
            sigma[elem_idx] = scalp_cond * 0.36
        elif mat_id == 2:  # Skull
            sigma[elem_idx] = skull_cond * 0.36
        elif mat_id == 3:  # CSF
            sigma[elem_idx] = csf_cond * 0.36
        elif mat_id == 4:  # Grey matter
            sigma[elem_idx] = grey_cond * 0.36
        elif mat_id == 5:  # White matter
            sigma[elem_idx] = white_cond * 0.36
        elif mat_id == 6:  # Ventricles
            sigma[elem_idx] = csf_cond * 0.36
        elif mat_id == 10:  # Ischemic stroke
            sigma[elem_idx] = ischemic_cond * 0.36
        elif mat_id == 11:  # Hemorrhagic stroke
            sigma[elem_idx] = hemorrhagic_cond * 0.36
        else:
            sigma[elem_idx] = 0.1  # Default

    return sigma


def create_conductivity_distribution_3layer(materials_modified):
    """
    Create conductivity distribution for 3-layer model with stroke.

    Material IDs:
        0: Background
        1: Scalp
        2: Skull
        3: Brain (use average of grey/white matter)
        10: Ischemic stroke
        11: Hemorrhagic stroke
    """
    n_elements = len(materials_modified)
    sigma = np.zeros(n_elements)

    # Sample conductivities
    scalp_cond = sample_conductivity('scalp', 1)[0]
    skull_cond = sample_conductivity('skull', 1)[0]
    grey_cond = sample_conductivity('grey_matter', 1)[0]
    white_cond = sample_conductivity('white_matter', 1)[0]
    brain_cond = (grey_cond + white_cond) / 2  # Average for 3-layer brain
    ischemic_cond = sample_conductivity('ischemic_stroke', 1)[0]
    hemorrhagic_cond = sample_conductivity('hemorrhagic_stroke', 1)[0]

    # Assign conductivities
    for elem_idx in range(n_elements):
        mat_id = materials_modified[elem_idx]
        if mat_id == 0:  # Background
            sigma[elem_idx] = 0.01
        elif mat_id == 1:  # Scalp
            sigma[elem_idx] = scalp_cond * 0.36
        elif mat_id == 2:  # Skull
            sigma[elem_idx] = skull_cond * 0.36
        elif mat_id == 3:  # Brain
            sigma[elem_idx] = brain_cond * 0.36
        elif mat_id == 10:  # Ischemic stroke
            sigma[elem_idx] = ischemic_cond * 0.36
        elif mat_id == 11:  # Hemorrhagic stroke
            sigma[elem_idx] = hemorrhagic_cond * 0.36
        else:
            sigma[elem_idx] = 0.1  # Default

    return sigma


def plot_conductivity_with_stroke(mesh, sigma, materials_modified, title, save_path):
    """Plot conductivity distribution with stroke highlighted."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    tri = Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)

    # Plot conductivity
    im1 = ax1.tripcolor(tri, sigma, shading='flat', cmap='viridis')
    ax1.set_aspect('equal')
    ax1.set_title(f'{title} - Conductivity (S/m)')
    plt.colorbar(im1, ax=ax1, label='σ (S/m)')

    # Plot materials with stroke highlighted
    # Create color map: stroke=red, other materials=blue scale
    materials_viz = materials_modified.copy().astype(float)
    stroke_mask = (materials_modified == 10) | (materials_modified == 11)
    materials_viz[stroke_mask] = materials_viz.max() + 1  # Separate color for stroke

    im2 = ax2.tripcolor(tri, materials_viz, shading='flat', cmap='tab10')
    ax2.set_aspect('equal')
    ax2.set_title(f'{title} - Materials (stroke highlighted)')
    plt.colorbar(im2, ax=ax2, label='Material ID')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def generate_samples():
    """Generate all stroke samples."""
    # Set random seed for reproducibility (optional - comment out for different results each run)
    np.random.seed(42)

    # Create output directory
    output_dir = Path("brainweb_stroke_samples")
    output_dir.mkdir(exist_ok=True)

    print("="*70)
    print("Generating Stroke Samples for Subject 00")
    print("="*70)

    # ====================
    # 3-LAYER MODEL
    # ====================
    print("\n" + "="*70)
    print("3-LAYER HEAD MODEL")
    print("="*70)

    mesh_path_3layer = "/mnt/d/Programming/EIT/brainweb_subjects/subject_00/meshes_3layer/head_mesh.npz"
    mesh_3layer, electrode_markers_3layer, materials_3layer = load_brainweb_mesh(
        mesh_path_3layer, n_electrodes=16
    )

    print(f"Loaded 3-layer mesh: {mesh_3layer.p.shape[1]} nodes, {mesh_3layer.t.shape[1]} elements")
    print(f"Material IDs: {np.unique(materials_3layer)}")

    # Generate 5 samples
    for sample_idx in range(5):
        print(f"\n--- Sample {sample_idx + 1}/5 ---")

        # Randomly choose stroke type
        stroke_type = np.random.choice(['ischemic', 'hemorrhagic'])
        print(f"  Stroke type: {stroke_type}")

        # Find stroke location
        stroke_center, stroke_radius = find_stroke_location_3layer(mesh_3layer, materials_3layer)
        print(f"  Stroke center: ({stroke_center[0]:.4f}, {stroke_center[1]:.4f})")
        print(f"  Stroke radius: {stroke_radius:.4f}")

        # Create stroke
        materials_with_stroke = create_circular_stroke(
            mesh_3layer, materials_3layer,
            stroke_center, stroke_radius,
            stroke_type=stroke_type,
            target_materials=[3]  # Brain
        )

        # Generate conductivity distribution
        sigma = create_conductivity_distribution_3layer(materials_with_stroke)
        print(f"  Conductivity range: [{sigma.min():.4f}, {sigma.max():.4f}] S/m")

        # Save mesh and conductivity
        output_file = output_dir / f"subject_00_3layer_sample_{sample_idx + 1}_{stroke_type}.npz"
        np.savez(
            output_file,
            points=mesh_3layer.p.T,
            cells=mesh_3layer.t.T,
            materials=materials_with_stroke,
            conductivity=sigma,
            stroke_center=stroke_center,
            stroke_radius=stroke_radius,
            stroke_type=stroke_type
        )
        print(f"  Saved: {output_file}")

        # Plot
        plot_path = output_dir / f"subject_00_3layer_sample_{sample_idx + 1}_{stroke_type}.png"
        plot_conductivity_with_stroke(
            mesh_3layer, sigma, materials_with_stroke,
            f"3-Layer Sample {sample_idx + 1} ({stroke_type.capitalize()} Stroke)",
            plot_path
        )

    # ====================
    # 6-LAYER MODEL
    # ====================
    print("\n" + "="*70)
    print("6-LAYER HEAD MODEL")
    print("="*70)

    mesh_path_6layer = "/mnt/d/Programming/EIT/brainweb_subjects/subject_00/meshes_6layer/head_mesh.npz"
    mesh_6layer, electrode_markers_6layer, materials_6layer = load_brainweb_mesh(
        mesh_path_6layer, n_electrodes=16
    )

    print(f"Loaded 6-layer mesh: {mesh_6layer.p.shape[1]} nodes, {mesh_6layer.t.shape[1]} elements")
    print(f"Material IDs: {np.unique(materials_6layer)}")

    # Generate 5 samples
    for sample_idx in range(5):
        print(f"\n--- Sample {sample_idx + 1}/5 ---")

        # Randomly choose stroke type
        stroke_type = np.random.choice(['ischemic', 'hemorrhagic'])
        print(f"  Stroke type: {stroke_type}")

        # Find stroke location in brain tissue (grey/white matter)
        stroke_center, stroke_radius = find_stroke_location_6layer(mesh_6layer, materials_6layer)
        print(f"  Stroke center: ({stroke_center[0]:.4f}, {stroke_center[1]:.4f})")
        print(f"  Stroke radius: {stroke_radius:.4f}")

        # Create circular stroke (can cover both grey and white matter)
        materials_with_stroke = create_circular_stroke(
            mesh_6layer, materials_6layer,
            stroke_center, stroke_radius,
            stroke_type=stroke_type,
            target_materials=[4, 5]  # Grey and white matter
        )

        # Generate conductivity distribution
        sigma = create_conductivity_distribution_6layer(materials_with_stroke)
        print(f"  Conductivity range: [{sigma.min():.4f}, {sigma.max():.4f}] S/m")

        # Save mesh and conductivity
        output_file = output_dir / f"subject_00_6layer_sample_{sample_idx + 1}_{stroke_type}.npz"
        np.savez(
            output_file,
            points=mesh_6layer.p.T,
            cells=mesh_6layer.t.T,
            materials=materials_with_stroke,
            conductivity=sigma,
            stroke_center=stroke_center,
            stroke_radius=stroke_radius,
            stroke_type=stroke_type
        )
        print(f"  Saved: {output_file}")

        # Plot
        plot_path = output_dir / f"subject_00_6layer_sample_{sample_idx + 1}_{stroke_type}.png"
        plot_conductivity_with_stroke(
            mesh_6layer, sigma, materials_with_stroke,
            f"6-Layer Sample {sample_idx + 1} ({stroke_type.capitalize()} Stroke)",
            plot_path
        )

    print("\n" + "="*70)
    print("✓ All samples generated successfully!")
    print(f"✓ Output directory: {output_dir.absolute()}")
    print("="*70)


if __name__ == "__main__":
    generate_samples()
