"""
Comprehensive BrainWeb Head Model Generation
Processes ALL 20 subjects with 3-layer, 6-layer, and 9-layer models
Cleans models to remove background pixels
Saves organized in subject folders
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import binary_erosion, binary_fill_holes, distance_transform_edt
from pathlib import Path
from tqdm import tqdm

# Check imports
print("Checking imports...")
try:
    import brainweb
    from brainweb import Act
    print(f"✓ BrainWeb version: {brainweb.__version__}")
    print("✓ All imports successful!\n")
except ImportError:
    print("✗ BrainWeb not found. Install with: pip install brainweb")
    sys.exit(1)


# =============================================================================
# DATA LOADING
# =============================================================================

def download_all_subjects():
    """Download all BrainWeb subjects"""
    print("="*70)
    print("Downloading BrainWeb Dataset")
    print("="*70)
    print("\nDownloading all subjects...")
    files = brainweb.get_files()
    print(f"✓ Available subjects: {len(files)}")
    return files


def load_subject_slice(subject_file, slice_type='middle'):
    """
    Load a 2D slice from a subject
    
    Args:
        subject_file: Path to subject file
        slice_type: 'middle', 'upper', 'lower', or specific index
    
    Returns:
        slice_2d: 2D array
    """
    raw_data = brainweb.load_file(subject_file)
    
    if slice_type == 'middle':
        slice_idx = raw_data.shape[0] // 2
    elif slice_type == 'upper':
        slice_idx = raw_data.shape[0] // 3
    elif slice_type == 'lower':
        slice_idx = 2 * raw_data.shape[0] // 3
    else:
        slice_idx = int(slice_type)
    
    slice_2d = raw_data[slice_idx, :, :]
    
    return slice_2d


# =============================================================================
# MODEL CLEANING
# =============================================================================

def clean_model_minimal(model):
    """
    Minimal cleaning:
      - remove very small isolated background pixels
      - fill enclosed background holes with nearest tissue label
    This keeps anatomical structure but ensures layers form a clean,
    non-overlapping partition of the head region (no internal gaps).
    
    Args:
        model: 2D array of tissue labels
    
    Returns:
        cleaned_model: Model with minimal cleanup
    """
    from scipy.ndimage import label as nd_label

    cleaned = model.copy()

    # ------------------------------------------------------------------
    # 1) OLD BEHAVIOUR: remove tiny isolated background "speckles"
    # ------------------------------------------------------------------
    background_mask = (cleaned == 0)
    labeled_bg, n_regions = nd_label(background_mask)

    for region_id in range(1, n_regions + 1):
        region_mask = (labeled_bg == region_id)
        region_size = region_mask.sum()

        # Only fill very tiny regions (< 10 pixels)
        if region_size < 10:
            # Find nearest tissue
            y_coords, x_coords = region_mask.nonzero()
            if len(y_coords) > 0:
                y, x = y_coords[0], x_coords[0]

                # Search in small radius for nearest tissue
                found = False
                for radius in range(1, 5):
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            ny, nx = y + dy, x + dx
                            if (0 <= ny < cleaned.shape[0] and
                                0 <= nx < cleaned.shape[1] and
                                cleaned[ny, nx] > 0):
                                cleaned[region_mask] = cleaned[ny, nx]
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break

    # ------------------------------------------------------------------
    # 2) NEW: fill interior background "holes" inside the head
    # ------------------------------------------------------------------
    tissue_mask = cleaned > 0

    # Fill holes inside the main head region (outside stays background)
    filled_head = binary_fill_holes(tissue_mask)
    hole_mask = filled_head & ~tissue_mask    # background pixels inside head

    if hole_mask.any():
        # Map each hole pixel to its nearest tissue pixel
        # (distance computed in the complement of tissue_mask)
        _, indices = distance_transform_edt(~tissue_mask, return_indices=True)
        ys = indices[0][hole_mask]
        xs = indices[1][hole_mask]

        cleaned[hole_mask] = cleaned[ys, xs]

    return cleaned

# def clean_model_minimal(model):
#     """
#     Minimal cleaning: only remove very small isolated background pixels
#     Does NOT fill holes to preserve anatomical accuracy
    
#     Args:
#         model: 2D array of tissue labels
    
#     Returns:
#         cleaned_model: Model with minimal cleanup
#     """
#     from scipy.ndimage import label as nd_label
    
#     cleaned = model.copy()
    
#     # Only remove tiny isolated background regions (< 10 pixels)
#     # These are likely noise/artifacts, not anatomical gaps
#     background_mask = (cleaned == 0)
#     labeled_bg, n_regions = nd_label(background_mask)
    
#     for region_id in range(1, n_regions + 1):
#         region_mask = (labeled_bg == region_id)
#         region_size = region_mask.sum()
        
#         # Only fill very tiny regions (< 10 pixels)
#         if region_size < 10:
#             # Find nearest tissue
#             y_coords, x_coords = region_mask.nonzero()
#             if len(y_coords) > 0:
#                 # Use first pixel to find nearest tissue
#                 y, x = y_coords[0], x_coords[0]
                
#                 # Search in small radius for nearest tissue
#                 found = False
#                 for radius in range(1, 5):
#                     for dy in range(-radius, radius + 1):
#                         for dx in range(-radius, radius + 1):
#                             ny, nx = y + dy, x + dx
#                             if (0 <= ny < cleaned.shape[0] and 
#                                 0 <= nx < cleaned.shape[1] and
#                                 cleaned[ny, nx] > 0):
#                                 # Fill with this tissue
#                                 cleaned[region_mask] = cleaned[ny, nx]
#                                 found = True
#                                 break
#                         if found:
#                             break
#                     if found:
#                         break
    
#     return cleaned


# =============================================================================
# MODEL CREATION
# =============================================================================

def create_three_layer_model(brainweb_slice):
    """
    3-layer simplified model:
    0. Background → will be removed
    1. Scalp (skin, muscle, fat)
    2. Skull (skull, dura, marrow)
    3. Brain (homogeneous - all internal tissues)
    """
    layers = np.zeros_like(brainweb_slice, dtype=np.uint8)
    
    # Layer 1: Scalp
    scalp_mask = (
        Act.indices(brainweb_slice, 'skin') |
        Act.indices(brainweb_slice, 'muscle') |
        Act.indices(brainweb_slice, 'fat') |
        Act.indices(brainweb_slice, 'aroundFat')
    )
    layers[scalp_mask] = 1
    
    # Layer 2: Skull
    skull_mask = (
        Act.indices(brainweb_slice, 'skull') |
        Act.indices(brainweb_slice, 'dura') |
        Act.indices(brainweb_slice, 'marrow')
    )
    layers[skull_mask] = 2
    
    # Layer 3: Brain (homogeneous)
    brain_mask = (
        Act.indices(brainweb_slice, 'csf') |
        Act.indices(brainweb_slice, 'greyMatter') |
        Act.indices(brainweb_slice, 'whiteMatter') |
        Act.indices(brainweb_slice, 'vessels')
    )
    layers[brain_mask] = 3
    
    # Minimal cleanup (only tiny artifacts)
    layers = clean_model_minimal(layers)
    
    return layers


def create_six_layer_model(brainweb_slice):
    """
    6-layer anatomically detailed model:
    0. Background → will be removed
    1. Scalp (skin, muscle, fat)
    2. Skull (skull, dura, marrow)
    3. CSF
    4. Grey Matter
    5. White Matter
    6. Ventricles (deep CSF)
    """
    layers = np.zeros_like(brainweb_slice, dtype=np.uint8)
    
    # Layer 1: Scalp
    scalp_mask = (
        Act.indices(brainweb_slice, 'skin') |
        Act.indices(brainweb_slice, 'muscle') |
        Act.indices(brainweb_slice, 'fat') |
        Act.indices(brainweb_slice, 'aroundFat')
    )
    layers[scalp_mask] = 1
    
    # Layer 2: Skull
    skull_mask = (
        Act.indices(brainweb_slice, 'skull') |
        Act.indices(brainweb_slice, 'dura') |
        Act.indices(brainweb_slice, 'marrow')
    )
    layers[skull_mask] = 2
    
    # Layer 3: CSF
    csf_mask = Act.indices(brainweb_slice, 'csf')
    layers[csf_mask] = 3
    
    # Layer 4: Grey Matter
    grey_mask = Act.indices(brainweb_slice, 'greyMatter')
    layers[grey_mask] = 4
    
    # Layer 5: White Matter
    white_mask = Act.indices(brainweb_slice, 'whiteMatter')
    layers[white_mask] = 5
    
    # Layer 6: Ventricles (deep CSF)
    csf_internal = binary_erosion(csf_mask, iterations=10)
    layers[csf_internal] = 6
    
    # Minimal cleanup (only tiny artifacts)
    layers = clean_model_minimal(layers)
    
    return layers


def create_nine_layer_model(brainweb_slice):
    """
    9-layer highly detailed model:
    0. Background → will be removed
    1. Scalp - Fat/Skin (outermost)
    2. Scalp - Muscle
    3. Skull (skull, dura, marrow)
    4. CSF
    5. Grey Matter - Cortical
    6. White Matter
    7. Grey Matter - Deep nuclei
    8. Ventricles (CSF)
    9. Blood vessels
    """
    layers = np.zeros_like(brainweb_slice, dtype=np.uint8)
    
    # Layer 1: Scalp outer (fat, skin)
    scalp_outer_mask = (
        Act.indices(brainweb_slice, 'skin') |
        Act.indices(brainweb_slice, 'fat') |
        Act.indices(brainweb_slice, 'aroundFat')
    )
    layers[scalp_outer_mask] = 1
    
    # Layer 2: Scalp muscle
    scalp_muscle_mask = Act.indices(brainweb_slice, 'muscle')
    layers[scalp_muscle_mask] = 2
    
    # Layer 3: Skull
    skull_mask = (
        Act.indices(brainweb_slice, 'skull') |
        Act.indices(brainweb_slice, 'dura') |
        Act.indices(brainweb_slice, 'marrow')
    )
    layers[skull_mask] = 3
    
    # Layer 4: CSF (outer)
    csf_mask = Act.indices(brainweb_slice, 'csf')
    csf_outer = csf_mask & ~binary_erosion(csf_mask, iterations=5)
    layers[csf_outer] = 4
    
    # Layer 5: Grey Matter - Cortical (outer grey)
    grey_mask = Act.indices(brainweb_slice, 'greyMatter')
    grey_cortical = grey_mask & ~binary_erosion(grey_mask, iterations=3)
    layers[grey_cortical] = 5
    
    # Layer 6: White Matter
    white_mask = Act.indices(brainweb_slice, 'whiteMatter')
    layers[white_mask] = 6
    
    # Layer 7: Grey Matter - Deep (inner grey)
    grey_deep = grey_mask & binary_erosion(grey_mask, iterations=3)
    layers[grey_deep & (layers == 0)] = 7
    
    # Layer 8: Ventricles (deep CSF)
    csf_internal = binary_erosion(csf_mask, iterations=10)
    layers[csf_internal] = 8
    
    # Layer 9: Blood vessels
    vessels_mask = Act.indices(brainweb_slice, 'vessels')
    layers[vessels_mask] = 9
    
    # Minimal cleanup (only tiny artifacts)
    layers = clean_model_minimal(layers)
    
    return layers


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualization(models_dict, subject_name, output_path):
    """
    Create comprehensive visualization of all models
    
    Args:
        models_dict: Dict with '3layer', '6layer', '9layer' models
        subject_name: Name for title
        output_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Color schemes
    colors_3 = ['white', '#FDB462', '#FFFFB3', '#80B1D3']
    colors_6 = ['white', '#FDB462', '#FFFFB3', '#8DD3C7',
                '#BEBADA', '#FB8072', '#80B1D3']
    colors_9 = ['white', '#FDB462', '#FFA07A', '#FFFFB3', '#8DD3C7',
                '#BEBADA', '#FB8072', '#DDA0DD', '#80B1D3', '#FF6347']
    
    cmap_3 = ListedColormap(colors_3)
    cmap_6 = ListedColormap(colors_6)
    cmap_9 = ListedColormap(colors_9)
    
    norm_3 = BoundaryNorm(np.arange(len(colors_3) + 1) - 0.5, cmap_3.N)
    norm_6 = BoundaryNorm(np.arange(len(colors_6) + 1) - 0.5, cmap_6.N)
    norm_9 = BoundaryNorm(np.arange(len(colors_9) + 1) - 0.5, cmap_9.N)
    
    # Plot 3-layer
    im1 = axes[0].imshow(models_dict['3layer'].T, cmap=cmap_3, norm=norm_3, origin='lower')
    axes[0].set_title('3-Layer Model', fontsize=14, fontweight='bold')
    axes[0].axis('equal')
    axes[0].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[0], ticks=range(4))
    cbar1.set_ticklabels(['BG', 'Scalp', 'Skull', 'Brain'])
    
    # Plot 6-layer
    im2 = axes[1].imshow(models_dict['6layer'].T, cmap=cmap_6, norm=norm_6, origin='lower')
    axes[1].set_title('6-Layer Model', fontsize=14, fontweight='bold')
    axes[1].axis('equal')
    axes[1].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[1], ticks=range(7))
    cbar2.set_ticklabels(['BG', 'Scalp', 'Skull', 'CSF', 'Grey', 'White', 'Vent'])
    
    # Plot 9-layer
    im3 = axes[2].imshow(models_dict['9layer'].T, cmap=cmap_9, norm=norm_9, origin='lower')
    axes[2].set_title('9-Layer Model', fontsize=14, fontweight='bold')
    axes[2].axis('equal')
    axes[2].axis('off')
    cbar3 = plt.colorbar(im3, ax=axes[2], ticks=range(10))
    cbar3.set_ticklabels(['BG', 'Skin', 'Musc', 'Skull', 'CSF', 
                         'GreyC', 'White', 'GreyD', 'Vent', 'Vess'], fontsize=8)
    
    fig.suptitle(f'{subject_name}', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_single_subject(subject_file, subject_idx, output_base_dir='brainweb_subjects'):
    """
    Process a single subject: create all 3 models and save
    
    Args:
        subject_file: Path to subject file
        subject_idx: Subject index (0-19)
        output_base_dir: Base directory for output
    
    Returns:
        dict with results
    """
    subject_name = f"subject_{subject_idx:02d}"
    subject_dir = Path(output_base_dir) / subject_name
    subject_dir.mkdir(parents=True, exist_ok=True)
    
    # Load slice
    slice_2d = load_subject_slice(subject_file, slice_type='middle')
    
    # Create models
    model_3 = create_three_layer_model(slice_2d)
    model_6 = create_six_layer_model(slice_2d)
    model_9 = create_nine_layer_model(slice_2d)
    
    # Check background
    bg_3 = (model_3 == 0).sum()
    bg_6 = (model_6 == 0).sum()
    bg_9 = (model_9 == 0).sum()
    
    # Save models
    np.savez_compressed(
        subject_dir / 'head_models.npz',
        model_3layer=model_3,
        model_6layer=model_6,
        model_9layer=model_9,
        raw_slice=slice_2d,
        subject_file=str(subject_file)
    )
    
    # Create visualization
    models_dict = {
        '3layer': model_3,
        '6layer': model_6,
        '9layer': model_9
    }
    
    viz_path = subject_dir / 'models_visualization.png'
    create_visualization(models_dict, subject_name, viz_path)
    
    # Save statistics
    stats = {
        'subject_idx': subject_idx,
        'subject_name': subject_name,
        'slice_shape': slice_2d.shape,
        '3layer': {
            'unique_labels': np.unique(model_3).tolist(),
            'background_pixels': int(bg_3),
            'tissue_pixels': int((model_3 > 0).sum())
        },
        '6layer': {
            'unique_labels': np.unique(model_6).tolist(),
            'background_pixels': int(bg_6),
            'tissue_pixels': int((model_6 > 0).sum())
        },
        '9layer': {
            'unique_labels': np.unique(model_9).tolist(),
            'background_pixels': int(bg_9),
            'tissue_pixels': int((model_9 > 0).sum())
        }
    }
    
    return stats


def process_all_subjects(output_base_dir='brainweb_subjects', max_subjects=None):
    """
    Process all BrainWeb subjects
    
    Args:
        output_base_dir: Base directory for output
        max_subjects: Limit number of subjects (None = all)
    
    Returns:
        list of statistics dicts
    """
    print("="*70)
    print("Processing All BrainWeb Subjects")
    print("="*70)
    
    # Download subjects
    subject_files = download_all_subjects()
    
    if max_subjects is not None:
        subject_files = subject_files[:max_subjects]
    
    n_subjects = len(subject_files)
    print(f"\nProcessing {n_subjects} subjects...")
    print(f"Output directory: {output_base_dir}\n")
    
    # Create base directory
    Path(output_base_dir).mkdir(exist_ok=True)
    
    # Process each subject
    all_stats = []
    
    for idx, subject_file in enumerate(tqdm(subject_files, desc="Processing subjects")):
        try:
            stats = process_single_subject(subject_file, idx, output_base_dir)
            all_stats.append(stats)
        except Exception as e:
            print(f"\n✗ Failed to process subject {idx}: {e}")
            continue
    
    print(f"\n✓ Successfully processed {len(all_stats)}/{n_subjects} subjects")
    
    # Save summary
    import json
    summary = {
        'n_subjects_processed': len(all_stats),
        'n_subjects_total': n_subjects,
        'output_directory': output_base_dir,
        'subjects': all_stats
    }
    
    summary_path = Path(output_base_dir) / 'processing_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved summary: {summary_path}")
    
    # Print statistics
    print("\n" + "="*70)
    print("Summary Statistics")
    print("="*70)
    
    if len(all_stats) > 0:
        for layer_type in ['3layer', '6layer', '9layer']:
            bg_pixels = [s[layer_type]['background_pixels'] for s in all_stats]
            tissue_pixels = [s[layer_type]['tissue_pixels'] for s in all_stats]
            
            print(f"\n{layer_type.upper()}:")
            print(f"  Background pixels: min={min(bg_pixels)}, max={max(bg_pixels)}, mean={np.mean(bg_pixels):.0f}")
            print(f"  Tissue pixels: min={min(tissue_pixels)}, max={max(tissue_pixels)}, mean={np.mean(tissue_pixels):.0f}")
            
            if max(bg_pixels) == 0:
                print(f"  ✓ All models have NO background pixels!")
            else:
                print(f"  ⚠ Some models still have background pixels")
    
    print("\n" + "="*70)
    print("✓ COMPLETE!")
    print("="*70)
    print(f"\nGenerated {len(all_stats)} subject folders in: {output_base_dir}/")
    print("\nEach subject folder contains:")
    print("  - head_models.npz (3, 6, and 9 layer models)")
    print("  - models_visualization.png")
    print("\nTotal summary saved to: processing_summary.json")
    
    return all_stats


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process all BrainWeb subjects with 3, 6, and 9 layer models"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='brainweb_subjects',
        help='Output directory (default: brainweb_subjects)'
    )
    parser.add_argument(
        '--max-subjects',
        type=int,
        default=None,
        help='Maximum number of subjects to process (default: all ~20)'
    )
    
    args = parser.parse_args()
    
    # Process all subjects
    stats = process_all_subjects(
        output_base_dir=args.output_dir,
        max_subjects=args.max_subjects
    )
    
    return stats


if __name__ == "__main__":
    main()