"""
BrainWeb Head Model Generation for EIT Simulations
Clean refactored version - Everything in one place
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # For environments without display
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import binary_erosion

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

def download_and_load_brainweb():
    """Download BrainWeb data and load a subject"""
    print("1. Downloading BrainWeb data...")
    files = brainweb.get_files()
    print(f"   ✓ Downloaded {len(files)} subjects")
    
    print("\n2. Loading subject data...")
    raw_data = brainweb.load_file(files[-1])
    print(f"   ✓ Loaded shape: {raw_data.shape}")
    
    print("\n3. Extracting 2D slice...")
    slice_idx = raw_data.shape[0] // 2  # middle slice
    slice_2d = raw_data[slice_idx, :, :]
    print(f"   ✓ Slice shape: {slice_2d.shape}")
    
    return slice_2d, files[-1]


# =============================================================================
# MODEL CREATION
# =============================================================================

def create_six_layer_model(brainweb_slice):
    """
    Create 6-layer anatomically detailed model:
    0. Background
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
    
    print("\nCreated 6-layer model:")
    layer_names = ['Background', 'Scalp', 'Skull', 'CSF', 
                   'Grey Matter', 'White Matter', 'Ventricles']
    for i, name in enumerate(layer_names):
        count = np.sum(layers == i)
        pct = 100 * count / layers.size
        print(f"  {name}: {count} pixels ({pct:.1f}%)")
    
    return layers


def create_three_layer_model(brainweb_slice):
    """
    Create 3-layer simplified model:
    0. Background
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
    
    # Layer 3: Brain (everything inside - homogeneous)
    brain_mask = (
        Act.indices(brainweb_slice, 'csf') |
        Act.indices(brainweb_slice, 'greyMatter') |
        Act.indices(brainweb_slice, 'whiteMatter') |
        Act.indices(brainweb_slice, 'vessels')
    )
    layers[brain_mask] = 3
    
    print("\nCreated 3-layer model:")
    layer_names = ['Background', 'Scalp', 'Skull', 'Brain']
    for i, name in enumerate(layer_names):
        count = np.sum(layers == i)
        pct = 100 * count / layers.size
        print(f"  {name}: {count} pixels ({pct:.1f}%)")
    
    return layers


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_and_save_models(six_layer, three_layer, raw_slice=None):
    """
    Plot and save both models side by side
    """
    # Define colors matching paper's style
    colors_6 = ['white',      # 0: background
                '#FDB462',    # 1: scalp (orange)
                '#FFFFB3',    # 2: skull (yellow)
                '#8DD3C7',    # 3: CSF (cyan)
                '#BEBADA',    # 4: grey matter (purple)
                '#FB8072',    # 5: white matter (pink/red)
                '#80B1D3']    # 6: ventricles (blue)
    
    colors_3 = ['white',      # 0: background
                '#FDB462',    # 1: scalp (orange)
                '#FFFFB3',    # 2: skull (yellow)
                '#80B1D3']    # 3: brain (blue)
    
    cmap_6 = ListedColormap(colors_6)
    cmap_3 = ListedColormap(colors_3)
    norm_6 = BoundaryNorm(np.arange(8) - 0.5, cmap_6.N)
    norm_3 = BoundaryNorm(np.arange(5) - 0.5, cmap_3.N)
    
    # Create figure
    if raw_slice is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Raw data
        im0 = axes[0].imshow(raw_slice.T, cmap='gist_ncar', origin='lower')
        axes[0].set_title('Raw BrainWeb Data', fontsize=14, fontweight='bold')
        axes[0].axis('equal')
        plt.colorbar(im0, ax=axes[0], label='Tissue Code')
        
        ax_6 = axes[1]
        ax_3 = axes[2]
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax_6 = axes[0]
        ax_3 = axes[1]
    
    # 6-layer model
    im1 = ax_6.imshow(six_layer.T, cmap=cmap_6, norm=norm_6, origin='lower')
    ax_6.set_title('6-Layer Model\n(Anatomically Detailed)', 
                   fontsize=14, fontweight='bold')
    ax_6.axis('equal')
    cbar1 = plt.colorbar(im1, ax=ax_6, ticks=range(7))
    cbar1.set_ticklabels(['BG', 'Scalp', 'Skull', 'CSF', 
                          'Grey', 'White', 'Vent.'])
    
    # 3-layer model
    im2 = ax_3.imshow(three_layer.T, cmap=cmap_3, norm=norm_3, origin='lower')
    ax_3.set_title('3-Layer Model\n(Simplified)', 
                   fontsize=14, fontweight='bold')
    ax_3.axis('equal')
    cbar2 = plt.colorbar(im2, ax=ax_3, ticks=range(4))
    cbar2.set_ticklabels(['BG', 'Scalp', 'Skull', 'Brain'])
    
    plt.tight_layout()
    
    # Save
    filename = 'brainweb_models_comparison.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {filename}")
    plt.close()
    
    return filename


# =============================================================================
# MAIN WORKFLOW
# =============================================================================

def main():
    """
    Main workflow: Download → Create models → Plot and save
    """
    print("="*70)
    print("BrainWeb Head Model Generation for EIT Simulations")
    print("Following: Paldanius et al., IEEE TBME 2022")
    print("="*70)
    
    # Step 1: Download and load data
    slice_2d, source_file = download_and_load_brainweb()
    
    # Step 2: Create 6-layer model
    print("\n" + "="*70)
    print("Creating 6-layer model...")
    print("="*70)
    six_layer = create_six_layer_model(slice_2d)
    
    # Step 3: Create 3-layer model
    print("\n" + "="*70)
    print("Creating 3-layer model...")
    print("="*70)
    three_layer = create_three_layer_model(slice_2d)
    
    # Step 4: Plot and save
    print("\n" + "="*70)
    print("Plotting and saving...")
    print("="*70)
    plot_and_save_models(six_layer, three_layer, raw_slice=slice_2d)
    
    # Step 5: Save data arrays
    print("\nSaving data arrays...")
    np.savez_compressed(
        'brainweb_head_models.npz',
        six_layer=six_layer,
        three_layer=three_layer,
        raw_slice=slice_2d,
        source_file=source_file
    )
    print("✓ Saved: brainweb_head_models.npz")
    
    # Summary
    print("\n" + "="*70)
    print("✓ COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. brainweb_models_comparison.png  - Visualization")
    print("  2. brainweb_head_models.npz        - Data arrays")
    print("\nTo load the data later:")
    print("  >>> data = np.load('brainweb_head_models.npz')")
    print("  >>> six_layer = data['six_layer']")
    print("  >>> three_layer = data['three_layer']")
    print("="*70)
    
    return {
        'six_layer': six_layer,
        'three_layer': three_layer,
        'raw_slice': slice_2d
    }


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    results = main()