"""
Stroke Simulation Generator for EIT
Creates synthetic stroke/hemorrhage samples by adding high-conductivity regions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import binary_dilation, distance_transform_edt, binary_closing, binary_fill_holes, generate_binary_structure, label as nd_label
import json
from datetime import datetime


# =============================================================================
# CONFIGURATION
# =============================================================================

class StrokeConfig:
    """Configuration for stroke simulations"""
    
    # Stroke types and their conductivity at 100 kHz
    STROKE_TYPES = {
        'hemorrhagic': {
            'conductivity': 0.70,  # S/m (Paldanius et al.)
            'description': 'Hemorrhagic stroke (bleeding)',
            'color': '#FF0000'  # Red
        },
        'ischemic': {
            'conductivity': 0.05,  # S/m (reduced blood flow)
            'description': 'Ischemic stroke (blocked vessel)',
            'color': '#000080'  # Dark blue
        }
    }
    
    # Location constraints (where strokes can occur)
    VALID_REGIONS = [3, 4, 5, 6]  # CSF, Grey, White, Ventricles
    
    # Size ranges (in pixels)
    SIZE_RANGE = {
        'small': (10, 30),      # 10-30 pixel radius
        'medium': (30, 60),     # 30-60 pixel radius
        'large': (60, 100)      # 60-100 pixel radius
    }


# =============================================================================
# MODEL PREPROCESSING
# =============================================================================

def clean_model(model):
    """
    Clean model by filling small background holes inside brain tissue
    Then fix any tissue contamination (e.g., skull in scalp)
    
    Args:
        model: 2D array of tissue labels
    
    Returns:
        cleaned_model: Model with filled holes and fixed contamination
    """
    cleaned = model.copy()
    
    print("  Step 1: Filling holes in each tissue layer...")
    
    # Fill holes within each tissue type
    tissue_names = {
        1: 'Scalp',
        2: 'Skull', 
        3: 'CSF',
        4: 'Grey matter',
        5: 'White matter',
        6: 'Ventricles'
    }
    
    for tissue_label in [6, 5, 4, 3, 2, 1]:  # Inner to outer
        tissue_mask = (model == tissue_label)
        if tissue_mask.sum() > 0:
            filled_mask = binary_fill_holes(tissue_mask)
            newly_filled = filled_mask & (cleaned == 0)
            
            if newly_filled.sum() > 0:
                print(f"    {tissue_names.get(tissue_label, f'Label {tissue_label}')}: "
                      f"filled {newly_filled.sum()} pixels")
                cleaned[newly_filled] = tissue_label
    
    # Step 2: Fix contamination - remove inner tissues from outer tissues
    print("  Step 2: Fixing tissue contamination...")
    
    # Remove skull (2) from scalp (1)
    # Find connected components of skull
    from scipy.ndimage import label as nd_label
    skull_mask = (cleaned == 2)
    skull_labeled, n_skull_regions = nd_label(skull_mask)
    
    # Find which skull regions touch the original skull
    original_skull = (model == 2)
    valid_skull_regions = set()
    for region_id in range(1, n_skull_regions + 1):
        region_mask = (skull_labeled == region_id)
        if np.any(region_mask & original_skull):
            valid_skull_regions.add(region_id)
    
    # Remove invalid skull regions (those that appeared in scalp during filling)
    for region_id in range(1, n_skull_regions + 1):
        if region_id not in valid_skull_regions:
            region_mask = (skull_labeled == region_id)
            # Revert these to scalp
            cleaned[region_mask] = 1
            print(f"    Removed {region_mask.sum()} skull pixels from scalp")
    
    # Similarly, remove scalp from skull
    scalp_mask = (cleaned == 1)
    scalp_labeled, n_scalp_regions = nd_label(scalp_mask)
    
    original_scalp = (model == 1)
    valid_scalp_regions = set()
    for region_id in range(1, n_scalp_regions + 1):
        region_mask = (scalp_labeled == region_id)
        if np.any(region_mask & original_scalp):
            valid_scalp_regions.add(region_id)
    
    for region_id in range(1, n_scalp_regions + 1):
        if region_id not in valid_scalp_regions:
            region_mask = (scalp_labeled == region_id)
            # Check what's underneath (most likely background or skull)
            # Revert to background
            cleaned[region_mask] = 0
            print(f"    Removed {region_mask.sum()} scalp pixels from interior")
    
    return cleaned


# =============================================================================
# STROKE GENERATION
# =============================================================================

def generate_random_stroke_location(model, valid_regions):
    """
    Generate random stroke location within valid brain regions
    
    Args:
        model: 2D array of tissue labels
        valid_regions: List of valid tissue labels for stroke placement
    
    Returns:
        center: (y, x) coordinates of stroke center
    """
    # Find all valid pixels
    valid_mask = np.isin(model, valid_regions)
    valid_coords = np.argwhere(valid_mask)
    
    if len(valid_coords) == 0:
        raise ValueError("No valid regions found for stroke placement")
    
    # Choose random location
    idx = np.random.randint(len(valid_coords))
    center = tuple(valid_coords[idx])
    
    return center


def create_stroke_region(model_shape, center, radius, shape_type='ellipse'):
    """
    Create stroke region mask
    
    Args:
        model_shape: Shape of the model array
        center: (y, x) center coordinates
        radius: Radius in pixels
        shape_type: 'circle', 'ellipse', or 'irregular'
    
    Returns:
        mask: Boolean mask of stroke region
    """
    y, x = np.ogrid[:model_shape[0], :model_shape[1]]
    cy, cx = center
    
    if shape_type == 'circle':
        # Perfect circle
        mask = ((y - cy)**2 + (x - cx)**2) <= radius**2
        
    elif shape_type == 'ellipse':
        # Ellipse with random aspect ratio
        aspect = np.random.uniform(0.6, 1.4)
        angle = np.random.uniform(0, np.pi)
        
        # Rotate coordinates
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        y_rot = (y - cy) * cos_a - (x - cx) * sin_a
        x_rot = (y - cy) * sin_a + (x - cx) * cos_a
        
        mask = ((y_rot / radius)**2 + (x_rot / (radius * aspect))**2) <= 1
        
    elif shape_type == 'irregular':
        # Irregular blob (more realistic)
        # Start with ellipse
        aspect = np.random.uniform(0.7, 1.3)
        base_mask = ((y - cy)**2 / radius**2 + (x - cx)**2 / (radius * aspect)**2) <= 1
        
        # Add randomness with dilation/erosion
        if np.random.rand() > 0.5:
            base_mask = binary_dilation(base_mask, iterations=np.random.randint(1, 4))
        
        mask = base_mask
    
    return mask


def add_stroke_to_model(base_model, stroke_type='hemorrhagic', 
                        size_category='medium', location=None,
                        shape_type='irregular'):
    """
    Add stroke to a base model
    
    Args:
        base_model: 2D array of tissue labels
        stroke_type: 'hemorrhagic' or 'ischemic'
        size_category: 'small', 'medium', or 'large'
        location: Optional (y, x) tuple for center, otherwise random
        shape_type: 'circle', 'ellipse', or 'irregular'
    
    Returns:
        modified_model: Model with stroke added
        stroke_info: Dictionary with stroke metadata
    """
    # Create copy
    modified_model = base_model.copy()
    
    # Get stroke configuration
    config = StrokeConfig()
    
    # Determine size
    size_range = config.SIZE_RANGE[size_category]
    radius = np.random.randint(size_range[0], size_range[1])
    
    # Determine location
    if location is None:
        location = generate_random_stroke_location(base_model, config.VALID_REGIONS)
    
    # Create stroke region
    stroke_mask = create_stroke_region(base_model.shape, location, radius, shape_type)
    
    # Ensure stroke is within valid regions AND not background
    valid_mask = np.isin(base_model, config.VALID_REGIONS)
    not_background = base_model != 0
    stroke_mask = stroke_mask & valid_mask & not_background
    
    # Label stroke region (use label 7 for stroke)
    STROKE_LABEL = 7
    modified_model[stroke_mask] = STROKE_LABEL
    
    # Store metadata
    stroke_info = {
        'type': stroke_type,
        'conductivity': config.STROKE_TYPES[stroke_type]['conductivity'],
        'size_category': size_category,
        'radius_pixels': int(radius),
        'center_y': int(location[0]),
        'center_x': int(location[1]),
        'area_pixels': int(stroke_mask.sum()),
        'shape_type': shape_type,
        'label': STROKE_LABEL
    }
    
    return modified_model, stroke_info


# =============================================================================
# SAMPLE GENERATION
# =============================================================================

def generate_stroke_samples(base_model, n_samples, output_dir='stroke_samples',
                           model_type='6layer'):
    """
    Generate multiple stroke samples
    
    Args:
        base_model: Clean base model (6-layer or 3-layer)
        n_samples: Number of samples to generate
        output_dir: Directory to save samples
        model_type: '6layer' or '3layer'
    
    Returns:
        samples: List of dictionaries with model and metadata
    """
    print(f"\n{'='*70}")
    print(f"Generating {n_samples} stroke samples")
    print(f"{'='*70}")
    
    # Clean base model first (fill background holes)
    print("\nCleaning base model (filling background holes)...")
    base_model_original = base_model.copy()
    base_model = clean_model(base_model)
    
    n_filled = ((base_model_original == 0) & (base_model != 0)).sum()
    print(f"✓ Filled {n_filled} background pixels")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ Output directory: {output_dir}")
    
    samples = []
    config = StrokeConfig()
    
    # Distribution of stroke types
    stroke_types = list(config.STROKE_TYPES.keys())
    size_categories = list(config.SIZE_RANGE.keys())
    shape_types = ['circle', 'ellipse', 'irregular']
    
    for i in range(n_samples):
        print(f"\nGenerating sample {i+1}/{n_samples}...", end=' ')
        
        # Random parameters
        stroke_type = np.random.choice(stroke_types)
        size_category = np.random.choice(size_categories, 
                                        p=[0.2, 0.5, 0.3])  # More medium-sized
        shape_type = np.random.choice(shape_types,
                                     p=[0.1, 0.3, 0.6])  # Mostly irregular
        
        # Generate stroke
        try:
            modified_model, stroke_info = add_stroke_to_model(
                base_model,
                stroke_type=stroke_type,
                size_category=size_category,
                shape_type=shape_type
            )
            
            # Add sample metadata
            sample_info = {
                'sample_id': i,
                'model_type': model_type,
                'stroke_info': stroke_info,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save
            sample_filename = f'sample_{i:04d}.npz'
            sample_path = os.path.join(output_dir, sample_filename)
            
            np.savez_compressed(
                sample_path,
                model=modified_model,
                base_model=base_model,
                metadata=sample_info
            )
            
            samples.append({
                'model': modified_model,
                'info': sample_info,
                'filename': sample_filename
            })
            
            print(f"✓ {stroke_type} ({size_category}, {shape_type})")
            
        except Exception as e:
            print(f"✗ Failed: {e}")
            continue
    
    print(f"\n{'='*70}")
    print(f"✓ Generated {len(samples)}/{n_samples} samples successfully")
    print(f"{'='*70}")
    
    # Save summary metadata
    summary = {
        'n_samples': len(samples),
        'model_type': model_type,
        'base_model_shape': base_model.shape,
        'generation_timestamp': datetime.now().isoformat(),
        'samples': [s['info'] for s in samples]
    }
    
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved summary: {summary_path}")
    
    return samples


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_samples(samples, output_dir='stroke_samples', n_display=16):
    """
    Create visualization grid of samples
    
    Args:
        samples: List of sample dictionaries
        output_dir: Directory to save visualization
        n_display: Number of samples to display
    """
    print(f"\nCreating visualization...")
    
    n_display = min(n_display, len(samples))
    n_cols = 4
    n_rows = (n_display + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten() if n_display > 1 else [axes]
    
    # Define colormap for 7 layers (including stroke)
    colors = ['white',      # 0: background
              '#FDB462',    # 1: scalp
              '#FFFFB3',    # 2: skull
              '#8DD3C7',    # 3: CSF
              '#BEBADA',    # 4: grey
              '#FB8072',    # 5: white
              '#80B1D3',    # 6: ventricles
              '#FF0000']    # 7: STROKE
    
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(9) - 0.5, cmap.N)
    
    for idx in range(n_display):
        ax = axes[idx]
        sample = samples[idx]
        model = sample['model']
        info = sample['info']['stroke_info']
        
        # Plot
        im = ax.imshow(model.T, cmap=cmap, norm=norm, origin='lower')
        
        # Mark stroke center
        cy, cx = info['center_y'], info['center_x']
        ax.plot(cy, cx, 'w+', markersize=15, markeredgewidth=2)
        
        # Title
        title = (f"Sample {idx}: {info['type']}\n"
                f"{info['size_category']} ({info['radius_pixels']}px)")
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Hide unused axes
    for idx in range(n_display, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    viz_path = os.path.join(output_dir, 'samples_overview.png')
    plt.savefig(viz_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved visualization: {viz_path}")
    plt.close()


def create_comparison_plot(base_model, stroke_model, stroke_info, 
                          output_path='stroke_comparison.png'):
    """
    Create detailed comparison plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Colormap
    colors = ['white', '#FDB462', '#FFFFB3', '#8DD3C7', 
              '#BEBADA', '#FB8072', '#80B1D3', '#FF0000']
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(9) - 0.5, cmap.N)
    
    # Base model
    axes[0].imshow(base_model.T, cmap=cmap, norm=norm, origin='lower')
    axes[0].set_title('Base Model (Healthy)', fontweight='bold', fontsize=14)
    axes[0].axis('off')
    
    # Stroke model
    axes[1].imshow(stroke_model.T, cmap=cmap, norm=norm, origin='lower')
    cy, cx = stroke_info['center_y'], stroke_info['center_x']
    axes[1].plot(cy, cx, 'w+', markersize=20, markeredgewidth=3)
    axes[1].set_title('Model with Stroke', fontweight='bold', fontsize=14)
    axes[1].axis('off')
    
    # Difference (highlight stroke)
    diff = (stroke_model != base_model).astype(float)
    axes[2].imshow(base_model.T, cmap='gray', alpha=0.3, origin='lower')
    axes[2].imshow(diff.T, cmap='Reds', alpha=0.7, origin='lower')
    axes[2].plot(cy, cx, 'b+', markersize=20, markeredgewidth=3)
    axes[2].set_title('Stroke Location', fontweight='bold', fontsize=14)
    axes[2].axis('off')
    
    # Add metadata text
    info_text = (
        f"Type: {stroke_info['type']}\n"
        f"Size: {stroke_info['size_category']} "
        f"({stroke_info['radius_pixels']} px radius)\n"
        f"Area: {stroke_info['area_pixels']} pixels\n"
        f"Conductivity: {stroke_info['conductivity']:.2f} S/m\n"
        f"Center: ({cy}, {cx})"
    )
    fig.text(0.5, 0.02, info_text, ha='center', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison: {output_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Main workflow: Load base model → Generate stroke samples → Visualize
    """
    print("="*70)
    print("Stroke Sample Generator for EIT Simulations")
    print("="*70)
    
    # Load base models
    print("\n1. Loading base models...")
    try:
        data = np.load('brainweb_head_models.npz')
        six_layer = data['six_layer']
        three_layer = data['three_layer']
        print(f"✓ Loaded 6-layer model: {six_layer.shape}")
        print(f"✓ Loaded 3-layer model: {three_layer.shape}")
    except FileNotFoundError:
        print("✗ Error: brainweb_head_models.npz not found!")
        print("  Run the BrainWeb model generation script first.")
        return
    
    # Configuration
    print("\n2. Configuration:")
    print(f"  Stroke types: {list(StrokeConfig.STROKE_TYPES.keys())}")
    print(f"  Size categories: {list(StrokeConfig.SIZE_RANGE.keys())}")
    
    # Get user input
    print("\n3. Sample generation:")
    n_samples = int(input("  How many samples to generate? (e.g., 100): ") or "100")
    model_choice = input("  Use 6-layer or 3-layer model? (6/3): ").strip() or "6"
    
    base_model = six_layer if model_choice == "6" else three_layer
    model_type = "6layer" if model_choice == "6" else "3layer"
    output_dir = f"stroke_samples_{model_type}"
    
    # Generate samples
    print("\n" + "="*70)
    samples = generate_stroke_samples(
        base_model,
        n_samples=n_samples,
        output_dir=output_dir,
        model_type=model_type
    )
    
    # Create visualizations
    print("\n" + "="*70)
    print("Creating visualizations...")
    print("="*70)
    
    # Overview grid
    visualize_samples(samples, output_dir=output_dir, n_display=16)
    
    # Detailed comparison for first sample
    if len(samples) > 0:
        sample = samples[0]
        create_comparison_plot(
            base_model,
            sample['model'],
            sample['info']['stroke_info'],
            output_path=os.path.join(output_dir, 'example_comparison.png')
        )
    
    # Summary
    print("\n" + "="*70)
    print("✓ COMPLETE!")
    print("="*70)
    print(f"\nGenerated files in: {output_dir}/")
    print(f"  - {len(samples)} sample files (sample_XXXX.npz)")
    print(f"  - summary.json (metadata)")
    print(f"  - samples_overview.png (visualization)")
    print(f"  - example_comparison.png (detailed example)")
    
    print("\nTo load a sample:")
    print(f"  >>> data = np.load('{output_dir}/sample_0000.npz', allow_pickle=True)")
    print("  >>> model = data['model']")
    print("  >>> metadata = data['metadata'].item()")
    print("="*70)
    
    return samples


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def generate_samples_cli(n_samples, model_type='6layer', output_dir=None):
    """
    Programmatic interface for generating samples
    
    Args:
        n_samples: Number of samples to generate
        model_type: '6layer' or '3layer'
        output_dir: Optional custom output directory
    """
    # Load base model
    data = np.load('brainweb_head_models.npz')
    base_model = data['six_layer'] if model_type == '6layer' else data['three_layer']
    
    print(f"\nLoaded {model_type} base model: {base_model.shape}")
    print(f"Background pixels before cleaning: {(base_model == 0).sum()}")
    
    # Set output directory
    if output_dir is None:
        output_dir = f"stroke_samples_{model_type}"
    
    # Generate (cleaning happens inside generate_stroke_samples)
    samples = generate_stroke_samples(
        base_model,
        n_samples=n_samples,
        output_dir=output_dir,
        model_type=model_type
    )
    
    # Visualize
    visualize_samples(samples, output_dir=output_dir)
    
    if len(samples) > 0:
        create_comparison_plot(
            samples[0]['model'],  # Use cleaned version
            samples[0]['model'],
            samples[0]['info']['stroke_info'],
            output_path=os.path.join(output_dir, 'example_comparison.png')
        )
    
    return samples


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Check if command line arguments provided
    if len(sys.argv) > 1:
        n_samples = int(sys.argv[1])
        model_type = sys.argv[2] if len(sys.argv) > 2 else '6layer'
        output_dir = sys.argv[3] if len(sys.argv) > 3 else None
        
        print(f"Running in CLI mode: {n_samples} samples, {model_type}")
        generate_samples_cli(n_samples, model_type, output_dir)
    else:
        # Interactive mode
        main()