"""
EIT Forward Model Simulator for BrainWeb Subjects
- Parallel processing with multiprocessing
- Complete Electrode Model (CEM)
- Gaussian white noise (1%)
- Saves measurements + mesh data for reconstruction
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
import sys
from datetime import datetime
import multiprocessing as mp
from functools import partial

print("Importing scikit-fem...")
try:
    from skfem import MeshTri, Basis, ElementTriP1, ElementTriP0, BilinearForm, Function
    from skfem.helpers import grad, dot
    from skfem import asm
    print("✓ scikit-fem imported successfully")
except ImportError as e:
    print(f"✗ Error: {e}")
    print("Install with: pip install scikit-fem[all]")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

class EITConfig:
    """EIT simulation configuration"""
    
    # Electrode configuration
    N_ELECTRODES_OPTIONS = [16, 32]  # User can choose
    CONTACT_IMPEDANCE = 1e-4  # Ω·m²
    
    # Current patterns
    CURRENT_AMPLITUDE = 1e-3  # 1 mA
    PATTERN_TYPE = 'adjacent'  # Best for 2D EIT
    
    # Noise
    NOISE_LEVEL = 0.01  # 1% noise
    NOISE_TYPE = 'gaussian'
    
    # Conductivity values at 100 kHz (S/m) - Paldanius et al.
    CONDUCTIVITY_3LAYER = {
        0: 0.01,   # Background
        1: 0.36,   # Scalp
        2: 0.02,   # Skull
        3: 0.15,   # Brain (homogeneous average)
    }
    
    CONDUCTIVITY_6LAYER = {
        0: 0.01,   # Background
        1: 0.36,   # Scalp
        2: 0.02,   # Skull
        3: 2.0,    # CSF
        4: 0.11,   # Grey matter
        5: 0.08,   # White matter
        6: 2.0,    # Ventricles
    }


# =============================================================================
# ELECTRODE PLACEMENT
# =============================================================================

def place_electrodes_uniform(mesh, n_electrodes=32):
    """
    Place electrodes uniformly around the boundary
    
    Returns:
        electrode_nodes: List of node index arrays for each electrode
        electrode_centers: (n_electrodes, 2) electrode center positions
        electrode_info: Dict with detailed electrode info
    """
    # Get boundary nodes
    boundary_facets = mesh.facets[:, mesh.boundary_facets()]
    boundary_nodes = np.unique(boundary_facets.flatten())
    
    # Get boundary coordinates
    boundary_coords = mesh.p[:, boundary_nodes]
    
    # Calculate angles from center
    center = mesh.p.mean(axis=1)
    angles = np.arctan2(boundary_coords[1] - center[1], 
                       boundary_coords[0] - center[0])
    
    # Sort by angle
    sorted_indices = np.argsort(angles)
    boundary_nodes_sorted = boundary_nodes[sorted_indices]
    
    # Divide into electrode segments
    n_boundary = len(boundary_nodes_sorted)
    nodes_per_electrode = n_boundary // n_electrodes
    
    electrode_nodes = []
    electrode_centers = []
    electrode_widths = []
    
    for i in range(n_electrodes):
        start_idx = i * nodes_per_electrode
        end_idx = start_idx + nodes_per_electrode if i < n_electrodes - 1 else n_boundary
        
        elec_nodes = boundary_nodes_sorted[start_idx:end_idx]
        electrode_nodes.append(elec_nodes)
        
        # Electrode center
        elec_coords = mesh.p[:, elec_nodes]
        center_coord = elec_coords.mean(axis=1)
        electrode_centers.append(center_coord)
        
        # Electrode width (arc length)
        width = 0.0
        for j in range(len(elec_nodes) - 1):
            n1, n2 = elec_nodes[j], elec_nodes[j+1]
            width += np.linalg.norm(mesh.p[:, n1] - mesh.p[:, n2])
        electrode_widths.append(width)
    
    electrode_centers = np.array(electrode_centers)
    electrode_widths = np.array(electrode_widths)
    
    electrode_info = {
        'n_electrodes': n_electrodes,
        'nodes_per_electrode': [len(e) for e in electrode_nodes],
        'electrode_widths_mm': electrode_widths,
        'total_boundary_length_mm': sum(electrode_widths)
    }
    
    return electrode_nodes, electrode_centers, electrode_info


# =============================================================================
# CURRENT PATTERNS
# =============================================================================

def create_all_patterns(n_electrodes, pattern_type='adjacent', current_amplitude=1e-3):
    """
    Create all current injection patterns
    
    For adjacent: n_electrodes independent patterns
    
    Returns:
        patterns: (n_patterns, n_electrodes) array
        pattern_info: Dict with pattern metadata
    """
    patterns = []
    
    if pattern_type == 'adjacent':
        # Adjacent: inject at i, remove at i+1
        for i in range(n_electrodes):
            pattern = np.zeros(n_electrodes)
            source = i
            sink = (i + 1) % n_electrodes
            pattern[source] = current_amplitude
            pattern[sink] = -current_amplitude
            patterns.append(pattern)
    
    elif pattern_type == 'opposite':
        # Opposite: inject at i, remove at i+n/2
        for i in range(n_electrodes // 2):
            pattern = np.zeros(n_electrodes)
            source = i
            sink = (i + n_electrodes // 2) % n_electrodes
            pattern[source] = current_amplitude
            pattern[sink] = -current_amplitude
            patterns.append(pattern)
    
    patterns = np.array(patterns)
    
    pattern_info = {
        'pattern_type': pattern_type,
        'n_patterns': len(patterns),
        'current_amplitude_A': current_amplitude,
        'independent_measurements': len(patterns)
    }
    
    return patterns, pattern_info


# =============================================================================
# COMPLETE ELECTRODE MODEL
# =============================================================================

def assemble_cem_system(mesh, materials, conductivity_map, electrode_nodes, 
                       contact_impedance=1e-4):
    """
    Assemble Complete Electrode Model system matrix
    
    Returns:
        K: System matrix (sparse)
        n_dofs: Total DOFs
        n_nodes: Number of mesh nodes
        n_electrodes: Number of electrodes
    """
    from skfem import Functional
    
    # Create FEM basis for potential (P1 - continuous)
    basis = Basis(mesh, ElementTriP1())
    
    # Create basis for conductivity (P0 - piecewise constant per element)
    basis_sigma = Basis(mesh, ElementTriP0())
    
    # Map materials to conductivity values
    sigma_values = np.array([conductivity_map.get(int(m), 0.1) for m in materials])
    
    # Create conductivity function on P0 basis
    sigma_func = Function(basis_sigma, sigma_values)
    
    n_nodes = mesh.p.shape[1]
    n_electrodes = len(electrode_nodes)
    n_dofs = n_nodes + n_electrodes
    
    # Assemble interior stiffness matrix with conductivity function
    @BilinearForm
    def laplacian(u, v, w):
        return w['sigma'] * dot(grad(u), grad(v))
    
    # Interpolate conductivity to quadrature points
    K_interior = asm(laplacian, basis, sigma=sigma_func)
    
    # Extend for electrode DOFs
    K = np.zeros((n_dofs, n_dofs))
    K[:n_nodes, :n_nodes] = K_interior.toarray()
    
    # Add CEM boundary terms
    for elec_idx, elec_nodes in enumerate(electrode_nodes):
        U_idx = n_nodes + elec_idx
        
        # Electrode area (perimeter length)
        electrode_area = 0.0
        for i in range(len(elec_nodes)):
            n1 = elec_nodes[i]
            n2 = elec_nodes[(i + 1) % len(elec_nodes)]
            electrode_area += np.linalg.norm(mesh.p[:, n1] - mesh.p[:, n2])
        
        # CEM coupling coefficient
        cem_coeff = 1.0 / (contact_impedance * electrode_area)
        
        for node in elec_nodes:
            K[node, node] += cem_coeff
            K[node, U_idx] -= cem_coeff
            K[U_idx, node] -= cem_coeff
            K[U_idx, U_idx] += cem_coeff
    
    return csr_matrix(K), n_dofs, n_nodes, n_electrodes


def solve_eit_forward_single(K, n_nodes, n_electrodes, current_pattern):
    """
    Solve single forward problem
    
    Returns:
        u_nodes: Node potentials
        U_electrodes: Electrode potentials
    """
    n_dofs = n_nodes + n_electrodes
    
    # Build RHS
    f = np.zeros(n_dofs)
    f[n_nodes:] = current_pattern
    
    # Ground one electrode
    ground_dof = n_nodes  # Ground electrode 0
    
    # Remove ground DOF
    active_dofs = np.ones(n_dofs, dtype=bool)
    active_dofs[ground_dof] = False
    
    K_reduced = K[active_dofs][:, active_dofs]
    f_reduced = f[active_dofs]
    
    # Solve
    u_reduced = spsolve(K_reduced, f_reduced)
    
    # Reconstruct full solution
    u_full = np.zeros(n_dofs)
    u_full[active_dofs] = u_reduced
    u_full[ground_dof] = 0.0
    
    return u_full[:n_nodes], u_full[n_nodes:]


# =============================================================================
# NOISE MODEL
# =============================================================================

def add_measurement_noise(voltages, noise_level=0.01, noise_type='gaussian'):
    """
    Add noise to voltage measurements
    
    Args:
        voltages: Clean voltage measurements
        noise_level: Noise level (1% = 0.01)
        noise_type: 'gaussian' for white noise
    
    Returns:
        noisy_voltages: Voltages with noise
        noise_std: Standard deviation of noise
    """
    if noise_type == 'gaussian':
        # Compute noise std as percentage of signal magnitude
        signal_magnitude = np.abs(voltages).mean()
        noise_std = noise_level * signal_magnitude
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_std, voltages.shape)
        noisy_voltages = voltages + noise
        
        return noisy_voltages, noise_std
    
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")


# =============================================================================
# FORWARD SIMULATION
# =============================================================================

def simulate_eit_measurements(mesh, materials, layer_type, n_electrodes=32,
                              contact_impedance=1e-4, noise_level=0.01):
    """
    Simulate complete EIT measurement protocol for one mesh
    
    Returns:
        measurements: Dict with all measurement data
    """
    config = EITConfig()
    
    # Get conductivity map
    if layer_type == '3layer':
        conductivity_map = config.CONDUCTIVITY_3LAYER
    elif layer_type == '6layer':
        conductivity_map = config.CONDUCTIVITY_6LAYER
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")
    
    # Place electrodes
    electrode_nodes, electrode_centers, electrode_info = place_electrodes_uniform(
        mesh, n_electrodes
    )
    
    # Create current patterns
    patterns, pattern_info = create_all_patterns(
        n_electrodes, 
        pattern_type=config.PATTERN_TYPE,
        current_amplitude=config.CURRENT_AMPLITUDE
    )
    
    # Assemble system matrix (only once!)
    K, n_dofs, n_nodes, n_elec = assemble_cem_system(
        mesh, materials, conductivity_map, electrode_nodes, contact_impedance
    )
    
    # Solve for all patterns
    n_patterns = patterns.shape[0]
    voltages_clean = np.zeros((n_patterns, n_electrodes))
    
    for i in range(n_patterns):
        _, U_electrodes = solve_eit_forward_single(
            K, n_nodes, n_elec, patterns[i]
        )
        voltages_clean[i] = U_electrodes
    
    # Add noise
    voltages_noisy, noise_std = add_measurement_noise(
        voltages_clean, noise_level, config.NOISE_TYPE
    )
    
    # Package results
    measurements = {
        'voltages_noisy': voltages_noisy,
        'voltages_clean': voltages_clean,
        'current_patterns': patterns,
        'electrode_centers': electrode_centers,
        'electrode_nodes': [e.tolist() for e in electrode_nodes],
        'electrode_info': electrode_info,
        'pattern_info': pattern_info,
        'noise_std': noise_std,
        'noise_level': noise_level,
        'contact_impedance': contact_impedance,
        'conductivity_map': conductivity_map,
        'config': {
            'n_electrodes': n_electrodes,
            'layer_type': layer_type,
            'n_patterns': n_patterns,
            'timestamp': datetime.now().isoformat()
        }
    }
    
    return measurements


# =============================================================================
# PROCESS SINGLE SUBJECT
# =============================================================================

def process_subject_layer(args):
    """
    Process one subject/layer combination (for parallel processing)
    
    Args:
        args: Tuple of (subject_dir, layer_type, n_electrodes, output_dir)
    
    Returns:
        result: Dict with processing results
    """
    subject_dir, layer_type, n_electrodes, output_dir = args
    subject_name = subject_dir.name
    
    try:
        # Load mesh
        mesh_dir = subject_dir / f"meshes_{layer_type}"
        mesh_file = mesh_dir / "head_mesh.npz"
        
        if not mesh_file.exists():
            return {
                'success': False,
                'subject': subject_name,
                'layer_type': layer_type,
                'error': 'Mesh file not found'
            }
        
        data = np.load(mesh_file)
        points = data['points']
        cells = data['cells']
        materials = data['materials']
        
        # Create scikit-fem mesh
        if points.shape[1] == 3:
            points = points[:, :2]
        
        mesh = MeshTri(points.T, cells.T)
        
        # Simulate measurements
        measurements = simulate_eit_measurements(
            mesh, materials, layer_type, 
            n_electrodes=n_electrodes,
            contact_impedance=EITConfig.CONTACT_IMPEDANCE,
            noise_level=EITConfig.NOISE_LEVEL
        )
        
        # Save results
        output_subdir = output_dir / subject_name / f"eit_{layer_type}_{n_electrodes}elec"
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Save measurements
        meas_file = output_subdir / "measurements.npz"
        np.savez_compressed(
            meas_file,
            **measurements
        )
        
        # Save mesh info (for reconstruction)
        mesh_info_file = output_subdir / "mesh_info.npz"
        np.savez_compressed(
            mesh_info_file,
            points=points,
            cells=cells,
            materials=materials
        )
        
        # Create quick visualization
        create_measurement_viz(measurements, mesh, materials, output_subdir)
        
        return {
            'success': True,
            'subject': subject_name,
            'layer_type': layer_type,
            'n_electrodes': n_electrodes,
            'n_measurements': measurements['voltages_noisy'].shape[0],
            'output_dir': str(output_subdir)
        }
        
    except Exception as e:
        return {
            'success': False,
            'subject': subject_name,
            'layer_type': layer_type,
            'error': str(e)
        }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_measurement_viz(measurements, mesh, materials, output_dir):
    """Create quick visualization of measurements"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Mesh with electrodes
    ax = axes[0, 0]
    from matplotlib.tri import Triangulation
    tri = Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
    ax.tripcolor(tri, materials, cmap='tab10', shading='flat', alpha=0.7)
    
    electrode_centers = measurements['electrode_centers']
    ax.plot(electrode_centers[:, 0], electrode_centers[:, 1], 'ro', ms=8, 
           markeredgecolor='white', markeredgewidth=2)
    
    for i, (x, y) in enumerate(electrode_centers):
        ax.text(x, y, str(i), fontsize=6, ha='center', va='center', color='white', 
               fontweight='bold')
    
    ax.set_title('Mesh with Electrodes', fontweight='bold')
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Plot 2: Voltage measurements (heatmap)
    ax = axes[0, 1]
    im = ax.imshow(measurements['voltages_noisy'], aspect='auto', cmap='RdBu_r')
    ax.set_title('Noisy Voltage Measurements', fontweight='bold')
    ax.set_xlabel('Electrode Number')
    ax.set_ylabel('Pattern Number')
    plt.colorbar(im, ax=ax, label='Voltage (V)')
    
    # Plot 3: SNR per pattern
    ax = axes[1, 0]
    signal = np.abs(measurements['voltages_clean']).mean(axis=1)
    noise = np.abs(measurements['voltages_noisy'] - measurements['voltages_clean']).mean(axis=1)
    snr_db = 20 * np.log10(signal / (noise + 1e-12))
    
    ax.plot(snr_db, 'b-', linewidth=2)
    ax.axhline(y=snr_db.mean(), color='r', linestyle='--', 
              label=f'Mean SNR: {snr_db.mean():.1f} dB')
    ax.set_title('SNR per Pattern', fontweight='bold')
    ax.set_xlabel('Pattern Number')
    ax.set_ylabel('SNR (dB)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 4: Voltage distribution
    ax = axes[1, 1]
    ax.hist(measurements['voltages_noisy'].flatten(), bins=50, alpha=0.7, 
           label='Noisy', color='blue')
    ax.hist(measurements['voltages_clean'].flatten(), bins=50, alpha=0.7, 
           label='Clean', color='green')
    ax.set_title('Voltage Distribution', fontweight='bold')
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'measurements_overview.png', dpi=200, bbox_inches='tight')
    plt.close()


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_all_subjects(subjects_dir='brainweb_subjects', 
                        output_dir='eit_measurements',
                        subject_indices=None,
                        layer_types=['3layer', '6layer'],
                        n_electrodes=32,
                        n_workers=None):
    """
    Process multiple subjects in parallel
    
    Args:
        subjects_dir: Directory with subjects
        output_dir: Output directory
        subject_indices: List of subject indices (None = all)
        layer_types: Which layer types to process
        n_electrodes: Number of electrodes
        n_workers: Number of parallel workers (None = CPU count)
    
    Returns:
        results: List of processing results
    """
    print("="*70)
    print("EIT Forward Simulation - Batch Processing")
    print("="*70)
    
    subjects_dir = Path(subjects_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Find subjects
    all_subject_dirs = sorted([d for d in subjects_dir.iterdir()
                               if d.is_dir() and d.name.startswith('subject_')])
    
    if subject_indices is not None:
        subject_dirs = [all_subject_dirs[i] for i in subject_indices 
                       if i < len(all_subject_dirs)]
    else:
        subject_dirs = all_subject_dirs
    
    print(f"\nConfiguration:")
    print(f"  Subjects: {len(subject_dirs)}")
    print(f"  Layer types: {layer_types}")
    print(f"  Electrodes: {n_electrodes}")
    print(f"  Noise level: {EITConfig.NOISE_LEVEL*100}%")
    print(f"  Pattern type: {EITConfig.PATTERN_TYPE}")
    
    # Create task list
    tasks = []
    for subject_dir in subject_dirs:
        for layer_type in layer_types:
            tasks.append((subject_dir, layer_type, n_electrodes, output_dir))
    
    print(f"\nTotal tasks: {len(tasks)}")
    print(f"Parallel workers: {n_workers if n_workers else mp.cpu_count()}")
    
    # Process in parallel
    print("\nProcessing...")
    
    if n_workers == 1:
        # Serial processing (for debugging)
        results = [process_subject_layer(task) for task in tasks]
    else:
        # Parallel processing
        with mp.Pool(processes=n_workers) as pool:
            results = pool.map(process_subject_layer, tasks)
    
    # Summary
    successful = sum(1 for r in results if r['success'])
    print(f"\n✓ Completed: {successful}/{len(results)} successful")
    
    # Save summary
    summary = {
        'n_tasks': len(results),
        'n_successful': successful,
        'config': {
            'n_electrodes': n_electrodes,
            'layer_types': layer_types,
            'noise_level': EITConfig.NOISE_LEVEL,
            'pattern_type': EITConfig.PATTERN_TYPE,
            'contact_impedance': EITConfig.CONTACT_IMPEDANCE
        },
        'results': results
    }
    
    summary_file = output_dir / 'simulation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Saved summary: {summary_file}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="EIT Forward Simulator")
    parser.add_argument('--subjects-dir', type=str, default='brainweb_subjects')
    parser.add_argument('--output-dir', type=str, default='eit_measurements')
    parser.add_argument('--subjects', type=int, nargs='+', default=None,
                       help='Subject indices to process (e.g., 0 1 2)')
    parser.add_argument('--layers', type=str, nargs='+', 
                       default=['3layer', '6layer'],
                       choices=['3layer', '6layer'])
    parser.add_argument('--electrodes', type=int, default=32, choices=[16, 32])
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: all CPUs)')
    
    args = parser.parse_args()
    
    results = process_all_subjects(
        subjects_dir=args.subjects_dir,
        output_dir=args.output_dir,
        subject_indices=args.subjects,
        layer_types=args.layers,
        n_electrodes=args.electrodes,
        n_workers=args.workers
    )
    
    print("\n" + "="*70)
    print("✓ SIMULATION COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()