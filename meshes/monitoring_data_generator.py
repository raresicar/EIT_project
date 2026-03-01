"""
Monitoring Data Generator for EIT Stroke Evolution

Generates paired (baseline, follow-up) EIT measurements for stroke monitoring.
The patient ALREADY has a stroke; we track how it evolves over time.

Scenarios:
    - Growth:     stroke expands  (r₂ > r₁)
    - Shrinkage:  stroke shrinks  (r₂ < r₁)  — recovery / treatment response
    - Evolution:  size stable, conductivity changes (edema development, blood breakdown)

For each sample we produce:
    - Forward solve on FINE mesh at state 1 → U₁
    - Forward solve on FINE mesh at state 2 → U₂
    - ΔU = U₂ − U₁  (+ noise)
    - Ground truth Δσ projected onto COARSE (inverse) mesh
    - Jacobian J on inverse mesh (linearised at baseline σ₁)

Output per sample (NPZ):
    delta_U:            (n_patterns, n_electrodes)  — noisy voltage difference
    delta_U_clean:      (n_patterns, n_electrodes)  — clean voltage difference
    delta_sigma_inv:    (n_elements_inv,)            — ground truth Δσ on inverse mesh
    sigma1_inv:         (n_elements_inv,)            — baseline σ on inverse mesh
    sigma2_inv:         (n_elements_inv,)            — follow-up σ on inverse mesh
    sigma1_fwd:         (n_elements_fwd,)            — baseline σ on forward mesh
    sigma2_fwd:         (n_elements_fwd,)            — follow-up σ on forward mesh
    stroke_center:      (2,)
    stroke_radius1:     scalar
    stroke_radius2:     scalar
    stroke_type:        string
    scenario:           string  ('growth', 'shrinkage', 'evolution')
    noise_level:        scalar  (relative noise %)

Output Jacobian (one per subject+layer, not per sample):
    jacobian.npz:
        J:              (n_patterns * n_electrodes, n_elements_inv)
        sigma_ref:      (n_elements_inv,)  — linearisation point

Directory structure:
    monitoring_data/
        6layer/
            subject_00/
                jacobian.npz
                sample_0000.npz
                sample_0001.npz
                ...
            subject_01/
                ...
        3layer/
            ...
        metadata.json
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import json
import time
import sys

from skfem import MeshTri

# ---------------------------------------------------------------------------
# These imports assume the files live next to this script or on PYTHONPATH.
# Adjust the import style to match your project layout.
# ---------------------------------------------------------------------------
sys.path.insert(0, "/mnt/d/Programming/EIT/forward")
from eit_forward_skfem import (
    EIT,
    current_method,
    load_brainweb_mesh,
    materials_to_conductivity,
)


# ========================================================================== #
#  CONDUCTIVITY SAMPLING  (same params as fi_stroke_generator.py)            #
# ========================================================================== #

CONDUCTIVITY_PARAMS = {
    "scalp":               (1.0000, 0.0333),
    "skull":               (0.0625, 0.0021),
    "csf":                 (6.2500, 0.2083),
    "grey_matter":         (0.3063, 0.0102),
    "white_matter":        (0.1938, 0.0065),
    "ischemic_stroke":     (0.0938, 0.0031),
    "hemorrhagic_stroke":  (2.1875, 0.0729),
}

SCALP_CONDUCTIVITY = 0.36  # S/m at ~100 kHz


def _sample(name: str) -> float:
    mu, std = CONDUCTIVITY_PARAMS[name]
    return max(np.random.normal(mu, std), 1e-4)


# ========================================================================== #
#  CONDUCTIVITY ASSIGNMENT                                                   #
# ========================================================================== #

def assign_conductivity_6layer(materials: np.ndarray) -> np.ndarray:
    """
    Assign conductivity (S/m) for 6-layer model with biological variability.

    Material IDs  (from load_brainweb_subjects.py / fi_stroke_generator.py):
        0  Background
        1  Scalp          5  White matter
        2  Skull           6  Ventricles (≈ CSF)
        3  CSF            10  Ischemic stroke
        4  Grey matter    11  Hemorrhagic stroke
    """
    s = SCALP_CONDUCTIVITY
    lut = {
        0:  0.01,
        1:  _sample("scalp") * s,
        2:  _sample("skull") * s,
        3:  _sample("csf") * s,
        4:  _sample("grey_matter") * s,
        5:  _sample("white_matter") * s,
        6:  _sample("csf") * s,
        10: _sample("ischemic_stroke") * s,
        11: _sample("hemorrhagic_stroke") * s,
    }
    return np.array([lut.get(int(m), 0.1) for m in materials])


def assign_conductivity_3layer(materials: np.ndarray) -> np.ndarray:
    """Same for 3-layer.  Brain = average of grey + white."""
    s = SCALP_CONDUCTIVITY
    brain = (_sample("grey_matter") + _sample("white_matter")) / 2 * s
    lut = {
        0:  0.01,
        1:  _sample("scalp") * s,
        2:  _sample("skull") * s,
        3:  brain,
        10: _sample("ischemic_stroke") * s,
        11: _sample("hemorrhagic_stroke") * s,
    }
    return np.array([lut.get(int(m), 0.1) for m in materials])


CONDUCTIVITY_FN = {
    "3layer": assign_conductivity_3layer,
    "6layer": assign_conductivity_6layer,
}


# ========================================================================== #
#  MESH HELPERS                                                              #
# ========================================================================== #

def load_mesh(npz_path: str) -> Tuple[MeshTri, np.ndarray]:
    """Load mesh + materials from NPZ."""
    data = np.load(npz_path)
    pts = data["points"]
    if pts.shape[1] == 3:
        pts = pts[:, :2]
    mesh = MeshTri(pts.T, data["cells"].T)
    return mesh, data["materials"]


def element_centroids(mesh: MeshTri) -> np.ndarray:
    """Return (n_elements, 2) array of triangle centroids."""
    n = mesh.t.shape[1]
    c = np.empty((n, 2))
    for i in range(n):
        nodes = mesh.t[:, i]
        c[i] = mesh.p[:, nodes].mean(axis=1)
    return c


# ========================================================================== #
#  STROKE PLACEMENT                                                          #
# ========================================================================== #

def _brain_mask_and_centroids(
    mesh: MeshTri, materials: np.ndarray, brain_ids: List[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boolean brain mask, all centroids, brain centroids."""
    mask = np.isin(materials, brain_ids)
    cent = element_centroids(mesh)
    return mask, cent, cent[mask]


def random_stroke_location(
    mesh: MeshTri,
    materials: np.ndarray,
    brain_ids: List[int],
) -> Tuple[np.ndarray, float]:
    """
    Pick a random stroke centre inside the brain and a baseline radius
    between 2 % and 5 % of the brain extent.
    """
    brain_mask, _, brain_cent = _brain_mask_and_centroids(mesh, materials, brain_ids)
    brain_centre = brain_cent.mean(axis=0)
    brain_std = brain_cent.std(axis=0)
    brain_size = np.linalg.norm(brain_cent.max(axis=0) - brain_cent.min(axis=0))

    # Centre: brain centre ± 30 % std
    centre = brain_centre + np.random.randn(2) * brain_std * 0.3

    # Baseline radius: 2–5 % of brain size
    radius = brain_size * np.random.uniform(0.02, 0.05)

    return centre, radius


def paint_stroke(
    materials: np.ndarray,
    centroids: np.ndarray,
    brain_mask: np.ndarray,
    centre: np.ndarray,
    radius: float,
    stroke_type: str,
) -> np.ndarray:
    """
    Set elements within *radius* of *centre* (and inside brain) to stroke ID.

    Returns a COPY of materials with stroke painted in.
    """
    mat = materials.copy()
    dist = np.linalg.norm(centroids - centre, axis=1)
    stroke_mask = (dist <= radius) & brain_mask
    stroke_id = 10 if stroke_type == "ischemic" else 11
    mat[stroke_mask] = stroke_id
    return mat


# ========================================================================== #
#  SCENARIO SAMPLING                                                         #
# ========================================================================== #

def sample_scenario() -> dict:
    """
    Sample a random monitoring scenario.

    Returns dict with keys:
        scenario    : 'growth' | 'shrinkage' | 'evolution'
        r_factor    : r₂ / r₁   (1.0 for evolution)
        kappa_shift : multiplicative conductivity shift for evolution scenario
        stroke_type : 'ischemic' | 'hemorrhagic'
        noise_level : relative noise  (fraction, e.g. 0.003 = 0.3 %)
    """
    scenario = np.random.choice(
        ["growth", "shrinkage", "evolution"],
        p=[0.40, 0.40, 0.20],
    )

    stroke_type = np.random.choice(["ischemic", "hemorrhagic"])

    if scenario == "growth":
        r_factor = np.random.uniform(1.10, 1.80)
        kappa_shift = 1.0
    elif scenario == "shrinkage":
        r_factor = np.random.uniform(0.30, 0.90)
        kappa_shift = 1.0
    else:  # evolution
        r_factor = 1.0
        # conductivity drifts ±30 %
        kappa_shift = np.random.uniform(0.70, 1.30)

    # Noise level: uniform in 0.1 % – 0.5 %
    noise_level = np.random.uniform(0.001, 0.005)

    return dict(
        scenario=scenario,
        r_factor=r_factor,
        kappa_shift=kappa_shift,
        stroke_type=stroke_type,
        noise_level=noise_level,
    )


# ========================================================================== #
#  PROJECTION:  fine mesh  →  coarse mesh                                    #
# ========================================================================== #

def project_conductivity(
    sigma_fine: np.ndarray,
    centroids_fine: np.ndarray,
    centroids_coarse: np.ndarray,
) -> np.ndarray:
    """
    Project per-element conductivity from fine mesh to coarse mesh.

    Strategy: for each coarse element, find the nearest fine-mesh centroid
    and copy its value.  (Both meshes are pixel-based triangulations from the
    same image, so nearest-centroid is a reasonable and cheap approach.)
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(centroids_fine)
    _, idx = tree.query(centroids_coarse)
    return sigma_fine[idx]


# ========================================================================== #
#  ADD NOISE                                                                 #
# ========================================================================== #

def add_noise(delta_U: np.ndarray, noise_level: float) -> np.ndarray:
    """
    Add relative Gaussian noise.

    noise_level is a fraction (e.g. 0.003 = 0.3 %).
    Noise std = noise_level * max(|ΔU|).
    """
    scale = noise_level * np.abs(delta_U).max()
    if scale < 1e-30:
        return delta_U.copy()
    return delta_U + np.random.randn(*delta_U.shape) * scale


# ========================================================================== #
#  SINGLE-SAMPLE GENERATION                                                  #
# ========================================================================== #

def generate_one_sample(
    solver_fwd: EIT,
    mesh_fwd: MeshTri,
    materials_fwd: np.ndarray,
    centroids_fwd: np.ndarray,
    brain_mask_fwd: np.ndarray,
    centroids_inv: np.ndarray,
    brain_ids: List[int],
    layer_type: str,
) -> dict:
    """
    Generate one monitoring sample.

    Returns a dict with all arrays ready to be saved to NPZ.
    """
    cond_fn = CONDUCTIVITY_FN[layer_type]
    scenario = sample_scenario()

    # --- Stroke location (shared for both states) ---
    centre, r1 = random_stroke_location(mesh_fwd, materials_fwd, brain_ids)
    r2 = r1 * scenario["r_factor"]

    mat1 = paint_stroke(
        materials_fwd, centroids_fwd, brain_mask_fwd,
        centre, r1, scenario["stroke_type"],
    )
    
    # --- Shared background conductivity (same tissue for both states) ---
    sigma_background = cond_fn(materials_fwd)

    # --- State 1 (baseline with existing stroke) ---
    mat1 = paint_stroke(
        materials_fwd, centroids_fwd, brain_mask_fwd,
        centre, r1, scenario["stroke_type"],
    )
    sigma1 = sigma_background.copy()
    stroke_mask1 = (mat1 == 10) | (mat1 == 11)
    stroke_cond = _sample(scenario["stroke_type"] + "_stroke") * SCALP_CONDUCTIVITY
    sigma1[stroke_mask1] = stroke_cond

    # --- State 2 (evolved stroke) ---
    mat2 = paint_stroke(
        materials_fwd, centroids_fwd, brain_mask_fwd,
        centre, r2, scenario["stroke_type"],
    )
    sigma2 = sigma_background.copy()
    stroke_mask2 = (mat2 == 10) | (mat2 == 11)

    if scenario["scenario"] == "evolution":
        # Same region, shifted conductivity
        sigma2[stroke_mask2] = stroke_cond * scenario["kappa_shift"]
    else:
        # Different region, same stroke conductivity
        sigma2[stroke_mask2] = stroke_cond

    # --- Forward solves on fine mesh ---
    _, U1 = solver_fwd.forward_solve(sigma1)
    _, U2 = solver_fwd.forward_solve(sigma2)

    U1 = np.array(U1)
    U2 = np.array(U2)
    delta_U_clean = U2 - U1

    # --- Noise ---
    delta_U_noisy = add_noise(delta_U_clean, scenario["noise_level"])

    # --- Project to inverse mesh ---
    sigma1_inv = project_conductivity(sigma1, centroids_fwd, centroids_inv)
    sigma2_inv = project_conductivity(sigma2, centroids_fwd, centroids_inv)
    delta_sigma_inv = sigma2_inv - sigma1_inv

    return dict(
        delta_U=delta_U_noisy,
        delta_U_clean=delta_U_clean,
        delta_sigma_inv=delta_sigma_inv,
        sigma1_inv=sigma1_inv,
        sigma2_inv=sigma2_inv,
        sigma1_fwd=sigma1,
        sigma2_fwd=sigma2,
        stroke_center=centre,
        stroke_radius1=r1,
        stroke_radius2=r2,
        stroke_type=scenario["stroke_type"],
        scenario=scenario["scenario"],
        noise_level=scenario["noise_level"],
        r_factor=scenario["r_factor"],
        kappa_shift=scenario["kappa_shift"],
    )


# ========================================================================== #
#  JACOBIAN ON INVERSE MESH                                                  #
# ========================================================================== #

def compute_and_save_jacobian(
    mesh_inv: MeshTri,
    materials_inv: np.ndarray,
    electrode_markers_inv: list,
    layer_type: str,
    output_path: Path,
    n_electrodes: int = 16,
    injection_method: int = 2,
    z: float = 0.01,
):
    """
    Compute Jacobian on the inverse mesh, linearised around the healthy
    (no-stroke) baseline conductivity.  Saved once per subject+layer.
    """
    L = n_electrodes
    Inj = current_method(L=L, l=L - 1, method=injection_method, value=1.0)

    solver_inv = EIT(
        L=L, Inj=Inj, z=z,
        mesh=mesh_inv,
        electrode_markers=electrode_markers_inv,
    )

    # Baseline conductivity (healthy — no stroke)
    sigma_ref = materials_to_conductivity(materials_inv, layer_type)

    print("    Computing Jacobian on inverse mesh ...")
    t0 = time.time()
    u_all, _ = solver_inv.forward_solve(sigma_ref)
    J = solver_inv.calc_jacobian(sigma_ref, u_all)
    print(f"    ✓ Jacobian {J.shape} in {time.time() - t0:.1f}s")

    np.savez_compressed(
        output_path,
        J=J,
        sigma_ref=sigma_ref,
    )
    print(f"    ✓ Saved {output_path}")

    return J, sigma_ref


# ========================================================================== #
#  BATCH GENERATION FOR ONE SUBJECT + LAYER                                  #
# ========================================================================== #

def generate_subject_layer(
    subject: str,
    layer_type: str,
    mesh_base_dir: str,
    output_dir: str,
    n_samples: int = 250,
    n_electrodes: int = 16,
    injection_method: int = 2,
    z: float = 0.01,
    electrode_coverage: float = 0.5,
    compute_jacobian: bool = True,
):
    """
    Generate all monitoring samples for one subject + layer type.
    """
    mesh_base = Path(mesh_base_dir)
    out = Path(output_dir) / layer_type / subject
    out.mkdir(parents=True, exist_ok=True)

    brain_ids = {"3layer": [3], "6layer": [4, 5]}[layer_type]

    # ---- Load meshes ----
    fwd_npz = mesh_base / subject / layer_type / "forward" / "head_mesh.npz"
    inv_npz = mesh_base / subject / layer_type / "inverse" / "head_mesh.npz"

    print(f"\n  Loading forward mesh: {fwd_npz}")
    mesh_fwd, electrode_markers_fwd, materials_fwd = load_brainweb_mesh(
        fwd_npz, n_electrodes=n_electrodes, electrode_coverage=electrode_coverage,
    )
    print(f"    {mesh_fwd.p.shape[1]} nodes, {mesh_fwd.t.shape[1]} elements")

    print(f"  Loading inverse mesh: {inv_npz}")
    mesh_inv, electrode_markers_inv, materials_inv = load_brainweb_mesh(
        inv_npz, n_electrodes=n_electrodes, electrode_coverage=electrode_coverage,
    )
    print(f"    {mesh_inv.p.shape[1]} nodes, {mesh_inv.t.shape[1]} elements")

    # ---- Precompute centroids & brain mask (forward mesh) ----
    centroids_fwd = element_centroids(mesh_fwd)
    brain_mask_fwd = np.isin(materials_fwd, brain_ids)
    centroids_inv = element_centroids(mesh_inv)

    # ---- Forward solver on fine mesh ----
    L = n_electrodes
    Inj = current_method(L=L, l=L - 1, method=injection_method, value=1.0)

    solver_fwd = EIT(
        L=L, Inj=Inj, z=z,
        mesh=mesh_fwd, electrode_markers=electrode_markers_fwd,
    )

    # ---- Jacobian on inverse mesh (once per subject+layer) ----
    if compute_jacobian:
        jac_path = out / "jacobian.npz"
        if not jac_path.exists():
            compute_and_save_jacobian(
                mesh_inv, materials_inv, electrode_markers_inv,
                layer_type, jac_path,
                n_electrodes=n_electrodes,
                injection_method=injection_method,
                z=z,
            )
        else:
            print(f"    Jacobian already exists: {jac_path}")

    # ---- Generate samples ----
    print(f"\n  Generating {n_samples} monitoring samples ...")
    t_start = time.time()

    metadata_samples = []

    for idx in range(n_samples):
        sample = generate_one_sample(
            solver_fwd=solver_fwd,
            mesh_fwd=mesh_fwd,
            materials_fwd=materials_fwd,
            centroids_fwd=centroids_fwd,
            brain_mask_fwd=brain_mask_fwd,
            centroids_inv=centroids_inv,
            brain_ids=brain_ids,
            layer_type=layer_type,
        )

        # Save
        fname = out / f"sample_{idx:04d}.npz"
        np.savez_compressed(fname, **sample)

        meta = {
            "index": idx,
            "scenario": sample["scenario"],
            "stroke_type": sample["stroke_type"],
            "noise_level": float(sample["noise_level"]),
            "r1": float(sample["stroke_radius1"]),
            "r2": float(sample["stroke_radius2"]),
            "delta_U_max": float(np.abs(sample["delta_U_clean"]).max()),
            "delta_sigma_max": float(np.abs(sample["delta_sigma_inv"]).max()),
        }
        metadata_samples.append(meta)

        if (idx + 1) % 50 == 0 or idx == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            eta = (n_samples - idx - 1) / rate
            print(f"    [{idx+1:4d}/{n_samples}]  "
                  f"{meta['scenario']:10s} {meta['stroke_type']:12s}  "
                  f"|ΔU|_max={meta['delta_U_max']:.3e}  "
                  f"|Δσ|_max={meta['delta_sigma_max']:.3e}  "
                  f"ETA {eta/60:.1f} min")

    elapsed = time.time() - t_start
    print(f"\n  ✓ Generated {n_samples} samples in {elapsed:.1f}s "
          f"({n_samples/elapsed:.1f} samples/s)")

    return metadata_samples


# ========================================================================== #
#  FULL PIPELINE                                                             #
# ========================================================================== #

def generate_all(
    mesh_base_dir: str = "/mnt/d/Programming/EIT/brainweb_meshes",
    output_dir: str = "/mnt/d/Programming/EIT/monitoring_data",
    subjects: Optional[List[str]] = None,
    layer_types: Optional[List[str]] = None,
    n_samples_per_subject: int = 250,
    n_electrodes: int = 16,
    injection_method: int = 2,
    z: float = 0.01,
    seed: int = 42,
):
    """
    Generate the full monitoring dataset.

    With 20 subjects × 250 samples = 5 000 samples per layer type.
    """
    np.random.seed(seed)

    mesh_base = Path(mesh_base_dir)
    out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    if subjects is None:
        subjects = sorted(
            d.name for d in mesh_base.iterdir()
            if d.is_dir() and d.name.startswith("subject_")
        )

    if layer_types is None:
        layer_types = ["6layer"]  # primary experimental setup

    print("=" * 70)
    print("MONITORING DATA GENERATION")
    print("=" * 70)
    print(f"  Subjects:            {len(subjects)}")
    print(f"  Layer types:         {layer_types}")
    print(f"  Samples / subject:   {n_samples_per_subject}")
    print(f"  Total samples:       {len(subjects) * n_samples_per_subject} per layer")
    print(f"  Electrodes:          {n_electrodes}")
    print(f"  Injection method:    {injection_method}")
    print(f"  Contact impedance:   {z}")
    print(f"  Output:              {out_base}")
    print("=" * 70)

    all_metadata = {}

    for layer_type in layer_types:
        print(f"\n{'='*70}")
        print(f"  LAYER TYPE: {layer_type.upper()}")
        print(f"{'='*70}")

        layer_meta = {}

        for subj in subjects:
            subj_dir = mesh_base / subj / layer_type
            if not subj_dir.exists():
                print(f"\n  ⚠ Skipping {subj}/{layer_type} — directory not found")
                continue

            print(f"\n{'─'*70}")
            print(f"  Subject: {subj}  |  Layer: {layer_type}")
            print(f"{'─'*70}")

            try:
                meta = generate_subject_layer(
                    subject=subj,
                    layer_type=layer_type,
                    mesh_base_dir=mesh_base_dir,
                    output_dir=output_dir,
                    n_samples=n_samples_per_subject,
                    n_electrodes=n_electrodes,
                    injection_method=injection_method,
                    z=z,
                )
                layer_meta[subj] = meta
            except Exception as e:
                print(f"  ✗ FAILED: {e}")
                import traceback; traceback.print_exc()
                continue

        all_metadata[layer_type] = layer_meta

    # ---- Save global metadata ----
    meta_path = out_base / "metadata.json"
    # Convert for JSON serialisation
    meta_json = {}
    for lt, subj_dict in all_metadata.items():
        meta_json[lt] = {}
        for subj, samples in subj_dict.items():
            meta_json[lt][subj] = {
                "n_samples": len(samples),
                "scenarios": {
                    s: sum(1 for x in samples if x["scenario"] == s)
                    for s in ["growth", "shrinkage", "evolution"]
                },
                "stroke_types": {
                    s: sum(1 for x in samples if x["stroke_type"] == s)
                    for s in ["ischemic", "hemorrhagic"]
                },
            }

    with open(meta_path, "w") as f:
        json.dump(meta_json, f, indent=2)

    print(f"\n{'='*70}")
    print("✓ COMPLETE")
    print(f"{'='*70}")
    print(f"  Metadata: {meta_path}")


# ========================================================================== #
#  CLI                                                                       #
# ========================================================================== #

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate monitoring data for EIT stroke evolution"
    )
    parser.add_argument(
        "--mesh-dir",
        default="/mnt/d/Programming/EIT/brainweb_meshes",
    )
    parser.add_argument(
        "--output-dir",
        default="/mnt/d/Programming/EIT/monitoring_data",
    )
    parser.add_argument(
        "--subjects", nargs="*", default=None,
        help="Specific subjects (default: all)",
    )
    parser.add_argument(
        "--layers", nargs="*", default=["6layer"],
        choices=["3layer", "6layer"],
    )
    parser.add_argument(
        "--n-samples", type=int, default=250,
        help="Samples per subject (default 250; 20 subjects × 250 = 5000)",
    )
    parser.add_argument(
        "--electrodes", type=int, default=16,
    )
    parser.add_argument(
        "--injection", type=int, default=2,
        choices=[1, 2, 3, 4, 5],
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )

    args = parser.parse_args()

    generate_all(
        mesh_base_dir=args.mesh_dir,
        output_dir=args.output_dir,
        subjects=args.subjects,
        layer_types=args.layers,
        n_samples_per_subject=args.n_samples,
        n_electrodes=args.electrodes,
        injection_method=args.injection,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()