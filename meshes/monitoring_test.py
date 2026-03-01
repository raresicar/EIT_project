"""
Sanity Check for Monitoring Data Generator

Generates a small batch of samples for one subject and creates
diagnostic visualisations.  Run this BEFORE generating the full dataset.

Usage:
    python sanity_check.py --mesh-dir /path/to/brainweb_meshes --subject subject_00
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from pathlib import Path
import sys
import time

sys.path.insert(0, "/mnt/d/Programming/EIT/forward")

from monitoring_data_generator import (
    load_mesh,
    element_centroids,
    generate_subject_layer,
)
from eit_forward_skfem import load_brainweb_mesh


def sanity_check(
    mesh_dir: str = "/mnt/d/Programming/EIT/brainweb_meshes",
    subject: str = "subject_00",
    layer_type: str = "6layer",
    n_samples: int = 8,
    output_dir: str = "./sanity_check_output",
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SANITY CHECK — Monitoring Data Generator")
    print("=" * 70)

    # Generate a small batch
    print(f"\nGenerating {n_samples} samples for {subject} / {layer_type} ...")

    meta = generate_subject_layer(
        subject=subject,
        layer_type=layer_type,
        mesh_base_dir=mesh_dir,
        output_dir=str(out / "data"),
        n_samples=n_samples,
        compute_jacobian=True,
    )

    # Load a few samples back and visualise
    sample_dir = out / "data" / layer_type / subject

    # Load inverse mesh for plotting
    inv_npz = Path(mesh_dir) / subject / layer_type / "inverse" / "head_mesh.npz"
    mesh_inv, _, materials_inv = load_brainweb_mesh(inv_npz, n_electrodes=16)
    tri_inv = Triangulation(mesh_inv.p[0], mesh_inv.p[1], mesh_inv.t.T)

    # Load forward mesh for plotting
    fwd_npz = Path(mesh_dir) / subject / layer_type / "forward" / "head_mesh.npz"
    mesh_fwd, _, materials_fwd = load_brainweb_mesh(fwd_npz, n_electrodes=16)
    tri_fwd = Triangulation(mesh_fwd.p[0], mesh_fwd.p[1], mesh_fwd.t.T)

    n_show = min(n_samples, 4)
    fig, axes = plt.subplots(n_show, 4, figsize=(20, 5 * n_show))
    if n_show == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_show):
        data = np.load(sample_dir / f"sample_{i:04d}.npz", allow_pickle=True)

        scenario = str(data["scenario"])
        stype = str(data["stroke_type"])
        noise = float(data["noise_level"])
        r1 = float(data["stroke_radius1"])
        r2 = float(data["stroke_radius2"])

        # 1. σ₁ on forward mesh
        ax = axes[i, 0]
        im = ax.tripcolor(tri_fwd, data["sigma1_fwd"], shading="flat", cmap="viridis")
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"σ₁ (fwd)\n{scenario} {stype}\nr₁={r1:.1f}", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)

        # 2. σ₂ on forward mesh
        ax = axes[i, 1]
        im = ax.tripcolor(tri_fwd, data["sigma2_fwd"], shading="flat", cmap="viridis")
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"σ₂ (fwd)\nr₂={r2:.1f}", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)

        # 3. Δσ on inverse mesh (ground truth)
        ax = axes[i, 2]
        ds = data["delta_sigma_inv"]
        vmax = max(abs(ds.min()), abs(ds.max()), 1e-6)
        im = ax.tripcolor(tri_inv, ds, shading="flat", cmap="RdBu_r",
                          vmin=-vmax, vmax=vmax)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"Δσ (inv mesh)\nmax |Δσ|={vmax:.3e}", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)

        # 4. ΔU (noisy) — heatmap
        ax = axes[i, 3]
        dU = data["delta_U"]
        im = ax.imshow(dU, aspect="auto", cmap="RdBu_r")
        ax.set_xlabel("Electrode")
        ax.set_ylabel("Pattern")
        ax.set_title(f"ΔU (noisy)\nnoise={noise*100:.2f}%\nmax|ΔU|={np.abs(dU).max():.3e}",
                      fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        f"Monitoring Samples — {subject} / {layer_type}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()

    viz_path = out / "sanity_check.png"
    plt.savefig(viz_path, dpi=200, bbox_inches="tight")
    print(f"\n✓ Saved visualisation: {viz_path}")
    plt.close()

    # Check Jacobian
    jac_path = sample_dir / "jacobian.npz"
    if jac_path.exists():
        jac = np.load(jac_path)
        J = jac["J"]
        print(f"\nJacobian:")
        print(f"  Shape: {J.shape}")
        print(f"  Range: [{J.min():.3e}, {J.max():.3e}]")
        print(f"  Rank (approx): {np.linalg.matrix_rank(J)}")

    # Print summary statistics
    print(f"\nSample summary:")
    for i, m in enumerate(meta):
        print(f"  [{i}] {m['scenario']:10s} {m['stroke_type']:12s}  "
              f"|ΔU|={m['delta_U_max']:.3e}  |Δσ|={m['delta_sigma_max']:.3e}")

    print("\n✓ Sanity check complete")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-dir", default="/mnt/d/Programming/EIT/brainweb_meshes")
    parser.add_argument("--subject", default="subject_00")
    parser.add_argument("--layer", default="6layer")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--output-dir", default="./sanity_check_output")
    args = parser.parse_args()

    sanity_check(
        mesh_dir=args.mesh_dir,
        subject=args.subject,
        layer_type=args.layer,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
    )