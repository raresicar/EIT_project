# check_brainweb_boundary.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from eit_forward_skfem import load_brainweb_mesh  # import from your file


def plot_brainweb_boundary(mesh_path, n_electrodes=16, electrode_coverage=0.5):
    # Use your existing helper (this internally does MeshTri(points.T, cells.T))
    mesh, electrode_markers, materials = load_brainweb_mesh(
        mesh_path,
        n_electrodes=n_electrodes,
        electrode_coverage=electrode_coverage,
    )

    print(f"Mesh nodes: {mesh.p.shape[1]}")
    print(f"Mesh elements: {mesh.t.shape[1]}")
    print(f"Boundary facets: {len(mesh.boundary_facets())}")
    for i, facets in enumerate(electrode_markers):
        print(f"Electrode {i}: {len(facets)} facets")

    # --- Base mesh plot ---
    fig, ax = plt.subplots(figsize=(6, 6))
    # Draw all triangles lightly
    ax.triplot(
        mesh.p[0], mesh.p[1], mesh.t.T,
        linewidth=0.2,
    )

    # --- Draw electrodes as colored boundary segments ---
    # One color per electrode
    cmap = plt.get_cmap("tab20")
    for i, facets in enumerate(electrode_markers):
        segments = []
        for facet_idx in facets:
            nodes = mesh.facets[:, facet_idx]
            pts = mesh.p[:, nodes]        # shape (2, 2)
            segments.append(pts.T)        # shape (2, 2) per segment

        if not segments:
            continue

        lc = LineCollection(
            segments,
            linewidths=3.0,
            label=f"Electrode {i}",
            color=cmap(i % cmap.N),
        )
        ax.add_collection(lc)

    ax.set_aspect("equal")
    ax.set_title("BrainWeb Mesh Boundary + Electrode Facets")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # CHANGE THIS to your .npz BrainWeb mesh
    mesh_path = "/mnt/d/Programming/EIT/brainweb_subjects/subject_00/meshes_6layer/head_mesh.npz"

    plot_brainweb_boundary(mesh_path, n_electrodes=16, electrode_coverage=0.5)
