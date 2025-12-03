from eit_forward_skfem import EIT, current_method, load_brainweb_mesh, materials_to_conductivity, plot_conductivity
import numpy
import time

# Load mesh
mesh, electrode_markers, materials = load_brainweb_mesh(
    'brainweb_subjects/subject_00/meshes_6layer/head_mesh.npz',
    n_electrodes=16
)

# Create solver
L = 16
Inj = current_method(L=L, l=L-1, method=2, value=1.0)  # Adjacent injection
z = 0.01  # Contact impedance

solver = EIT(L=L, Inj=Inj, z=z, mesh=mesh, electrode_markers=electrode_markers)

# Get conductivity
sigma = materials_to_conductivity(materials, '6layer')

# Plot to check everything is non-zero where expected
# plot_conductivity(mesh, sigma, title="BrainWeb 6-layer conductivity")

print("→ starting forward solve")
t0 = time.time()
u_all, U_all = solver.forward_solve(sigma)
print(f"→ forward solve done in {time.time() - t0:.2f} s")
print(f"  number of patterns: {len(u_all)}")
print(f"  u_all[0] range: [{u_all[0].min():.4e}, {u_all[0].max():.4e}]")
print(f"  U_all[0] range: [{U_all[0].min():.4e}, {U_all[0].max():.4e}]")

U = numpy.array(U_all)
print("U_all shape:", U.shape)
print("Number of patterns:", U.shape[0])
print("Number of electrodes:", U.shape[1])
print("Voltage range:", U.min(), "to", U.max())
print("Example row:", U[0])
mean_per_pattern = U.mean(axis=1)
print("Mean electrode potential per pattern:", mean_per_pattern)
print("Max |mean|:", numpy.abs(mean_per_pattern).max())

print("→ starting Jacobian")
t0 = time.time()
J = solver.calc_jacobian(sigma, u_all)
print(f"→ Jacobian done in {time.time() - t0:.2f} s")
print(f"  Jacobian shape: {J.shape}")
print(f"  J range: [{J.min():.4e}, {J.max():.4e}]")

# Visualization
print("\nCreating visualization...")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

tri = Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)

# Mesh with electrodes
ax = axes[0, 0]
ax.triplot(tri, 'k-', linewidth=0.3, alpha=0.5)

colors = plt.cm.tab20(numpy.linspace(0, 1, L))
for i, facets in enumerate(electrode_markers):
    for facet_idx in facets:
        nodes = mesh.facets[:, facet_idx]
        coords = mesh.p[:, nodes]
        ax.plot(coords[0], coords[1], '-', color=colors[i], linewidth=4)
    # Label
    mid_facet = facets[len(facets)//2]
    mid_nodes = mesh.facets[:, mid_facet]
    mid = mesh.p[:, mid_nodes].mean(axis=1)
    ax.text(mid[0]*1.15, mid[1]*1.15, str(i+1), ha='center', va='center', fontsize=8)

ax.set_aspect('equal')
ax.set_title(f'Mesh with {L} Electrodes')

# Potential field
ax = axes[0, 1]
im = ax.tripcolor(tri, u_all[0], cmap='RdBu_r', shading='gouraud')
plt.colorbar(im, ax=ax, label='Potential')
ax.set_aspect('equal')
ax.set_title('Potential (Pattern 1: I₁=+1, I₂=-1)')

# Electrode voltages
ax = axes[1, 0]
im = ax.imshow(U, aspect='auto', cmap='RdBu_r')
plt.colorbar(im, ax=ax, label='Voltage')
ax.set_xlabel('Electrode')
ax.set_ylabel('Pattern')
ax.set_title('Electrode Potentials U')

# Jacobian sensitivity
ax = axes[1, 1]
sens = numpy.abs(J).mean(axis=0)
im = ax.tripcolor(tri, sens, cmap='hot', shading='flat')
plt.colorbar(im, ax=ax, label='Mean |J|')
ax.set_aspect('equal')
ax.set_title('Jacobian Sensitivity')

plt.tight_layout()
plt.savefig('eit_cem_skfem_demo.png', dpi=150, bbox_inches='tight')
print("Saved: eit_cem_skfem_demo.png")
plt.close()
    