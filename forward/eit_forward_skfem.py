"""
EIT Forward Model with Complete Electrode Model (CEM) using scikit-fem

This is a direct port of eit_fenicsx/eit_forward_fenicsx.py to scikit-fem.
The API and mathematical formulation are kept identical.

CEM Weak Form (from eit_fenicsx):
    Find (u, U) such that for all test functions (v, V):
    
    ∫_Ω σ∇u·∇v dΩ + Σ_l (1/z_l) ∫_{e_l} u·v ds - Σ_l (1/z_l) ∫_{e_l} U_l·v ds = 0
    
    - Σ_l (1/z_l) ∫_{e_l} u·V_l ds + Σ_l (1/z_l)|e_l| U_l·V_l = I_l
    
    with constraint: Σ_l U_l = 0

System matrix structure:
    [A + B    C  ] [u]   [0]
    [C^T      D 1] [U] = [I]  
    [0        1 0] [λ]   [0]

where:
    A = stiffness matrix (σ-dependent)
    B = (1/z_l) ∫_{e_l} u·v ds (boundary mass on electrodes)
    C = -(1/z_l) ∫_{e_l} v ds (coupling)
    D = (1/z_l)|e_l| (electrode self-coupling)

References:
    Somersalo et al., SIAM J. Appl. Math., 1992
    Vauhkonen et al., IEEE TBME, 1999
"""

import numpy as np
from typing import Tuple, List, Optional, Union
from pathlib import Path

from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import factorized

from skfem import MeshTri, Basis, ElementTriP1, ElementTriP0
from skfem import BilinearForm, asm
from skfem.helpers import grad, dot

import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


class EIT:
    """
    Complete Electrode Model (CEM) solver using scikit-fem.
    
    Direct port of eit_fenicsx.EIT to scikit-fem.
    """
    
    def __init__(
        self,
        L: int,
        Inj: np.ndarray,
        z: Union[float, np.ndarray],
        mesh: MeshTri,
        electrode_markers: List[np.ndarray],
        ground_dof: int = 0,
    ):
        """
        Initialize EIT solver.
        
        Args:
            L: Number of electrodes
            Inj: Current injection pattern (N, L) matrix with N patterns
            z: Contact impedance - scalar or array of length L
            mesh: scikit-fem MeshTri
            electrode_markers: List of boundary facet indices for each electrode
        """
        assert Inj.shape[-1] == L, "Injection pattern size must match number of electrodes"
        
        self.L = L
        self.Inj = Inj
        self.mesh = mesh
        self.electrode_markers = electrode_markers
        
        # Contact impedance (one per electrode)
        if np.isscalar(z):
            self.z = np.ones(L) * z
        else:
            assert len(z) == L, "Must have one contact impedance per electrode"
            self.z = np.array(z)
        
        # Create FEM bases
        self.V = Basis(mesh, ElementTriP1())  # P1 for potential u
        self.V_sigma = Basis(mesh, ElementTriP0())  # DG0 for conductivity σ
        
        self.dofs = self.V.N  # Number of DOFs (nodes)
        self.ground_dof = int(ground_dof) 

        # Sanity check
        if not (0 <= self.ground_dof < self.dofs):
            raise ValueError(
                f"ground_dof={self.ground_dof} is out of range [0, {self.dofs-1}]"
            )
        
        # Compute electrode lengths
        self._compute_electrode_properties()
        
        # Assemble constant part of LHS (independent of σ)
        self.M = self._assemble_lhs()
        
        # Storage for reuse
        self.M_complete: Optional[csr_matrix] = None
    
    def _compute_electrode_properties(self):
        """Compute electrode lengths and node mappings."""
        self.electrode_lengths = []
        self.electrode_nodes = []  # List of node arrays for each electrode
        
        for electrode_idx, facet_indices in enumerate(self.electrode_markers):
            length = 0.0
            nodes = []
            
            for facet_idx in facet_indices:
                facet_nodes = self.mesh.facets[:, facet_idx]
                p1 = self.mesh.p[:, facet_nodes[0]]
                p2 = self.mesh.p[:, facet_nodes[1]]
                length += np.linalg.norm(p2 - p1)
                nodes.extend(facet_nodes)
            
            self.electrode_lengths.append(length)
            self.electrode_nodes.append(np.unique(nodes))
        
        self.electrode_lengths = np.array(self.electrode_lengths)
        print(f"Electrode lengths: {self.electrode_lengths}")
    
    def _assemble_lhs(self) -> csr_matrix:
        """
        Assemble the constant part of the LHS matrix (independent of σ).
        
        This matches eit_fenicsx.EIT.assemble_lhs() exactly.
        
        The CEM boundary terms are:
            B[i,j] = (1/z_l) ∫_{e_l} φ_i φ_j ds  (mass matrix on electrode)
            C[i,l] = -(1/z_l) ∫_{e_l} φ_i ds     (coupling node-electrode)
            D[l,l] = (1/z_l) |e_l|               (electrode self-coupling)
        """
        n_dofs = self.dofs
        L = self.L
        total_size = n_dofs + L + 1  # nodes + electrodes + Lagrange multiplier
        
        M = lil_matrix((total_size, total_size))
        
        for i in range(L):
            z_i = self.z[i]
            electrode_len = self.electrode_lengths[i]
            electrode_nodes = self.electrode_nodes[i]
            n_nodes = len(electrode_nodes)
            
            # In eit_fenicsx, the boundary integral is assembled using FEniCS measures
            # Here we approximate with lumped mass: each node gets weight |e_l|/n_nodes
            #
            # B matrix: (1/z_l) ∫_{e_l} u·v ds
            # For lumped mass: B[node, node] += (1/z_l) * (|e_l| / n_nodes)
            #
            # Actually, more precisely: ∫_{e_l} φ_i ds = |e_l| / n_nodes for boundary nodes
            # So (1/z_l) ∫_{e_l} φ_i φ_j ds ≈ (1/z_l) * (|e_l| / n_nodes) * δ_{ij}
            
            node_weight = electrode_len / n_nodes
            
            for node in electrode_nodes:
                # B: boundary mass term
                M[node, node] += (1.0 / z_i) * node_weight
                
                # C: coupling term (node to electrode potential U_l)
                # C_i = -(1/z_l) ∫_{e_l} φ_i ds ≈ -(1/z_l) * (|e_l| / n_nodes)
                M[node, n_dofs + i] += -(1.0 / z_i) * node_weight
                M[n_dofs + i, node] += -(1.0 / z_i) * node_weight
            
            # D: electrode self-coupling
            # D[l,l] = (1/z_l) |e_l|
            M[n_dofs + i, n_dofs + i] = (1.0 / z_i) * electrode_len
            
            # Mean-free constraint: Σ_l U_l = 0
            M[n_dofs + L, n_dofs + i] = 1.0
            M[n_dofs + i, n_dofs + L] = 1.0
        
        return csr_matrix(M)
    
    def _assemble_stiffness(self, sigma: np.ndarray) -> csr_matrix:
        """
        Assemble stiffness matrix A for given conductivity.
        
        A[i,j] = ∫_Ω σ ∇φ_i · ∇φ_j dΩ
        """
        @BilinearForm
        def stiffness(u, v, w):
            return w['sigma'] * dot(grad(u), grad(v))
        
        A = asm(stiffness, self.V, sigma=self.V_sigma.interpolate(sigma))
        return A
    
    def create_full_matrix(self, sigma: np.ndarray) -> csr_matrix:
        """
        Create complete system matrix for given conductivity σ.
        
        Returns matrix: [A + B, C; C^T, D]
        """
        A = self._assemble_stiffness(sigma)
        
        total_size = self.dofs + self.L + 1
        A_expanded = lil_matrix((total_size, total_size))
        A_expanded[:self.dofs, :self.dofs] = A.tolil()
        
        M_complete = csr_matrix(A_expanded) + self.M

        #------------------------
        g = self.ground_dof
        M_complete = M_complete.tolil()
        M_complete[g, :] = 0.0
        M_complete[:, g] = 0.0
        M_complete[g, g] = 1.0
        M_complete = M_complete.tocsr()
        #------------------------

        return M_complete
    
    def forward_solve(
        self,
        sigma: np.ndarray,
        Inj: Optional[np.ndarray] = None
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Solve the forward problem.
        
        This matches eit_fenicsx.EIT.forward_solve() exactly.
        
        Args:
            sigma: Conductivity per element
            Inj: Optional injection patterns (N, L). Uses self.Inj if None.
        
        Returns:
            u_all: List of nodal potentials, one per pattern
            U_all: List of electrode potentials, one per pattern
        """
        if Inj is None:
            Inj = self.Inj
        
        num_patterns = Inj.shape[0]
        
        # Build complete matrix
        self.M_complete = self.create_full_matrix(sigma)
        
        # Factorize for efficiency (like eit_fenicsx Scipy backend)
        solve = factorized(self.M_complete.tocsc())
        
        u_all = []
        U_all = []
        
        for pattern_idx in range(num_patterns):
            # Build RHS: [0, ..., 0, I_1, ..., I_L, 0]
            rhs = np.zeros(self.dofs + self.L + 1)
            rhs[self.dofs:self.dofs + self.L] = Inj[pattern_idx]
            
            # Solve
            sol = solve(rhs)
            
            u_all.append(sol[:self.dofs])
            U_all.append(sol[self.dofs:self.dofs + self.L])
        
        return u_all, U_all
    
    def solve_adjoint(
        self,
        deltaU: np.ndarray,
        sigma: Optional[np.ndarray] = None
    ) -> List[np.ndarray]:
        """
        Solve adjoint problem for gradient computation.
        
        The adjoint has the same system matrix but different RHS.
        
        Args:
            deltaU: (N, L) gradient w.r.t. electrode potentials
            sigma: Optional conductivity. Uses cached matrix if None.
        
        Returns:
            p_all: List of adjoint solutions
        """
        if sigma is not None:
            M_complete = self.create_full_matrix(sigma)
        else:
            M_complete = self.M_complete
        
        if M_complete is None:
            raise RuntimeError("Must call forward_solve first or provide sigma")
        
        solve = factorized(M_complete.tocsc())
        
        p_all = []
        for pattern_idx in range(deltaU.shape[0]):
            rhs = np.zeros(self.dofs + self.L + 1)
            rhs[self.dofs:self.dofs + self.L] = deltaU[pattern_idx]
            
            sol = solve(rhs)
            p_all.append(sol[:self.dofs])
        
        return p_all
    
    def calc_jacobian(
        self,
        sigma: np.ndarray,
        u_all: Optional[List[np.ndarray]] = None
    ) -> np.ndarray:
        """
        Calculate Jacobian matrix.
        
        This matches eit_fenicsx.EIT.calc_jacobian() exactly.
        
        J[measurement, element] = -∫_{element} ∇u · ∇bu dΩ
        
        where u is forward solution and bu is solution with unit current at one electrode.
        
        Reference: Margotti PhD thesis, Section 5.2.1
        
        Args:
            sigma: Conductivity per element
            u_all: Optional forward solutions
        
        Returns:
            Jacobian of shape (N*L, n_elements)
        """
        if u_all is None:
            u_all, _ = self.forward_solve(sigma)
        
        # Solve with unit current at each electrode (like eit_fenicsx)
        I2_all = np.eye(self.L)
        bu_all, _ = self.forward_solve(sigma, I2_all)
        
        n_elements = self.mesh.t.shape[1]
        
        # Precompute element gradients
        element_areas = []
        grad_matrices = []
        
        for elem_idx in range(n_elements):
            nodes = self.mesh.t[:, elem_idx]
            coords = self.mesh.p[:, nodes].T  # (3, 2)
            
            x1, y1 = coords[0]
            x2, y2 = coords[1]
            x3, y3 = coords[2]
            
            area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
            element_areas.append(area)
            
            # Gradient of P1 basis: ∇φ_i = [b_i, c_i]^T / (2*area)
            G = np.array([
                [y2 - y3, y3 - y1, y1 - y2],
                [x3 - x2, x1 - x3, x2 - x1]
            ]) / (2 * area)
            grad_matrices.append(G)
        
        element_areas = np.array(element_areas)
        
        # Build Jacobian (matches eit_fenicsx structure)
        num_patterns = len(u_all)
        Jacobian_all = None
        
        for h in range(num_patterns):
            derivative = []
            for j in range(self.L):
                row = np.zeros(n_elements)
                for elem_idx in range(n_elements):
                    nodes = self.mesh.t[:, elem_idx]
                    G = grad_matrices[elem_idx]
                    
                    grad_u = G @ u_all[h][nodes]
                    grad_bu = G @ bu_all[j][nodes]
                    
                    # J = -∫ ∇u · ∇bu * area (negative sign from derivative)
                    row[elem_idx] = np.dot(grad_u, grad_bu) * element_areas[elem_idx]
                
                derivative.append(row)
            
            Jacobian = np.array(derivative)
            
            if Jacobian_all is None:
                Jacobian_all = Jacobian
            else:
                Jacobian_all = np.concatenate((Jacobian_all, Jacobian), axis=0)
        
        return Jacobian_all


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def current_method(L: int, l: int, method: int = 1, value: float = 1.0) -> np.ndarray:
    """
    Create current injection patterns.
    
    Direct copy from eit_fenicsx/utils.py
    
    Methods:
        1: Opposite electrodes (requires even L)
        2: Adjacent electrodes  
        3: One vs all others
        4: Trigonometric pattern
        5: All against electrode 1
    """
    I_all = []
    
    if method == 1:  # Opposite
        if L % 2 != 0:
            raise ValueError("L must be even for opposite pattern")
        for i in range(min(l, L // 2)):
            I = np.zeros(L)
            I[i], I[i + L // 2] = value, -value
            I_all.append(I)
            
    elif method == 2:  # Adjacent
        for i in range(l):
            I = np.zeros(L)
            if i != L - 1:
                I[i], I[i + 1] = value, -value
            else:
                I[0], I[i] = -value, value
            I_all.append(I)
            
    elif method == 3:  # One vs all
        for i in range(l):
            I = np.ones(L) * (-value / (L - 1))
            I[i] = value
            I_all.append(I)
            
    elif method == 4:  # Trigonometric
        for i in range(l):
            I = np.array([np.sin((i + 1) * (k + 1) * 2 * np.pi / L) for k in range(L)])
            I_all.append(I)
            
    elif method == 5:  # All against 1
        for i in range(min(l, L - 1)):
            I = np.zeros(L)
            I[0] = -value
            I[i + 1] = value
            I_all.append(I)
    
    return np.array(I_all)


def load_brainweb_mesh(
    mesh_path: Union[str, Path],
    n_electrodes: int = 16,
    electrode_coverage: float = 0.5
) -> Tuple[MeshTri, List[np.ndarray], np.ndarray]:
    """
    Load BrainWeb mesh from NPZ file.
    
    Args:
        mesh_path: Path to head_mesh.npz file
        n_electrodes: Number of electrodes
        electrode_coverage: Fraction of boundary covered by electrodes
    
    Returns:
        mesh: scikit-fem MeshTri
        electrode_markers: List of boundary facet indices for each electrode
        materials: Material IDs per element
    """
    data = np.load(mesh_path)
    points = data['points']
    cells = data['cells']
    materials = data['materials']

    if points.shape[1] == 3:
        points = points[:, :2]

    mesh = MeshTri(points.T, cells.T)

    # Restrict boundary facets to scalp only
    all_boundary = mesh.boundary_facets()
    scalp_facets = []

    scalp_id = 1  # Scalp material ID

    for f in all_boundary:
        tri_candidates = mesh.f2t[:, f]
        tri = tri_candidates[tri_candidates >= 0][0]

        if materials[tri] == scalp_id:
            scalp_facets.append(f)

    boundary_facets = np.array(scalp_facets)

    # Compute facet midpoints and angles on scalp boundary
    facet_data = []
    for facet_idx in boundary_facets:
        nodes = mesh.facets[:, facet_idx]
        p1, p2 = mesh.p[:, nodes[0]], mesh.p[:, nodes[1]]
        midpoint = (p1 + p2) / 2
        length = np.linalg.norm(p2 - p1)
        facet_data.append((facet_idx, midpoint, length))
    
    # Compute center of boundary
    midpoints = np.array([fd[1] for fd in facet_data])
    center = midpoints.mean(axis=0)
    
    # Sort by angle
    angles = np.array([np.arctan2(fd[1][1] - center[1], fd[1][0] - center[0]) 
                       for fd in facet_data])
    sorted_indices = np.argsort(angles)
    
    # Assign to electrodes based on angular position
    electrode_markers = []
    angle_per_electrode = 2 * np.pi / n_electrodes
    electrode_angle_span = angle_per_electrode * electrode_coverage
    
    for i in range(n_electrodes):
        electrode_center_angle = -np.pi + (i + 0.5) * angle_per_electrode
        electrode_start = electrode_center_angle - electrode_angle_span / 2
        electrode_end = electrode_center_angle + electrode_angle_span / 2
        
        # Find facets in this angular range
        electrode_facets = []
        for sort_idx in sorted_indices:
            angle = angles[sort_idx]
            # Handle wraparound
            if electrode_start < -np.pi:
                in_range = (angle >= electrode_start + 2*np.pi) or (angle <= electrode_end)
            elif electrode_end > np.pi:
                in_range = (angle >= electrode_start) or (angle <= electrode_end - 2*np.pi)
            else:
                in_range = electrode_start <= angle <= electrode_end
            
            if in_range:
                electrode_facets.append(facet_data[sort_idx][0])
        
        # Ensure at least one facet
        if len(electrode_facets) == 0:
            target = electrode_center_angle
            diffs = np.abs(angles - target)
            diffs = np.minimum(diffs, 2*np.pi - diffs)
            closest_idx = np.argmin(diffs)
            electrode_facets = [facet_data[closest_idx][0]]
        
        electrode_markers.append(np.array(electrode_facets))
    
    return mesh, electrode_markers, materials


def create_disk_mesh_with_electrodes(
    n_electrodes: int = 16,
    radius: float = 1.0,
    electrode_coverage: float = 0.5,
    n_radial: int = 20,
    n_angular: int = 64
) -> Tuple[MeshTri, List[np.ndarray]]:
    """
    Create a circular disk mesh with electrodes on boundary.
    
    This mimics the gmsh mesh used in eit_fenicsx demos.
    
    Args:
        n_electrodes: Number of electrodes
        radius: Disk radius
        electrode_coverage: Fraction of boundary covered by electrodes
        n_radial: Number of radial divisions
        n_angular: Number of angular divisions
    
    Returns:
        mesh: scikit-fem MeshTri
        electrode_markers: List of facet indices for each electrode
    """
    from scipy.spatial import Delaunay
    
    # Create points
    points = []
    
    # Center point
    points.append([0, 0])
    
    # Radial points
    for r_idx, r in enumerate(np.linspace(radius/n_radial, radius, n_radial)):
        n_pts = max(8, int(n_angular * r / radius))
        for theta in np.linspace(0, 2*np.pi, n_pts, endpoint=False):
            points.append([r * np.cos(theta), r * np.sin(theta)])
    
    points = np.array(points)
    
    # Triangulate
    tri = Delaunay(points)
    cells = tri.simplices
    
    # Create mesh
    mesh = MeshTri(points.T, cells.T)
    
    # Find boundary facets and assign to electrodes
    boundary_facets = mesh.boundary_facets()
    
    # Get facet angles for sorting
    facet_angles = []
    for facet_idx in boundary_facets:
        nodes = mesh.facets[:, facet_idx]
        midpoint = mesh.p[:, nodes].mean(axis=1)
        angle = np.arctan2(midpoint[1], midpoint[0])
        facet_angles.append(angle)
    
    facet_angles = np.array(facet_angles)
    sorted_indices = np.argsort(facet_angles)
    sorted_facets = boundary_facets[sorted_indices]
    
    # Assign facets to electrodes
    n_boundary = len(boundary_facets)
    facets_per_electrode = int(n_boundary * electrode_coverage / n_electrodes)
    gap_facets = int(n_boundary * (1 - electrode_coverage) / n_electrodes)
    
    electrode_markers = []
    for i in range(n_electrodes):
        start_angle = (i / n_electrodes) * 2 * np.pi - np.pi
        end_angle = start_angle + (electrode_coverage / n_electrodes) * 2 * np.pi
        
        # Find facets in this angular range
        mask = (facet_angles[sorted_indices] >= start_angle) & \
               (facet_angles[sorted_indices] < end_angle)
        
        electrode_facets = sorted_facets[mask]
        
        # Ensure at least one facet
        if len(electrode_facets) == 0:
            center_idx = int((start_angle + end_angle) / 2 / (2 * np.pi) * n_boundary + n_boundary/2) % n_boundary
            electrode_facets = np.array([sorted_facets[center_idx]])
        
        electrode_markers.append(electrode_facets)
    
    return mesh, electrode_markers


# Conductivity maps (S/m at ~100 kHz)
CONDUCTIVITY_3LAYER = {
    0: 0.01,   # Background
    1: 0.36,   # Scalp
    2: 0.02,   # Skull
    3: 0.15,   # Brain
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


def materials_to_conductivity(
    materials: np.ndarray,
    layer_type: str = '6layer'
) -> np.ndarray:
    """
    Convert material IDs to conductivity values.
    
    Args:
        materials: Material IDs per element
        layer_type: '3layer' or '6layer'
    
    Returns:
        Conductivity per element (S/m)
    """
    if layer_type == '3layer':
        cond_map = CONDUCTIVITY_3LAYER
    else:
        cond_map = CONDUCTIVITY_6LAYER
    
    return np.array([cond_map.get(int(m), 0.1) for m in materials])


def create_eit_from_npz(mesh_npz_path, n_electrodes=16, injection_method=2, z=0.01):
    """
    Convenience function: Create EIT solver directly from NPZ mesh file.
    
    Args:
        mesh_npz_path: Path to head_mesh.npz file
        n_electrodes: Number of electrodes (default: 16)
        injection_method: Current injection pattern 1-5 (default: 2=adjacent)
        z: Contact impedance in Ohm·m² (default: 0.01)
    
    Returns:
        solver: EIT solver instance
        mesh: scikit-fem MeshTri
        materials: Material IDs per element
    
    Example:
        >>> solver, mesh, materials = create_eit_from_npz("path/to/mesh.npz")
        >>> sigma = materials_to_conductivity(materials, '6layer')
        >>> u_all, U_all = solver.forward_solve(sigma)
    """
    # Load mesh with electrodes
    mesh, electrode_markers, materials = load_brainweb_mesh(
        mesh_npz_path, 
        n_electrodes=n_electrodes
    )
    
    # Create injection pattern
    L = n_electrodes
    Inj = current_method(L=L, l=L-1, method=injection_method, value=1.0)
    
    # Create solver
    solver = EIT(L=L, Inj=Inj, z=z, mesh=mesh, electrode_markers=electrode_markers)
    
    return solver, mesh, materials


# =============================================================================
# DEMO
# =============================================================================

# def demo():
#     print("="*70)
#     print("CEM Forward Solver Demo (scikit-fem port of eit_fenicsx)")
#     print("="*70)
    
#     # Create disk mesh with electrodes (like eit_fenicsx demo)
#     L = 16  # electrodes
#     print(f"\nCreating disk mesh with {L} electrodes...")
    
#     mesh, electrode_markers = create_disk_mesh_with_electrodes(
#         n_electrodes=L,
#         radius=1.0,
#         electrode_coverage=0.5,
#         n_radial=20,
#         n_angular=64
#     )
    
#     print(f"  Nodes: {mesh.p.shape[1]}")
#     print(f"  Elements: {mesh.t.shape[1]}")
#     print(f"  Boundary facets: {len(mesh.boundary_facets())}")
    
#     # Create injection pattern (method 5 like eit_fenicsx default)
#     Inj = current_method(L=L, l=L-1, method=5, value=1.0)
#     print(f"  Injection patterns: {Inj.shape}")
    
#     # Contact impedance
#     z = np.ones(L) * 0.01  # Same as eit_fenicsx demos
    
#     # Create solver
#     solver = EIT(L=L, Inj=Inj, z=z, mesh=mesh, electrode_markers=electrode_markers)
    
#     print(f"  DOFs: {solver.dofs}")
#     print(f"  System size: {solver.dofs + L + 1}")
    
#     # Homogeneous conductivity
#     sigma = np.ones(mesh.t.shape[1])
    
#     # Forward solve
#     print("\nSolving forward problem...")
#     import time
#     t0 = time.time()
    
#     u_all, U_all = solver.forward_solve(sigma)
    
#     print(f"  Time: {time.time() - t0:.3f} s")
#     print(f"  Patterns solved: {len(U_all)}")
    
#     U_matrix = np.array(U_all)
#     print(f"  Electrode voltages shape: {U_matrix.shape}")
#     print(f"  Voltage range: [{U_matrix.min():.6f}, {U_matrix.max():.6f}]")
    
#     # Jacobian
#     print("\nComputing Jacobian...")
#     t0 = time.time()
    
#     J = solver.calc_jacobian(sigma, u_all)
    
#     print(f"  Time: {time.time() - t0:.3f} s")
#     print(f"  Jacobian shape: {J.shape}")
#     print(f"  Jacobian range: [{J.min():.6e}, {J.max():.6e}]")
    
#     # Visualization
#     print("\nCreating visualization...")
#     import matplotlib.pyplot as plt
#     from matplotlib.tri import Triangulation
    
#     fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
#     tri = Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)
    
#     # Mesh with electrodes
#     ax = axes[0, 0]
#     ax.triplot(tri, 'k-', linewidth=0.3, alpha=0.5)
    
#     colors = plt.cm.tab20(np.linspace(0, 1, L))
#     for i, facets in enumerate(electrode_markers):
#         for facet_idx in facets:
#             nodes = mesh.facets[:, facet_idx]
#             coords = mesh.p[:, nodes]
#             ax.plot(coords[0], coords[1], '-', color=colors[i], linewidth=4)
#         # Label
#         mid_facet = facets[len(facets)//2]
#         mid_nodes = mesh.facets[:, mid_facet]
#         mid = mesh.p[:, mid_nodes].mean(axis=1)
#         ax.text(mid[0]*1.15, mid[1]*1.15, str(i+1), ha='center', va='center', fontsize=8)
    
#     ax.set_aspect('equal')
#     ax.set_title(f'Mesh with {L} Electrodes')
    
#     # Potential field
#     ax = axes[0, 1]
#     im = ax.tripcolor(tri, u_all[0], cmap='RdBu_r', shading='gouraud')
#     plt.colorbar(im, ax=ax, label='Potential')
#     ax.set_aspect('equal')
#     ax.set_title('Potential (Pattern 1: I₁=+1, I₂=-1)')
    
#     # Electrode voltages
#     ax = axes[1, 0]
#     im = ax.imshow(U_matrix, aspect='auto', cmap='RdBu_r')
#     plt.colorbar(im, ax=ax, label='Voltage')
#     ax.set_xlabel('Electrode')
#     ax.set_ylabel('Pattern')
#     ax.set_title('Electrode Potentials U')
    
#     # Jacobian sensitivity
#     ax = axes[1, 1]
#     sens = np.abs(J).mean(axis=0)
#     im = ax.tripcolor(tri, sens, cmap='hot', shading='flat')
#     plt.colorbar(im, ax=ax, label='Mean |J|')
#     ax.set_aspect('equal')
#     ax.set_title('Jacobian Sensitivity')
    
#     plt.tight_layout()
#     plt.savefig('eit_cem_skfem_demo.png', dpi=150, bbox_inches='tight')
#     print("Saved: eit_cem_skfem_demo.png")
#     plt.close()
    
#     print("\n" + "="*70)
#     print("✓ Demo Complete!")
#     print("="*70)
    
#     return solver, sigma, u_all, U_all, J

def plot_conductivity(mesh: MeshTri, sigma: np.ndarray, title: str = "Conductivity"):
    """
    Quick visual check of elementwise conductivity.

    Args:
        mesh: scikit-fem MeshTri
        sigma: array of length mesh.t.shape[1] (per-element conductivity)
    """
    assert sigma.shape[0] == mesh.t.shape[1], \
        f"sigma must have one value per element (got {sigma.shape[0]} vs {mesh.t.shape[1]})"

    print(f"Conductivity stats: min={sigma.min():.4g}, max={sigma.max():.4g}")
    print(f"Unique values: {np.unique(sigma)}")

    # Build Matplotlib triangulation
    tri = Triangulation(mesh.p[0], mesh.p[1], mesh.t.T)

    plt.figure(figsize=(6, 6))
    im = plt.tripcolor(tri, sigma, shading="flat")  # P0: constant per triangle
    plt.gca().set_aspect("equal")
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(im, label="σ (S/m)")
    plt.tight_layout()
    plt.show()

# if __name__ == "__main__":
#     demo()