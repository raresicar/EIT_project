import numpy as np
from tqdm import tqdm
from scipy.sparse import csr_array, diags
import time 

from forward.eit_forward_skfem import EIT
from inverse.reconstructor import Reconstructor


class GaussNewtonSolverTV(Reconstructor):
    def __init__(
        self,
        eit_solver: EIT,
        GammaInv: np.array,
        num_steps: int = 8,
        lamb: float = 0.04,
        beta: float = 1e-6,
        Uel_background: np.array = None,
        clip=[0.001, 3.0],       
        **kwargs,
    ):
        super().__init__(eit_solver)

        self.Ltv = self.construct_tv_matrix()

        self.num_steps = num_steps
        self.lamb = lamb
        self.beta = beta
        self.GammaInv = GammaInv
        self.Uel_background = Uel_background
        self.clip = clip

    def construct_tv_matrix(self):
        """
        Construct TV matrix for scikit-fem mesh.
        
        For scikit-fem MeshTri:
            mesh.t: (3, n_elements) - element connectivity
            mesh.facets: (2, n_facets) - facet connectivity
            mesh.t2f: (3, n_elements) - element to facet mapping
            mesh.f2t: (2, n_facets) - facet to element mapping (-1 if boundary)
        """
        mesh = self.eit_solver.mesh
        n_elements = mesh.t.shape[1]
        
        # Get facet-to-element connectivity
        f2t = mesh.f2t  # (2, n_facets)
        n_facets = f2t.shape[1]
        
        rows = []
        cols = []
        data = []
        
        row_idx = 0
        for facet_idx in range(n_facets):
            # Get elements adjacent to this facet
            adjacent_cells = f2t[:, facet_idx]
            
            # Only interior facets (both elements exist)
            if adjacent_cells[0] >= 0 and adjacent_cells[1] >= 0:
                rows.append(row_idx)
                rows.append(row_idx)
                cols.append(adjacent_cells[0])
                cols.append(adjacent_cells[1])
                data.append(1)
                data.append(-1)
                row_idx += 1
        
        return csr_array((data, (rows, cols)), shape=(row_idx, n_elements))

    def forward(self, Umeas: np.array, **kwargs):
        Umeas = Umeas.flatten()

        verbose = kwargs.get("verbose", False)
        debugging = kwargs.get("debugging", False)
        sigma_init = kwargs.get("sigma_init", None)

        if sigma_init is None:
            # Initialize with homogeneous conductivity
            n_elements = self.eit_solver.mesh.t.shape[1]
            sigma = np.ones(n_elements) * self.backCond
        else:
            sigma = sigma_init.copy()

        sigma_old = sigma.copy()

        disable = not verbose
        with tqdm(total=self.num_steps, disable=disable) as pbar:
            for i in range(self.num_steps):
                sigma_k = sigma.copy()
                sigma_old = sigma.copy()

                if debugging:
                    print("Forward Solve...")
                    t1 = time.time()
                    
                u_all, Usim = self.eit_solver.forward_solve(sigma_k)
                Usim = np.asarray(Usim).flatten()
                
                if debugging:
                    t2 = time.time() 
                    print(f"Took {t2-t1}s")

                if debugging:
                    print("Create Jacobian...")
                    t1 = time.time()
                    
                J = self.eit_solver.calc_jacobian(sigma_k, u_all)
                
                if debugging:
                    t2 = time.time() 
                    print(f"Took {t2-t1}s")

                if self.Uel_background is not None and i == 0:
                    deltaU = self.Uel_background.flatten() - Umeas
                else:
                    deltaU = Usim - Umeas

                if debugging:
                    print("Create A and b...")
                    t1 = time.time()
                    
                A = np.linalg.multi_dot([J.T, np.diag(self.GammaInv), J])
                b = np.linalg.multi_dot([J.T, np.diag(self.GammaInv), deltaU])

                L_sigma = np.abs(self.Ltv @ sigma_k) ** 2
                eta = np.sqrt(L_sigma + self.beta)
                E = diags(1 / eta)
                LTEL = self.Ltv.T @ E @ self.Ltv
                A = A + self.lamb * LTEL
                b = b - self.lamb * LTEL @ sigma_k

                if debugging:
                    t2 = time.time() 
                    print(f"Took {t2-t1}s")

                delta_sigma = np.linalg.solve(A, b)

                step_sizes = np.linspace(0.01, 1.0, 6)
                losses = []
                for step_size in step_sizes:
                    sigma_new = sigma + step_size * delta_sigma
                    sigma_new = np.clip(sigma_new, self.clip[0], self.clip[1])

                    _, Utest = self.eit_solver.forward_solve(sigma_new)
                    Utest = np.asarray(Utest).flatten()

                    tv_value = self.lamb * np.sqrt(((self.Ltv @ sigma_new) ** 2) + self.beta).sum()
                    meas_value = 0.5 * np.sum((np.diag(self.GammaInv) @ (Utest - Umeas)) ** 2)
                    losses.append(meas_value + tv_value)

                step_size = step_sizes[np.argmin(losses)]
                sigma = sigma + step_size * delta_sigma
                sigma = np.clip(sigma, self.clip[0], self.clip[1])

                s = np.linalg.norm(sigma - sigma_old) / np.linalg.norm(sigma)
                loss = np.min(losses)

                pbar.set_description(
                    f"Relative Change: {np.format_float_positional(s, 4)} | Obj. fun: {np.format_float_positional(loss, 4)} | Step size: {np.format_float_positional(step_size, 4)}"
                )
                pbar.update(1)

        return sigma


    def forward_cg(self, Umeas: np.array, **kwargs):
        Umeas = Umeas.flatten()

        verbose = kwargs.get("verbose", False)
        debugging = kwargs.get("debugging", False)
        sigma_init = kwargs.get("sigma_init", None)

        if sigma_init is None:
            n_elements = self.eit_solver.mesh.t.shape[1]
            sigma = np.ones(n_elements) * self.backCond
        else:
            sigma = sigma_init.copy()

        sigma_old = sigma.copy()

        disable = not verbose
        with tqdm(total=self.num_steps, disable=disable) as pbar:
            for i in range(self.num_steps):
                sigma_k = sigma.copy()
                sigma_old = sigma.copy()

                if debugging:
                    print("Forward Solve...")
                    t1 = time.time()
                    
                u_all, Usim = self.eit_solver.forward_solve(sigma_k)
                Usim = np.asarray(Usim).flatten()
                
                if debugging:
                    t2 = time.time() 
                    print(f"Took {t2-t1}s")

                if debugging:
                    print("Create Jacobian...")
                    t1 = time.time()
                    
                J = self.eit_solver.calc_jacobian(sigma_k, u_all)
                
                if debugging:
                    t2 = time.time() 
                    print(f"Took {t2-t1}s")

                if self.Uel_background is not None and i == 0:
                    deltaU = self.Uel_background.flatten() - Umeas
                else:
                    deltaU = Usim - Umeas
                
                if debugging:
                    print("Solve System with CG...")
                    t1 = time.time()
                    
                L_sigma = (self.Ltv @ sigma)**2
                eta = np.sqrt(L_sigma + self.beta)
                E = diags(1 / eta)
                LTET = self.Ltv.T @ E @ self.Ltv
                
                def Afwd(x):
                    return np.linalg.multi_dot([J.T, np.diag(self.GammaInv), J, x]) + self.lamb * LTET @ x

                b = np.linalg.multi_dot([J.T, np.diag(self.GammaInv), deltaU])
                b = b - self.lamb * self.Ltv.T @ E @ self.Ltv @ sigma

                delta_sigma = conjugate_gradient(Afwd, b, max_iter=30, verbose=debugging)
                
                if debugging:
                    t2 = time.time()
                    print(f"Took {t2-t1}s")

                step_sizes = np.linspace(0.01, 1.0, 6)
                losses = []
                for step_size in step_sizes:
                    sigma_new = sigma + step_size * delta_sigma
                    sigma_new = np.clip(sigma_new, self.clip[0], self.clip[1])

                    _, Utest = self.eit_solver.forward_solve(sigma_new)
                    Utest = np.asarray(Utest).flatten()

                    tv_value = self.lamb * np.sqrt(((self.Ltv @ sigma_new) ** 2) + self.beta).sum()
                    meas_value = 0.5 * np.sum((np.diag(self.GammaInv) @ (Utest - Umeas)) ** 2)
                    losses.append(meas_value + tv_value)

                step_size = step_sizes[np.argmin(losses)]
                sigma = sigma + step_size * delta_sigma
                sigma = np.clip(sigma, self.clip[0], self.clip[1])

                s = np.linalg.norm(sigma - sigma_old) / np.linalg.norm(sigma)
                loss = np.min(losses)

                pbar.set_description(
                    f"Relative Change: {np.format_float_positional(s, 4)} | Obj. fun: {np.format_float_positional(loss, 4)} | Step size: {np.format_float_positional(step_size, 4)}"
                )
                pbar.update(1)

        return sigma


def conjugate_gradient(
    A,
    b,
    max_iter: float = 1e2,
    tol: float = 1e-5,
    eps: float = 1e-8,
    init=None,
    verbose=False,
):
    """
    Standard conjugate gradient algorithm.
    """
    if init is not None:
        x = init
    else:
        x = np.zeros_like(b)

    r = b - A(x)
    p = r
    rsold = np.dot(r, r).real
    flag = True
    tol = np.dot(b, b).real * (tol**2)
    
    for _ in range(int(max_iter)):
        Ap = A(p)
        alpha = rsold / (np.dot(p, Ap) + eps)
        x = x + p * alpha
        r = r - Ap * alpha
        rsnew = np.dot(r, r).real
        
        if rsnew < tol:
            if verbose:
                print("CG Converged at iteration", _)
            flag = False
            break
            
        p = r + p * (rsnew / (rsold + eps))
        rsold = rsnew

    if flag and verbose:
        print(f"CG did not converge: Residual {rsnew}")

    return x