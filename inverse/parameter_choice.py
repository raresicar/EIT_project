# src/reconstructor/parameter_selection.py

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Callable


class LCurveSelector:
    """L-curve parameter selection for single regularization parameter"""
    
    def __init__(self, reconstructor, U_measured: np.ndarray):
        self.reconstructor = reconstructor
        self.U_measured = U_measured
        self.results = []
        
    def compute_lcurve(self, alpha_range: np.ndarray, **kwargs) -> List[Dict]:
        """
        Compute L-curve by reconstructing with different alpha values
        
        Parameters:
        -----------
        alpha_range : array
            Range of regularization parameters
        **kwargs : dict
            Additional arguments for reconstructor.forward()
            
        Returns:
        --------
        results : list of dict
            Each dict contains: alpha, misfit, reg_norm, sigma
        """
        self.results = []
        
        for alpha in alpha_range:
            # Update regularization parameter
            self.reconstructor.lamb = alpha
            
            # Reconstruct
            sigma_reco = self.reconstructor.forward(self.U_measured, **kwargs)
            
            # Compute forward pass for misfit
            _, U_pred = self.reconstructor.eit_solver.forward_solve(sigma_reco)
            U_pred = np.asarray(U_pred).flatten()
            
            # Compute metrics
            residual = U_pred - self.U_measured.flatten()
            misfit = 0.5 * np.linalg.norm(residual)**2
            
            # Regularization norm (depends on type)
            if hasattr(self.reconstructor, 'Ltv'):
                # TV regularization
                reg_norm = np.sqrt(
                    ((self.reconstructor.Ltv @ sigma_reco.x.array[:])**2 + 
                     self.reconstructor.beta).sum()
                )
            else:
                # L2 or other
                reg_norm = np.linalg.norm(sigma_reco.x.array[:])
            
            self.results.append({
                'alpha': alpha,
                'misfit': misfit,
                'reg_norm': reg_norm,
                'sigma': sigma_reco.x.array[:].copy()
            })
            
        return self.results
    
    def find_corner(self, method='curvature') -> float:
        """
        Find corner of L-curve
        
        Parameters:
        -----------
        method : str
            'curvature' - maximum curvature
            'distance' - minimum distance to origin
            
        Returns:
        --------
        alpha_opt : float
            Optimal regularization parameter
        """
        if not self.results:
            raise ValueError("Must call compute_lcurve first")
        
        misfits = np.array([r['misfit'] for r in self.results])
        reg_norms = np.array([r['reg_norm'] for r in self.results])
        
        # Work in log-log space
        log_misfit = np.log10(misfits + 1e-10)
        log_reg = np.log10(reg_norms + 1e-10)
        
        if method == 'curvature':
            # Compute curvature using finite differences
            curvature = self._compute_curvature(log_misfit, log_reg)
            idx_opt = np.argmax(np.abs(curvature))
            
        elif method == 'distance':
            # Normalize and find minimum distance to origin
            log_misfit_norm = (log_misfit - log_misfit.min()) / (log_misfit.max() - log_misfit.min())
            log_reg_norm = (log_reg - log_reg.min()) / (log_reg.max() - log_reg.min())
            distance = np.sqrt(log_misfit_norm**2 + log_reg_norm**2)
            idx_opt = np.argmin(distance)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return self.results[idx_opt]['alpha']
    
    @staticmethod
    def _compute_curvature(x, y):
        """Compute discrete curvature using Menger curvature"""
        n = len(x)
        curvature = np.zeros(n)
        
        for i in range(1, n-1):
            # Three points for curvature
            p0 = np.array([x[i-1], y[i-1]])
            p1 = np.array([x[i], y[i]])
            p2 = np.array([x[i+1], y[i+1]])
            
            # Menger curvature
            area = 0.5 * abs((p1[0]-p0[0])*(p2[1]-p0[1]) - 
                            (p2[0]-p0[0])*(p1[1]-p0[1]))
            d01 = np.linalg.norm(p1-p0)
            d12 = np.linalg.norm(p2-p1)
            d02 = np.linalg.norm(p2-p0)
            
            curvature[i] = 4*area / (d01*d12*d02 + 1e-10)
        
        return curvature
    
    def plot_lcurve(self, alpha_opt=None):
        """Plot L-curve with optional optimal point marked"""
        if not self.results:
            raise ValueError("Must call compute_lcurve first")
        
        misfits = np.array([r['misfit'] for r in self.results])
        reg_norms = np.array([r['reg_norm'] for r in self.results])
        alphas = np.array([r['alpha'] for r in self.results])
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        scatter = ax.scatter(np.log10(misfits), np.log10(reg_norms), 
                           c=np.log10(alphas), cmap='viridis', s=50)
        ax.plot(np.log10(misfits), np.log10(reg_norms), 'k-', alpha=0.3)
        
        if alpha_opt is not None:
            idx = np.argmin(np.abs(alphas - alpha_opt))
            ax.scatter(np.log10(misfits[idx]), np.log10(reg_norms[idx]), 
                      c='red', s=200, marker='*', edgecolors='black',
                      label=f'Optimal α={alpha_opt:.2e}')
            ax.legend()
        
        ax.set_xlabel('log₁₀(Data Misfit)')
        ax.set_ylabel('log₁₀(Regularization Norm)')
        ax.set_title('L-Curve')
        plt.colorbar(scatter, ax=ax, label='log₁₀(α)')
        
        return fig


class DiscrepancyPrincipleSelector:
    """Discrepancy principle for parameter selection"""
    
    def __init__(self, reconstructor, U_measured: np.ndarray, noise_level: float, tau: float = 1.2):
        self.reconstructor = reconstructor
        self.U_measured = U_measured
        self.noise_level = noise_level
        self.tau = tau
        
    def find_parameter(self, alpha_init: float = 1e-3, max_iter: int = 20, **kwargs) -> float:
        """
        Find alpha such that ||F(sigma_alpha) - U|| ≈ tau * delta
        
        Parameters:
        -----------
        alpha_init : float
            Initial guess for alpha
        max_iter : int
            Maximum iterations
            
        Returns:
        --------
        alpha_opt : float
            Optimal regularization parameter
        """
        alpha = alpha_init
        target_residual = self.tau * self.noise_level
        
        for iteration in range(max_iter):
            self.reconstructor.lamb = alpha
            
            sigma_reco = self.reconstructor.forward(self.U_measured, **kwargs)
            
            _, U_pred = self.reconstructor.eit_solver.forward_solve(sigma_reco)
            U_pred = np.asarray(U_pred).flatten()
            
            residual_norm = np.linalg.norm(U_pred - self.U_measured.flatten())
            
            print(f"Iter {iteration}: α={alpha:.2e}, residual={residual_norm:.2e}, target={target_residual:.2e}")
            
            if abs(residual_norm - target_residual) < 0.1 * self.noise_level:
                print("Converged!")
                return alpha
            
            # Adjust alpha
            if residual_norm > target_residual:
                alpha *= 0.7  # Too much regularization
            else:
                alpha *= 1.3  # Too little regularization
        
        print("Warning: Did not converge")
        return alpha


class GCVSelector:
    """Generalized Cross Validation for parameter selection"""
    
    def __init__(self, reconstructor, U_measured: np.ndarray):
        self.reconstructor = reconstructor
        self.U_measured = U_measured
        
    def compute_gcv(self, alpha_range: np.ndarray, **kwargs) -> Tuple[float, np.ndarray]:
        """
        Compute GCV scores for different alpha values
        
        GCV(α) = ||F(σ_α) - U||² / (M - trace(influence))²
        
        Note: This is expensive as it requires computing influence matrix
        """
        gcv_scores = []
        M = len(self.U_measured.flatten())
        
        for alpha in alpha_range:
            self.reconstructor.lamb = alpha
            
            sigma_reco = self.reconstructor.forward(self.U_measured, **kwargs)
            
            _, U_pred = self.reconstructor.eit_solver.forward_solve(sigma_reco)
            U_pred = np.asarray(U_pred).flatten()
            
            residual = U_pred - self.U_measured.flatten()
            numerator = np.linalg.norm(residual)**2
            
            # Approximate trace (expensive - use randomized estimate for large problems)
            # For now, use simplified formula
            denominator = M  # Simplified - full version needs Jacobian analysis
            
            gcv_scores.append(numerator / denominator**2)
        
        idx_opt = np.argmin(gcv_scores)
        return alpha_range[idx_opt], np.array(gcv_scores)