function sigma_final = gauss_newton(sigma_init, rec_fmdl, measurements, options)
% GAUSS_NEWTON Gauss-Newton with L2 regularization
%
% Solves the EIT inverse problem:
%   minimize: (1/2)||F(σ) - m||² + (λ/2)||σ - σ_ref||²
%
% Uses linearized Gauss-Newton method with Tikhonov (L2) regularization
%
% Inputs:
%   sigma_init    - Initial conductivity estimate (n_elements x 1)
%   rec_fmdl      - EIDORS forward model structure
%   measurements  - Target measurements (n_meas x 1)
%   options       - Structure with fields:
%                   .n_iterations - Number of iterations (default: 20)
%                   .lambda       - L2 regularization parameter (default: 0.01)
%                   .min_cond     - Minimum conductivity (default: 1e-6)
%                   .sigma_ref    - Reference conductivity (default: sigma_init)
%                   .alpha_init   - Initial step size (default: 1.0)
%                   .beta         - Backtracking factor (default: 0.5)
%                   .max_ls_iter  - Max line search iterations (default: 20)
%                   .verbose      - Display progress (default: true)

    % Set default options
    if ~isfield(options, 'n_iterations'), options.n_iterations = 20; end
    if ~isfield(options, 'lambda'), options.lambda = 0.01; end
    if ~isfield(options, 'min_cond'), options.min_cond = 1e-6; end
    if ~isfield(options, 'sigma_ref'), options.sigma_ref = sigma_init; end
    if ~isfield(options, 'alpha_init'), options.alpha_init = 1.0; end
    if ~isfield(options, 'beta'), options.beta = 0.5; end
    if ~isfield(options, 'max_ls_iter'), options.max_ls_iter = 20; end
    if ~isfield(options, 'verbose'), options.verbose = true; end
    
    sigma = sigma_init(:);
    measurements = measurements(:);
    sigma_ref = options.sigma_ref(:);
    
    if options.verbose
        fprintf('\n=== Gauss-Newton with L2 Regularization ===\n');
        fprintf('lambda = %.4e\n\n', options.lambda);
    end
    
    % Main Gauss-Newton iteration loop
    for iter = 1:options.n_iterations
        
        % ====================================================================
        % STEP 1: Compute Forward Problem and Jacobian
        % ====================================================================
        img = mk_image(rec_fmdl, sigma);
        Ui = fwd_solve(img);
        J = calc_jacobian(img);
        
        % Residual
        residual = Ui.meas(:) - measurements;
        
        % Current objective value
        obj_current = compute_objective(sigma, sigma_ref, options.lambda, residual);
        
        % ====================================================================
        % STEP 2: Solve Gauss-Newton System with L2 Regularization
        % ====================================================================
        % The Gauss-Newton update solves:
        %   (J'*J + λ*I) * Δσ = -J'*r - λ*(σ - σ_ref)
        %
        % This is the normal equation for the linearized least squares problem:
        %   minimize: (1/2)||J*Δσ + r||² + (λ/2)||Δσ + (σ - σ_ref)||²
        
        JtJ = J' * J;
        Jtr = J' * residual;
        
        % Add L2 regularization to Hessian approximation
        H = JtJ + options.lambda * speye(size(JtJ, 1));
        
        % Right-hand side: gradient of data term + L2 term
        rhs = -Jtr - options.lambda * (sigma - sigma_ref);
        
        % Solve for search direction (this is the Newton step)
        delta_sigma = H \ rhs;
        
        % ====================================================================
        % STEP 3: Line Search (Backtracking)
        % ====================================================================
        alpha = options.alpha_init;
        accepted = false;
        
        for ls_iter = 1:options.max_ls_iter
            
            % Trial step
            sigma_new = sigma + alpha * delta_sigma;
            
            % Enforce minimum conductivity constraint
            sigma_new = max(sigma_new, options.min_cond);
            
            % Evaluate objective at new point
            img_new = mk_image(rec_fmdl, sigma_new);
            Ui_new = fwd_solve(img_new);
            residual_new = Ui_new.meas(:) - measurements;
            obj_new = compute_objective(sigma_new, sigma_ref, options.lambda, residual_new);
            
            % Check for sufficient decrease
            if obj_new < obj_current
                accepted = true;
                break;
            end
            
            % Reduce step size
            alpha = options.beta * alpha;
            
            if alpha < 1e-12
                if options.verbose
                    warning('Step size too small at iteration %d', iter);
                end
                break;
            end
        end
        
        % ====================================================================
        % STEP 4: Update and Check Convergence
        % ====================================================================
        if ~accepted
            if options.verbose
                fprintf('  Iter %3d: Line search failed, terminating\n', iter);
            end
            break;
        end
        
        % Compute metrics
        rel_change = norm(sigma_new - sigma) / max(norm(sigma), eps);
        obj_decrease = obj_current - obj_new;
        
        % Update sigma
        sigma = sigma_new;
        
        % Display progress
        if options.verbose
            fprintf('  Iter %3d: Obj = %.6e, ΔObj = %.3e, α = %.4f, LS = %2d, RelChg = %.3e\n', ...
                   iter, obj_new, obj_decrease, alpha, ls_iter, rel_change);
        end
        
        % Check convergence
        if rel_change < 1e-6
            if options.verbose
                fprintf('\n  ✓ Converged: relative change < 1e-6\n');
            end
            break;
        end
        
        if obj_decrease < 1e-10
            if options.verbose
                fprintf('\n  ✓ Converged: objective change < 1e-10\n');
            end
            break;
        end
    end
    
    sigma_final = sigma;
end

function obj = compute_objective(sigma, sigma_ref, lambda, residual)
% COMPUTE_OBJECTIVE Total objective value
%
% obj = (1/2)||residual||² + (λ/2)||σ - σ_ref||²

    % Data fidelity term
    data_term = 0.5 * (residual' * residual);
    
    % L2 regularization term (Tikhonov)
    deviation = sigma - sigma_ref;
    l2_term = 0.5 * lambda * (deviation' * deviation);
    
    obj = data_term + l2_term;
end