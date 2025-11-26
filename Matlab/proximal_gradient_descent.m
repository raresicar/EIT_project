function sigma_final = proximal_gradient_descent(sigma_init, rec_fmdl, measurements, options)
% PROXIMAL_GRADIENT_DESCENT Iterative optimization with L1+L2 regularization
%
% Inputs:
%   sigma_init    - Initial conductivity estimate (n_elements x 1)
%   rec_fmdl      - EIDORS forward model structure
%   measurements  - Target measurements (n_meas x 1)
%   options       - Structure with fields:
%                   .n_iterations - Number of iterations (default: 20)
%                   .lambda_l2    - L2 regularization parameter (default: 0.01)
%                   .lambda_l1    - L1 regularization parameter (default: 0.001)
%                   .min_cond     - Minimum conductivity (default: 1e-3)
%                   .step_size    - Step size strategy: 'fixed', 'linesearch', or 'backtracking'
%                                   (default: 'backtracking')
%                   .alpha        - Fixed step size if step_size='fixed' (default: 0.1)
%                   .verbose      - Display progress (default: true)

    % Set default options
    if ~isfield(options, 'n_iterations'), options.n_iterations = 50; end
    if ~isfield(options, 'lambda_l2'), options.lambda_l2 = 0.01; end
    if ~isfield(options, 'lambda_l1'), options.lambda_l1 = 0.001; end
    if ~isfield(options, 'min_cond'), options.min_cond = 1e-6; end
    if ~isfield(options, 'beta'), options.beta = 0.5; end
    if ~isfield(options, 'max_ls_iter'),  options.max_ls_iter  = 20; end
    if ~isfield(options, 'verbose'), options.verbose = true; end
    
    sigma = sigma_init;
    measurements = measurements(:);

    % Initialize step size (will be adapted)
    lambda_prev = 100;
    
    % Main iteration loop
    for iter = 1:options.n_iterations
        % Compute gradient of data fidelity term
        [f, grad_f] = gradient_objective(sigma, rec_fmdl, measurements);

        lambda_k = lambda_prev;
        sigma_old = sigma;
        
        % Line search loop
        for ls_iter = 1:options.max_ls_iter

            % Compute proximal gradient step
            z = proximal_step(sigma, grad_f, lambda_k, ...
                 options);
            
            % Evaluate data part at new point
            [f_z, ~] = gradient_objective(z, rec_fmdl, measurements);
            fprintf('    grad_f: min=%.3e, max=%.3e, norm=%.3e\n', ...
            min(grad_f), max(grad_f), norm(grad_f));

            
            % Compute upper bound
            f_lambda = f + grad_f' * (z - sigma) + (1/(2*lambda_k)) * ...
                ((z - sigma)' * (z - sigma));
            
            % Check acceptance
            if f_z <= f_lambda
                break;
            end
            
            % Reduce step size
            lambda_k = options.beta * lambda_k;
            
            if lambda_k < 1e-12
                warning('Step size became too small at iteration %d', iter);
                break;
            end
        end
        
        % Store step size for next iteration
        lambda_prev = lambda_k;
        
        % Update sigma
        sigma = z;
        sigma_range = max(sigma) - min(sigma);
        fprintf('       sigma: min=%.3e, max=%.3e, range=%.3e\n', ...
        min(sigma), max(sigma), sigma_range);
        
        % ---- Convergence check (relative change) -------------------------
        rel_change = norm(sigma - sigma_old) / max(norm(sigma_old), eps);

        if options.verbose
            fprintf('  iter %3d: f=%.3e, rel_change=%.3e, lambda=%.3e\n', ...
                    iter, f_z, rel_change, lambda_k);
        end

        if rel_change < 1e-6
            if options.verbose
                fprintf('  Converged at iteration %d (relative change < 1e-6)\n', iter);
            end
            break;
        end
    end
    
    sigma_final = sigma;

end

function z = proximal_step(x, grad_f, lambda, options)
% PROXIMAL_STEP Proximal step
    
    % One proximal gradient step for:
    % g(σ) = λ1 ||σ - σ_ref||_1 + (λ2/2) ||σ||_2^2

    v = x - lambda * grad_f;          % gradient step on f

    % --- L1 around sigma_ref (soft threshold) ---
    if options.lambda_l1 > 0
        y = v;                  % shift
        t = lambda * options.lambda_l1;
        y = sign(y) .* max(abs(y) - t, 0);  % soft-threshold
        v = y;                  % back-shift
    end

    % --- L2 (ridge) part: shrinkage ---
    if options.lambda_l2 > 0
        v = v ./ (1 + lambda * options.lambda_l2);
    end

    if isfield(options, 'min_cond')
        v = max(v, options.min_cond);
    end
    z = v;

end

function [f, grad_f] = gradient_objective(sigma, rec_fmdl, measurements)
% GRADIENT_OBJECTIVE

    img = mk_image(rec_fmdl, sigma);
    Ui = fwd_solve(img);
    residual = Ui.meas(:) - measurements(:);
    f = 0.5 * (residual' * residual);
    J = calc_jacobian(img);
    grad_f = J' * residual;

end

function obj = compute_objective(sigma, lambda_l1, lambda_l2, rec_fmdl, measurements)
% COMPUTE_OBJECTIVE

    % Forward solve
    img = mk_image(rec_fmdl, sigma);
    Ui = fwd_solve(img);
    
    % Data fidelity term
    residual = Ui.meas(:) - measurements(:);
    data_term = 0.5 * (residual' * residual);
    
    % L2 regularization on sigma
    l2_term = 0.5 * lambda_l2 * (sigma * sigma);
    
    % L1 regularization on deviation
    l1_term = lambda_l1 * sum(abs(sigma));
    
    obj = data_term + l2_term + l1_term;
end