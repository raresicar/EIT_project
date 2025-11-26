% =========================================================================

% =========================================================================

clear all; clc; close all;

% Start EIDORS
eidors_root = 'D:\EIDORS\eidors-v3.12-ng\eidors';
addpath(genpath(eidors_root))
run(fullfile(eidors_root, 'eidors_startup.m'))
disp('EIDORS ready!');

% Remove Octave specific functions
rmpath('D:\EIDORS\eidors-v3.12-ng\eidors\overloads\octave');
disp('Overloads removed!');

%% Set up

% Where the data will be saved
filename = "simulated_data.mat";

% Number of samples
n_samples = 1;

% Model Geometry (meters)
model_radius = 0.14;
model_height = 0; % 2D model
n_elec = 16;
elec_rings = 1;
elec_width = 0.02;
elec_height = 0.02;

% Simulation mesh size -> finer mesh for forward problem
sim_model_maxsz = 0.005;
sim_elec_maxsz = sim_model_maxsz / 3;

% Reconstruction mesh size -> coarser mesh for inverse problem
rec_model_maxsz = 0.0065;
rec_elec_maxsz = rec_model_maxsz / 3.5;

% ---- Conductivity Distribution Parameters

% background conductivity
background_conds = rand(n_samples, 1) * 0.03 + 0.12; 

% we will either have more conductive or less conductive targets (w.r.t.
% the background)
target_conds = 0.19 + (randi(2, n_samples, 1) * 2 - 3) .* (rand(n_samples, 1) * 0.03 + 0.11); 

% the target is a circle at random 
target_theta = rand(n_samples, 1) * 2*pi; 
target_alpha = rand(n_samples, 1) * 0.08;
target_xs = target_alpha .* cos(target_theta);
target_ys = target_alpha .* sin(target_theta);
target_radii = rand(n_samples, 1) * 0.02 + 0.03;
min_cond = 0.001;

% ----
% Define current patterns
inj_pat = '{ad}';
meas_pat = '{mono}';
stim_pat_options = {
    'meas_current', ... % Measure on injecting electrodes
    'balance_meas' ... % This grounds the voltage data
};
amplitude = 0.003; % Amp
stim = mk_stim_patterns(n_elec, 1, inj_pat, meas_pat, stim_pat_options, amplitude);

% Added noise
noise_level = 0.01;

% How many iterations of the iterative reconstruction algorithm?
n_iterations = 100;

%% Create the models

% % Simulation Mesh
% sim_fmdl = ng_mk_ellip_models( ...
%     [model_height, model_radius, model_radius, sim_model_maxsz], ... 
%     [n_elec, elec_rings], ...
%     [elec_width, elec_height, sim_elec_maxsz] ...
% );
% figure();
% show_fem(sim_fmdl);
% title(sprintf("Simulation Mesh: %d Elements", size(sim_fmdl.elems, 1)));
% 
% % Reconstruction Mesh
% rec_fmdl = ng_mk_ellip_models( ...
%     [model_height, model_radius, model_radius, rec_model_maxsz], ...
%     [n_elec, elec_rings], ...
%     [elec_width, elec_height, rec_elec_maxsz] ...
% );
% figure();
% show_fem(rec_fmdl);
% title(sprintf("Reconstruction Mesh: %d Elements", size(rec_fmdl.elems, 1)));
% 
% % Set forward solver information for EIDORS
% sim_fmdl.stimulation = stim;
% sim_fmdl.solve      = @fwd_solve_1st_order;
% sim_fmdl.system_mat = @system_mat_1st_order;
% sim_fmdl.jacobian   = @jacobian_adjoint;
% rec_fmdl.stimulation = stim;
% rec_fmdl.solve      = @fwd_solve_1st_order;
% rec_fmdl.system_mat = @system_mat_1st_order;
% rec_fmdl.jacobian   = @jacobian_adjoint;
% 
% % Save the models
% save("eidors_models_v0.mat", "sim_fmdl", "rec_fmdl");

% Load the models
load("eidors_models_v0.mat", "sim_fmdl", "rec_fmdl");

% Simulation mesh centroids
sim_n_elements = size(sim_fmdl.elems, 1);
sim_centroids = [(sim_fmdl.nodes(sim_fmdl.elems(:, 1), 1) + sim_fmdl.nodes(sim_fmdl.elems(:, 2), 1) + sim_fmdl.nodes(sim_fmdl.elems(:, 3), 1)) / 3, ...
                 (sim_fmdl.nodes(sim_fmdl.elems(:, 1), 2) + sim_fmdl.nodes(sim_fmdl.elems(:, 2), 2) + sim_fmdl.nodes(sim_fmdl.elems(:, 3), 2)) / 3];

% Reconstruction mesh centroids
rec_n_elements = size(rec_fmdl.elems, 1);
rec_centroids = [(rec_fmdl.nodes(rec_fmdl.elems(:, 1), 1) + rec_fmdl.nodes(rec_fmdl.elems(:, 2), 1) + rec_fmdl.nodes(rec_fmdl.elems(:, 3), 1)) / 3, ...
                 (rec_fmdl.nodes(rec_fmdl.elems(:, 1), 2) + rec_fmdl.nodes(rec_fmdl.elems(:, 2), 2) + rec_fmdl.nodes(rec_fmdl.elems(:, 3), 2)) / 3];

%% Simulate the data

% Initialize storage
VSIM = zeros(n_samples, numel(stim(1).meas_pattern));

% For each sample, set the conductivity and then solve the forward problem
for i = 1:n_samples
    
    % Set the conductivity
    condunctivity = ones(sim_n_elements, 1) * background_conds(i);
    d = sqrt((sim_centroids(:, 1) - target_xs(i)).^2 + (sim_centroids(:, 2) - target_ys(i)).^2);
    condunctivity(d < target_radii(i)) = target_conds(i);

    % Solve the forward problem
    img = mk_image(sim_fmdl, condunctivity);
    U = fwd_solve(img);
    figure()
    show_fem(img)

    % Add noise to the simulated data
    Vsim = U.meas + noise_level * mean(abs(U.meas), 1) * normrnd(0, 1, size(U.meas));

    % Store the data
    VSIM(i, :) = Vsim(:);

end

%% Calculate Reconstructions (network inputs)

% Initialize storage
% Purpose: These will be inputs/outputs for training neural networks
X0 = zeros(n_samples, rec_n_elements); % Initialization: sigma_0
X1 = zeros(n_samples, rec_n_elements); % First update term: del_sigma_0
X2 = zeros(n_samples, rec_n_elements); % Final iterate: sigma_f
Y  = zeros(n_samples, rec_n_elements); % Ground Truth

% Solve the forward problem with conductivity of 1: This gives us a reference voltage pattern
% Used for homogeneous initialization
img1 = mk_image(rec_fmdl, 0.8);
U1 = fwd_solve(img1);

% For each sample, get initial conductivity and run iteration algorithm
for i = 1:n_samples

    fprintf("Sample %d\n", i);
    
    % Get ground truth
    conductivity = ones(rec_n_elements, 1) * background_conds(i);
    d = sqrt((rec_centroids(:, 1) - target_xs(i)).^2 + (rec_centroids(:, 2) - target_ys(i)).^2);
    conductivity(d < target_radii(i)) = target_conds(i);
    Y(i, :) = conductivity;
    
    % Get initialization
    conductivity_hom = VSIM(i, :)' \ U1.meas(:);
    sigma = ones(rec_n_elements, 1) * conductivity_hom;
    X0(i, :) = sigma;

    fprintf('       diff: min=%.3e, max=%.3e\n', ...
        min(sigma - conductivity), max(sigma - conductivity));
    
    % Set up options for proximal gradient descent
    opt.n_iterations = n_iterations;
    opt.lambda_l2 = 1e-3;      % L2 regularization parameter (adjust as needed)
    opt.lambda_l1 = 1e-6;     % L1 regularization parameter (adjust as needed)
    opt.min_cond = min_cond;
    opt.beta = 0.5;
    
    % Run proximal gradient descent
    sigma_final = proximal_gradient_descent(sigma, rec_fmdl, VSIM(i, :)', opt);
    
    
    % Plot for sanity
    figure()
    img = mk_image(rec_fmdl, conductivity);
    img.calc_colours.ref_level = background_conds(i);
    img.calc_colours.clim = 0.21;
    subplot(1,2,1); show_fem(img); eidors_colourbar(img)
    img = mk_image(rec_fmdl, sigma_final);
    img.calc_colours.ref_level = background_conds(i);
    img.calc_colours.clim = 0.21;
    subplot(1,2,2); show_fem(img); eidors_colourbar(img)

end

%% Save everything

save(filename, "sim_fmdl", "rec_fmdl", "VSIM", "Y", "X0", "X1", "X2");
fprintf("Done and saved :)\n");