function evoi = expected_voi_gppe(covfunc_t, covfunc_x, theta, f, Kx, Kinv, ...
    W, L, t, x, train_pairs, idx_global, ind_t, ind_x, test_pair, fbest)
% evoi = expected_voi_gppe(covfunc_t, covfunc_x, theta, f, Kx, Kinv, ...
%    W, L, t, x, train_pairs, idx_global, ind_t, ind_x, test_pair, fbest)
% 
% Computes the expected value of information of including queries involving
% the pair test_pair. It  asssumes that:
%     M is the number of training users + 1
%     t(M) = tstar
%     train_pairs{M} exists
%
% INPUT:
%   - covfunc_t: Covariance function on user space 
%   - covfunc_x: Covariance function on item space 
%   - theta: [theta_t; theta_x; theta_sigma]: vector of hyperparameters 
%        theta_t and theta_x are the hyperparameters of the covariances. 
%       theta_sigma = log (sigma.
%   - f: The current mode of the posterior 
%   - Kx: The covariance matrix on item space
%   - Kinv: The inverse covariance of the full system
%   - W: The matrix of negative second derivatives (wrt f) of the conditional likelihood
%   - L: chol(W + Kinv)' 
%   - t: The matrix of users' features (including the test user) 
%   - x: The matrix of item features 
%   - train_pairs: Cell array of M elements. Each element is a O_m x 2 matrix 
%       where O_m is the number of preferences observed for the corresponding
%       user. Each row all_pairs{m} contains a preference relation 
%       of the form train_pairs{m}(1) > train_pairs{m}(2)   
%   - idx_global: The unique global indices of the observed preferences
%   - ind_t: Indices of seen tasks
%   - ind_x: Indices of seen items
%   - test_pair: The 2d vector representing the test query
%           test_pair(1)>test_pair(2)?
%   - fbest: An estimate of the best utility function value
%
% OUTPUT:
%   evoi: The expected value of information of asking the query
%       test_pair(1)>test_pair(2)?

% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last update: 22/05/2012


M = size(t,1);
N = size(x,1);
tstar = t(M,:);

 
% mel before adding test_pair 
% mei = maximum_expected_improvement_gppe(covfunc_t, covfunc_x, theta, f, Kx, ...
%     Kinv, W, L, t, x, idx_global, ind_t, ind_x, tstar, 1 : N, fbest);
mei = 0;

% predictive distribution under current data
p_12  = predict_gppe_laplace(covfunc_t, covfunc_x, theta, f, Kx, Kinv, ...
    W, L, t, x, idx_global, ind_t, ind_x, tstar, test_pair);
p_21  = 1 - p_12;

%% We recompute Laplace approximation and p(f|D U {q_{12})
% train_pairs
train_pairs{M} = [train_pairs{M}; test_pair];
[idx_global_1, idx_global_2] = compute_global_index(train_pairs, N);
idx_global = unique([idx_global_1; idx_global_2]); 
[ind_x ind_t] = ind2sub([N M], idx_global); % indices of "seen" data-points and tasks

[f_new Kx_new, Kinv_new, W_new, L_new] = approx_gppe_laplace_fast(covfunc_t, ...
    covfunc_x, theta, t, x, train_pairs, idx_global, ...
    idx_global_1, idx_global_2, ind_t, ind_x, M, N);

mei_12 = maximum_expected_improvement_gppe(covfunc_t, covfunc_x, theta, f_new, ...
    Kx_new, Kinv_new, W_new, L_new, t, x, idx_global, ind_t, ind_x, tstar, 1 : N, fbest);


%% We recompute Laplace approximation and p(f|D U {q_{21})
train_pairs{M}(end,:) = []; % remove previous assignment
train_pairs{M} = [train_pairs{M}; fliplr(test_pair)];
[idx_global_1, idx_global_2] = compute_global_index(train_pairs, N);
idx_global = unique([idx_global_1; idx_global_2]);
[ind_x ind_t] = ind2sub([N M], idx_global); % indices of "seen" data-points and tasks

[f_new Kx_new, Kinv_new, W_new, L_new] = approx_gppe_laplace_fast(covfunc_t, ...
    covfunc_x, theta, t, x, train_pairs, idx_global, ...
    idx_global_1, idx_global_2, ind_t, ind_x, M, N);

mei_21 = maximum_expected_improvement_gppe(covfunc_t, covfunc_x, theta, ...
    f_new, Kx_new, Kinv_new, W_new, L_new, t, x, idx_global, ind_t, ind_x, tstar, 1 : N, fbest);



%%  evoi = <MEL(D U q_{ij})> - MEL(D)
evoi = (p_12*mei_12 + p_21*mei_21) - mei;


return;


