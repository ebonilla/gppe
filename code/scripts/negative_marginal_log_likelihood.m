function nl = negative_marginal_log_likelihood(theta, covfunc_t, covfunc_x,...
    t, x, all_pairs, idx_global, idx_global_1, idx_global_2, ind_t, ind_x, M, N)
% nl = negative_marginal_log_likelihood(theta, covfunc_t, covfunc_x,...
%    t, x, all_pairs, idx_global, idx_global_1, idx_global_2, ind_t, ind_x, M, N)
%    
% Computes the negative marginal likelihood - log p (D | theta) of a gppe
% model
%
% INPUT:
%   - theta = [theta_t; theta_x; theta_sigma]: vector of hyperparameters
%       theta_t and theta_x are the hyperparameters of the covariences. 
%       theta_sigma = log (sigma)
%   - covfunc_t: Covariance function on user space
%   - covfunc_x: Covariance function on item space 
%   - t: Users' features
%   - x: Items' features
%   - all_pairs: Cell array of M elements. Each element is a O_m x 2 matrix 
%       where O_m is the number of preferences observed for the corresponding
%       user. Each row all_pairs{m} contains a preference relation 
%       of the form all_pairs{m}(1) > all_pairs{m}(2)     
%   - idx_global: The unique global indices of the observed preferences
%   - idx_global_1: The global indices of the first objects in the preferences
%   - idx_global_2: The gobal indices of the second objects in the preferences
%   - ind_t: Indices of seen tasks
%   - ind_x: Indices of seen items
%   - M: The number of users
%   - N: The number of items
%
% OUTPUT:
%   - nl: The negative marginal log likelihood

% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last update: 22/05/2012


covfunc_t = check_covariance(covfunc_t);
covfunc_x = check_covariance(covfunc_x);

% Get the separate hyper-parameters
[theta_t, theta_x, theta_sigma] = get_gppe_parameters(covfunc_t, covfunc_x, ...
    theta, t, x);
sigma = exp(theta_sigma); 

% Laplace approximation to the posterior p(f | D)
% L is the cholesky decomposition of (W+Kinv)
[fhat Kx, Kinv, W, L] = approx_gppe_laplace_fast(covfunc_t, covfunc_x, theta, t, ...
    x, all_pairs, idx_global, idx_global_1, idx_global_2, ind_t, ind_x, M, N);


%fprintf('sigma = %.3f\n', sigma);

clear theta;
cond_loglike = log_likelihood_gppe(fhat, sigma, all_pairs, idx_global_1, idx_global_2, M, N);

fvis = fhat(idx_global); % f visible

%  -1/2 log det ( Sigma*W + I) - 1/2 f^ Sigma^{-1}f + log p (D | f, theta)
%  log det ( Sigma*W + I) = logdet ( Sigma (W+ Sigma^{-1}) ) 
% = logdet(Sigma) + logdet(W+ Sigma^{-1}) ) 
% = -logdet(Sigma^{-1}) + logdet(W + Sigma^{-1})
% Need to pass LK instead of Kinv
margl = -0.5*( - log(det(Kinv)) + 2*sum(log(diag(L))) ) - 0.5*fvis'*Kinv*fvis ...
    + cond_loglike;

nl = - margl;

return;

 


 