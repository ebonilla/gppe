function [mustar varstar]  = get_predictive_utility_distribution(covfunc_t, ...
    theta_t, f, Kx, Kinv, W, L, t, tstar, idx_xstar, idx_global)
% mustar varstar]  = get_predictive_utility_distribution(covfunc_t, ...
%    theta_t, f, Kx, Kinv, W, L, t, tstar, idx_xstar, idx_global)
%
% Returns the mean and variance of the predictive distribution of a gppe
% model
%
% INPUT:
%   - covfunc_t: Covariance function on user space
%   - theta_t: The parameters of the covariance function on users 
%   - f: The current mode of the posterior 
%   - Kx: The covariance matrix on item space
%   - Kinv: The inverse covariance of the full system
%   - W: The matrix of negative second derivatives (wrt f) of the conditional likelihood
%   - L: chol(W + Kinv)' 
%   - t: Matrix training users' features 
%   - tstar: Test user features 
%   - idx_xstar: Indices of test items 
%   - idx_global: Global indices of observations
%
% OUTPUT:
%   - mustar: The mean of the predictive distribution
%   - varstar: The variance of the predictive distribution
    

% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last update: 22/05/2012

covfunc_t = check_covariance(covfunc_t);

[Kt_ss, Kt_star] = feval(covfunc_t{:}, theta_t, t, tstar);
Kx_star = Kx(idx_xstar,:)';                 % test to training
Kx_star_star = Kx(idx_xstar, idx_xstar);    % test to test

kstar = kron(Kt_star, Kx_star);
kstar = kstar(idx_global,:);
Kss = Kt_ss * Kx_star_star;


mustar = kstar'*Kinv*f(idx_global);
Css    = Kss - kstar'*W*solve_chol(L',Kinv*kstar);  % Kss - Kstar'*(K + W^{-1} )^{-1} * kstar
varstar = diag(Css); % I dont need c

return;