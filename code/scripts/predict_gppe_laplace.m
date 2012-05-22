function [p mustar] = predict_gppe_laplace(covfunc_t, covfunc_x, theta, f,...
    Kx, Kinv, W, L, t, x, idx_global, ind_t, ind_x, tstar, test_pair)
% [p mustar] = predict_gppe_laplace(covfunc_t, covfunc_x, theta, f,...
%    Kx, Kinv, W, L, t, x, idx_global, ind_t, ind_x, tstar, test_pair)
%     
% Computes the probability p(x(test_pair(1),:) > x(test_pair(2),:) for a
% new user tstar using the parameters given by the Laplace method
%
% INPUT:
%   - covfunc_t: Covariance function on user space
%   - covfunc_x: Covariance function on item space
%   - theta = [theta_t; theta_x; theta_sigma]: vector of hyperparameters
%       theta_t and theta_x are the hyperparameters of the covariences. 
%       theta_sigma = log (sigma).
%   - f: The current mode of the posterior 
%   - Kx: The covariance matrix on item space
%   - Kinv: The inverse covariance of the full system
%   - W: The matrix of negative second derivatives (wrt f) of the conditional likelihood
%   - L: chol(W + Kinv)' 
%   - t: Matrix training users' features 
%   - x: Matrix of items' features
%   - idx_global: Global indices of observations
%   - ind_t: Indices of seen tasks
%   - ind_x: Indices of seen items
%   - tstar: Test user features 
%   - test_pair: The test pair preference 
%   
% OUTPUT:
%   - p: p(x(test_pair(1),:) > x(test_pair(2),:)
%   - mustar: The mean of the predictive distribution

% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last update: 22/05/2012


covfunc_t = check_covariance(covfunc_t);
covfunc_x = check_covariance(covfunc_x);

[theta_t, theta_x, theta_sigma] = get_gppe_parameters(covfunc_t, covfunc_x, theta, t, x);
clear theta;
sigma = exp(theta_sigma);


[Kt_ss, Kt_star] = feval(covfunc_t{:}, theta_t, t, tstar);
Kx_star = Kx(test_pair,:)';             % test to training
Kx_star_star = Kx(test_pair, test_pair); % test to test

kstar = kron(Kt_star, Kx_star);
kstar = kstar(idx_global,:);
Kss = Kt_ss * Kx_star_star;


mustar = kstar'*Kinv*f(idx_global);

Css    = Kss - kstar'*W*solve_chol(L',Kinv*kstar);  % Kss - Kstar'*(K + W^{-1} )^{-1} * kstar

sigma_star = sqrt(Css(1,1) + Css(2,2) - 2*Css(1,2) + sigma^2);
val = ( mustar(1) - mustar(2) )/sigma_star;
p   = normcdf(val);

return;


