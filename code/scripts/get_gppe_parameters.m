function [theta_t, theta_x, theta_sigma] = get_gppe_parameters(covfunc_t, covfunc_x, theta, t, x)
% [theta_t, theta_x, theta_sigma] = get_gppe_parameters(covfunc_t, covfunc_x, theta, t, x)
%
% Returns the subsets of parameters for a GPPE model
%
% INPUT:
%   - covfunc_t: Covariance function on user space 
%   - covfunc_x: Covariance function on item space 
%   - theta: [theta_t; theta_x; theta_sigma]: vector of hyperparameters 
%   - t: The matrix of users' features (including the test user) 
%   - x: The matrix of item features 
%
% OUTPUT:
%   - theta_t: Parameters of the covariance on users
%   - theta_x: Parameters of the covariance on items 
%   - theta_sigma: The noise parameter (scale factor of the preferences)

% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last update: 22/05/2012

D = size(t,2);   ntheta_t = eval(feval(covfunc_t{:})); % D_t  = D;% number of cov parameteres
D = size(x,2);   ntheta_x = eval(feval(covfunc_x{:})); % D_x = D; clear D;

theta_t = theta(1:ntheta_t);
theta_x = theta(ntheta_t+1:ntheta_t+ntheta_x);
theta_sigma = theta(end); 

return;





