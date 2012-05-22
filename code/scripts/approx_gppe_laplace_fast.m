function [f Kx, Kinv, W, L] = approx_gppe_laplace_fast(covfunc_t, covfunc_x,...
    theta, t, x, all_pairs, idx_global, idx_global_1, idx_global_2, ind_t, ind_x, M, N)
% [f Kx, Kinv, W, L] = approx_gppe_laplace_fast(covfunc_t, covfunc_x,...
%     theta, t, x, all_pairs, idx_global, idx_global_1, idx_global_2, ind_t, ind_x, M, N)
%
% Approximates the posterior distribution of the gppe model with the Laplace method
%
% INPUT:
%   - covfunc_t: Covariance function on user space
%   - covfunc_x: Covariance function on item space
%   - theta = [theta_t; theta_x; theta_sigma]: vector of hyperparameters
%       theta_t and theta_x are the hyperparameters of the covariences. 
%       theta_sigma = log (sigma).
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
%   - f: The mode of the posterior
%   - Kx: The covariance matrix on item space
%   - Kinv: The inverse covariance of the full system
%   - W: The matrix of negative second derivatives (wrt f) of the conditional likelihood
%   - L: chol(W + Kinv)' 

% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last update: 22/05/2012
%

tol = 1e-6;                   % tolerance for when to stop the Newton iterations
persistent fvis;
covfunc_t = check_covariance(covfunc_t);
covfunc_x = check_covariance(covfunc_x);

[theta_t, theta_x, theta_sigma] = get_gppe_parameters(covfunc_t, covfunc_x, theta, t, x);
sigma = exp(theta_sigma); 
clear theta;

n = M*N; % need to change this when not everthing is fully observerd

Kt = feval(covfunc_t{:}, theta_t, t); 
Kx = feval(covfunc_x{:}, theta_x, x);  

%Kold  = kron(Kt, Kx); Kold  = Kold(idx_global, idx_global);
K = Kt(ind_t, ind_t).*Kx(ind_x,ind_x);



%% We do Newton method here psi is the  approximate log posterior  
%if (isempty(all_pairs{M}))
%
f = zeros(n,1);
%
%f = NaN*ones(n,1);
f(idx_global) = 0;
%    clear fvis;
%else
%    f(idx_global) = fvis; % we initialize with previous value
%end
fvis = f(idx_global); % f visible

loglike = log_likelihood_gppe(f, sigma, all_pairs, idx_global_1, idx_global_2, M, N);
Kinv = inv(K);                                             % Need to do numerically efficient/stable later
psi_new = loglike - 0.5 * fvis'*Kinv*fvis; 
psi_old = -Inf;                                            % make sure while loop starts
%fprintf('psi_new = %.6f\n', psi_new);
while ( (psi_new - psi_old) > tol  )                       % begin Newton's iterations
  psi_old = psi_new;   
  deriv =   deriv_log_likelihood_gppe_fast(f, sigma, all_pairs, ...
      idx_global_1, idx_global_2, M, N);
  W     = - deriv2_log_likelihood_gppe_fast(f, sigma, all_pairs, ...
      idx_global_1, idx_global_2, M, N);
  
  W     = W(idx_global, idx_global);   % only at visible locations
  L = chol(W + Kinv)';
  fvis  = solve_chol(L', deriv(idx_global) + W*fvis);        % (W + Kinv)^-1 (deriv + Wf)
  f(idx_global) = fvis;                                      % updating visible values  
  loglike = log_likelihood_gppe(f, sigma, all_pairs, idx_global_1, idx_global_2, M, N);
  
  psi_new = loglike - 0.5 * fvis'*Kinv*fvis;
  
 %fprintf('psi_new = %.6f\n', psi_new);
end        


return;
  

