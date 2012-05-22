function mei = maximum_expected_improvement_gppe(covfunc_t, covfunc_x, theta, f, Kx, ...
    Kinv, W, L, t, x, idx_global, ind_t, ind_x, tstar, idx_xstar, fbest)
%  mei = maximum_expected_improvement_gppe(covfunc_t, covfunc_x, theta, f, Kx, ...
%    Kinv, W, L, t, x, idx_global, ind_t, ind_x, tstar, idx_xstar, fbest)
%
% Computes the maximum expected improvement (MEI) of recommending items
% given by indices idx_xstar on user with features tstar.
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
%   - tstar: Features of test user 
%   - idx_xstar: Indices of test items that are to be considered
%   - fbest:  An estimate of the best utility function value
%
% OUTPUT:
%   - mei:  Maximum expected improvement for test items

% Adding the case of a new item is straightforward as I just need to use
% the new features xstar and compute the kernel kx using covfunc_x


% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last update: 22/05/2012

covfunc_t = check_covariance(covfunc_t);
covfunc_x = check_covariance(covfunc_x);

[theta_t, theta_x, theta_sigma] = get_gppe_parameters(covfunc_t, covfunc_x, theta, t, x);
clear theta;
sigma = exp(theta_sigma);


%% here we compute the expected utilities (and variances) of all items!
[Kt_ss, Kt_star] = feval(covfunc_t{:}, theta_t, t, tstar);
Kx_star = Kx(idx_xstar,:)';                 % test to training
Kx_star_star = Kx(idx_xstar, idx_xstar);    % test to test

kstar = kron(Kt_star, Kx_star);
kstar = kstar(idx_global,:);
Kss = Kt_ss * Kx_star_star;


mustar = kstar'*Kinv*f(idx_global);
Css    = Kss - kstar'*W*solve_chol(L',Kinv*kstar);  % Kss - Kstar'*(K + W^{-1} )^{-1} * kstar
varstar = diag(Css); % I dont need covariances, may consider to do things more efficiently


%%
sigmastar = sqrt(varstar);
z = (fbest - mustar)./sigmastar;
pdfval = normpdf(z);
cdfval = normcdf(z);
el = sigmastar .* ( z.*(1-cdfval) - pdfval );



mei = max( - el);


%figure;plot_confidence_interval(idx_xstar, mustar , sigmastar, 1);

