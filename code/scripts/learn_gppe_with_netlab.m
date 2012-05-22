function theta_learned = learn_gppe_with_netlab(theta, covfunc_t, covfunc_x, ...
    train_t, x, train_pairs, idx_global, idx_global_1, idx_global_2, ...
    ind_t, ind_x, Mtrain, N)
% theta_learned = learn_gppe_with_netlab(theta, covfunc_t, covfunc_x, ...
%    train_t, x, train_pairs, idx_global, idx_global_1, idx_global_2, ...
%    ind_t, ind_x, Mtrain, N)
% 
% Learns a gppe model with Netlab's scaled conjugate gradient
%
% INPUT:
%   - theta = [theta_t; theta_x; theta_sigma]: vector of hyperparameters
%       theta_t and theta_x are the hyperparameters of the covariences. 
%       theta_sigma = log (sigma)
%   - covfunc_t: Covariance function on user space
%   - covfunc_x: Covariance function on item space 
%   - train_t: Training Users' features
%   - x: Items' features
%   - train_pairs: Cell array of M elements. Each element is a O_m x 2 matrix 
%       where O_m is the number of preferences observed for the corresponding
%       user. Each row all_pairs{m} contains a preference relation 
%       of the form train_pairs{m}(1) > train_pairs{m}(2)     
%   - idx_global: The unique global indices of the observed preferences
%   - idx_global_1: The global indices of the first objects in the preferences
%   - idx_global_2: The gobal indices of the second objects in the preferences
%   - ind_t: Indices of seen tasks
%   - ind_x: Indices of seen items
%   - Mtrain: The number of training users
%   - N: The number of items
%
% OUTPUT:
%   - theta_learned: The learned hyper-parameters

% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last update: 22/05/2012

PRECISION = 1e-4;
MAXITER   = 100;
options = foptions();
options(1)  = 1; % Shows error values
options(2)  = PRECISION;
options(3)  = PRECISION;
options(14) = MAXITER;
options(15) = PRECISION;

ptr_func      = @negative_marginal_log_likelihood;
ptr_gradfunc = @gradient_negative_marginal_loglikelihood; 
[theta_learned, options, flog] = scg(ptr_func, theta', options, ...
    ptr_gradfunc, covfunc_t, covfunc_x, train_t, x, train_pairs, ...
    idx_global, idx_global_1, idx_global_2,  ind_t, ind_x, Mtrain, N);

% save('tmp.mat', 'theta_learned', 'options', 'flog');
return;


