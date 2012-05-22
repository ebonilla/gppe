function check_gradients_marginal_loglikelihood(theta, covfunc_t, covfunc_x, ...
    train_t, x, train_pairs, idx_global, idx_global_1, idx_global_2, ...
    ind_t, ind_x, Mtrain, N)
% check_gradients_marginal_loglikelihood(theta, covfunc_t, covfunc_x, ...
%    train_t, x, train_pairs, idx_global, idx_global_1, idx_global_2, ...
%    ind_t, ind_x, Mtrain, N)
%
% checks the gradients of marginal loglikelihood wrt hyperparameters
%
% INPUT:
%   - theta:  theta_t; theta_x; theta_sigma]: vector of hyperparameters
%   - covfunc_t: Covariance function on user space
%   - covfunc_x: Covariance function on item space
%   - train_t: Users' features
%   - x: Items' features
%   - train_pairs: ell array of M elements. Each element is a O_m x 2 matrix 
%       where O_m is the number of preferences observed for the corresponding
%       user. Each row all_pairs{m} contains a preference relation 
%       of the form train_pairs{m}(1) > train_pairs{m}(2)      
%   - idx_global: The unique global indices of the observed preferences
%   - idx_global_1: The global indices of the first objects in the preferences
%   - idx_global_2: The gobal indices of the second objects in the preferences
%   - ind_t: Indices of seen tasks
%   - ind_x: Indices of seen items
%   - Mtrain: The number of users
%   - N: The number of items


% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last update: 22/05/2012

ptr_func      = @negative_marginal_log_likelihood;
ptr_gradfunc = @gradient_negative_marginal_loglikelihood; 
delta = zeros(length(theta), 10);
for j = 1 : 10
    theta = rand(size(theta));
    [gradient, delta(:,j)] = gradchek(theta', ptr_func, ...
        ptr_gradfunc, covfunc_t, covfunc_x,  train_t, ...
        x, train_pairs, idx_global,  idx_global_1, idx_global_2, ind_t, ind_x, Mtrain, N);
end
figure;
hist(delta(end,:));
% save('delta.mat', 'delta');




