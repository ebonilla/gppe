function fstar =  make_predictions_new_user(covfunc_t, covfunc_x, theta, train_t, x, ...
    train_pairs, idx_global, idx_global_1, idx_global_2, ind_t, ind_x, test_t, idx_pairs, ftrue, ytrue)
%  fstar =  make_predictions_new_user(covfunc_t, covfunc_x, theta, train_t, x, ...
%     train_pairs, idx_global, idx_global_1, idx_global_2, ind_t, ind_x, test_t, idx_pairs, ftrue, ytrue)
% Makes predictions on a new user
% INPUT:
%   - covfunc_t: Covariance function on user space
%   - covfunc_x: Covariance function on item space
%   - theta = [theta_t; theta_x; theta_sigma]: vector of hyperparameters
%       theta_t and theta_x are the hyperparameters of the covariences. 
%       theta_sigma = log (sigma).
%   - train_t: Users' features
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
%   - test_t: Test user features 
%   - idx_pairs: The matrix of all pairwise item comparisons:
%           idx_pairs(i,1) > idx_pairs(i,2) 
%   - ftrue: The true value of utility function of test user (used for
%       evaluation)
%   - ytrue: Binary vector indicating if the corresponding 
%       item comparisons hold for the test user (used for evaluation)
% 
% OUTPUT:
%   - fstar: the test users' predicted utility at all items

% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last update: 22/05/2012

N      = size(x,1);
Mtrain = size(train_t,1);

% get the f values first
[f Kx, Kinv, W, L]  = approx_gppe_laplace_fast(covfunc_t, covfunc_x, ...
    theta, train_t, x, train_pairs, idx_global, idx_global_1, idx_global_2, ind_t, ind_x, Mtrain, N);

Npairs = size(idx_pairs,1);
Fstar  = NaN(N,Npairs);
P      = zeros(Npairs,1);
for i = 1 : Npairs
    pair = idx_pairs(i,:);
    [p mustar] = predict_gppe_laplace(covfunc_t, covfunc_x, theta, f, Kx, ...
        Kinv, W, L, train_t, x, idx_global, ind_t, ind_x, test_t, pair);
    P(i,1) = p;    
    Fstar([pair(1), pair(2)],i) = mustar;
end
fstar = mynanmean(Fstar, 2);

% P is the preditive probabilities of the pair being a > relationship
ypred = P > 0.5;
fprintf('error=%.2f\n', sum(ytrue ~= ypred,1)/size(ytrue,1));


% ystar = ( fstar(idx_pairs(:,1)) - fstar(idx_pairs(:,2)) ) > EPSILON; 


% Plotting the underlying utility functions
plot(ftrue, 'b'); hold on; plot(fstar, 'r');
legend({'True Utility', 'Predicted Utility'});

% [val, idx_true] = sort(ftrue, 'descend');
% [val, idx_pred] = sort(fstar, 'descend');

return;