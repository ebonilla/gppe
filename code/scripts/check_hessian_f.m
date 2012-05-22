function check_hessian_f(train_pairs, idx_global_1, idx_global_2, sigma, M, N)
% check_hessian_f(train_pairs, idx_global_1, idx_global_2, sigma, M, N)
%
% Checks the hessian of the conditional log likelihood 
%
% INPUT:
%   - train_pairs: Cell array of M elements. Each element is a O_m x 2 matrix 
%       where O_m is the number of preferences observed for the corresponding
%       user. Each row all_pairs{m} contains a preference relation 
%       of the form train_pairs{m}(1) > train_pairs{m}(2)     
%   - idx_global_1: The global indices of the first objects in the preferences
%   - idx_global_2: The gobal indices of the second objects in the preferences
%   - sigma: The noise parameter (scale factor of the preferences)
%   - M: The number of users
%   - N: The number of items


% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last update: 22/05/2012

delta = zeros(M*N*M*N,10);
for j = 1 : 10
    f = rand(M*N,1);
    [h hcent delta(:,j)] = myhesschek(f, @log_likelihood_gppe, ...
        @deriv2_log_likelihood_gppe_fast, sigma, train_pairs, idx_global_1, idx_global_2, M, N);
end
hist(delta(:));

return;
