function check_gradient_f(train_pairs, idx_global_1, idx_global_2, sigma, M, N)
% function check_gradient_f(train_pairs, sigma, M, N)
%
% Checks the gradient of the conditional loglikelihood wrt f
%
% INPUT:
%   - train_pairs: Cell array of M elements. Each element is a O_m x 2 matrix 
%       where O_m is the number of preferences observed for the corresponding
%       user. Each row all_pairs{m} contains a preference relation  
%   - idx_global_1:  The global indices of the first objects in the preferences
%   - idx_global_2:  The gobal indices of the second objects in the preferences
%   - sigma: The noise parameter (scale factor of the preferences)
%   - M: The number of users
%   - N: The number of items

% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last update: 22/05/2012

delta = zeros(M*N, 10);
for j = 1 : 10
    f = rand(M*N,1);
    [gradient, delta(:,j)] = gradchek(f', @log_likelihood_gppe, ...
        @deriv_log_likelihood_gppe_fast_gradchek, sigma, train_pairs, idx_global_1, idx_global_2, M, N);
end
hist(delta(:));
return;


