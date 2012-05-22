function loglike = log_likelihood_gppe(f, sigma, all_pairs, idx_global_1, idx_global_2, M, N)
% loglike = log_likelihood_gppe(f, sigma, all_pairs, idx_global_1, idx_global_2, M, N)
% 
% Computes the conditional log-likelihood p(D | f,theta) of a gppe model
%
% INPUT:
%   - f: The current value of f  
%   - sigma:  The noise parameter (scale factor of the preferences)
%   - sigma:  The noise parameter (scale factor of the preferences)
%   - all_pairs: Cell array of M elements. Each element is a O_m x 2 matrix 
%       where O_m is the number of preferences observed for the corresponding
%       user. Each row all_pairs{m} contains a preference relation 
%       of the form all_pairs{m}(1) > all_pairs{m}(2)     
%   - idx_global_1: The global indices of the first objects in the preferences
%   - idx_global_2: The gobal indices of the second objects in the preferences
%   - M: The number of users
%   - N: The number of items
%
% OUTPUT:
%   - loglike: The value of the log conditional likelihood

% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last update: 22/05/2012

loglike = 0; 
M = length(all_pairs);
for j = 1 :  M
    if ( isempty(all_pairs{j}) )
        continue;
    end
    
    pairs = all_pairs{j};
    idx_1 = ind2global(pairs(:,1), j, N);
    idx_2 = ind2global(pairs(:,2), j, N);
    
    
    
    z = ( f(idx_1) - f(idx_2) )/sigma;
    
    % disp(z');
    
    cdf_val  = normcdf(z);
    loglike = loglike + sum(log(cdf_val));
end

return;


