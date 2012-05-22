function deriv_loglike = deriv_log_likelihood_gppe_fast(f, sigma, all_pairs, ...
    idx_global_1, idx_global_2, M, N)
% deriv_loglike = deriv_log_likelihood_gppe_fast(f, sigma, all_pairs, ...
%    idx_global_1, idx_global_2, M, N)
% 
% Computes the  derivatives of the conditional likelihood wrt f
%
% INPUT:
%   - f: The current value of f  
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
%   - deriv_loglike: vector of derivatives
% 
% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last update: 22/05/2012



% Doing things naively now
% can do things more efficiently (?)
% derivative wrt f


if (isempty(idx_global_1) || isempty(idx_global_2))
    error('GPPE:deriv_log_likelihood_gppe_fast', 'idx_global_1 and idx_global_2 Undefined');
end

M = length(all_pairs);
n = M*N;
z = ( f(idx_global_1) - f(idx_global_2) )/sigma;
cdf_val = normcdf(z); pdf_val = normpdf(z);    
val = (1/sigma) * pdf_val./cdf_val;
coef = get_cumulative_val(idx_global_1, val, n);
coef = coef - get_cumulative_val(idx_global_2, val, n);
deriv_loglike = coef;


return;

 