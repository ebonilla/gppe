function Deriv2 = deriv2_log_likelihood_gppe_fast(f, sigma, all_pairs, ...
    idx_global_1, idx_global_2, M, N)
% Deriv2 = deriv2_log_likelihood_gppe_fast(f, sigma, all_pairs, ...
%    idx_global_1, idx_global_2, M, N)
%
% Computes the second derivatives of the conditional likelihood wrt f
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
%   Deriv2: The nxn matrix of second derivatives where n=MxN

% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last update: 22/05/2012


% Doing things naively now
% can do things more efficiently (?)
% Note that W is obtaned by taking the negative of Deriv2
% second derivative wrt f


M = length(all_pairs);
n = M*N;
Deriv2 = zeros(n, n);

all_diag_idx = sub2ind([n,n],1 : n, 1 :n);
z = ( f(idx_global_1) - f(idx_global_2) )/sigma;
cdf_val = normcdf(z); pdf_val = normpdf(z);
ratio = pdf_val./cdf_val;
val = - (1/sigma^2) .* ratio .* (z + ratio);

ind = sub2ind([n,n],idx_global_1, idx_global_2);
Deriv2(ind) = - val;
    
ind_trans = sub2ind([n,n],idx_global_2, idx_global_1);
Deriv2(ind_trans) = - val;
    
% Now the diagonal
Deriv2(all_diag_idx) = Deriv2(all_diag_idx) + get_cumulative_val(idx_global_1, val, n)';
Deriv2(all_diag_idx) = Deriv2(all_diag_idx) + get_cumulative_val(idx_global_2, val, n)';



return;

 