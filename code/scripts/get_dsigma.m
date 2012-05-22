function [dWdsigma dloglike_dsigma]  =  get_dsigma(f, sigma, all_pairs, M, N)
% [dWdsigma dloglike_dsigma]  =  get_dsigma(f, sigma, all_pairs, M, N)
%
% Computes derivatives wrt sigma appearing in the explicit derivatives 
% of the marginal likelihood. 
% 
% INPUT: 
%   - f: The current value of f  (mode) 
%   - sigma:  The noise parameter (scale factor of the preferences)
%   - all_pairs: ell array of M elements. Each element is a O_m x 2 matrix 
%       where O_m is the number of preferences observed for the corresponding
%       user. Each row all_pairs{m} contains a preference relation 
%       of the form all_pairs{m}(1) > all_pairs{m}(2)   
%   - M: The number of users
%   - N: The number of items
% 
% OUTPUT:
%   - dWdsigma: Derivatives of the matrix W wrt dtheta_sigma 
%       with theta_sigma = log (sigma)
%   - dloglike_dsigma: Derivative of the conditional likelihood wrt
%       theta_sigma           

% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last update: 22/05/2012

 
M = length(all_pairs);
n = M*N;
dWdsigma = zeros(n, n);

all_diag_idx = sub2ind([n,n],1 : n, 1 :n);

dloglike_dsigma = 0;

for j = 1 :  M
    if ( isempty(all_pairs{j}) )
        continue;
    end
    
    pairs = all_pairs{j};
    idx_1 = pairs(:,1);
    idx_2 = pairs(:,2);
    
    idx_global_1 = ind2global(idx_1, j, N);
    idx_global_2 = ind2global(idx_2, j, N);
    z = ( f(idx_global_1) - f(idx_global_2) )/sigma;
 
    pdf_val = normpdf(z);
    cdf_val = normcdf(z);
    
    ratio1   = pdf_val./cdf_val;
    %ratio2   = (z.*cdf_val + pdf_val ) ./ (cdf_val.^2);
    %val      = - 2*ratio1 .* (z + ratio1) ...
    %           + z.*pdf_val.*(ratio2).*(z+ratio1) ...
    %           + ratio1.*(-z + z.*pdf_val.*(ratio2));
    %val      = val*(1/sigma^2);
    %
    %
    val = (-1/(sigma^2)) *( z .* ratio1.* ( 1 - (z + ratio1).*(z + 2.*ratio1) ) ...
         + 2 * ratio1.*(z + ratio1) ) ;
    
    ind = sub2ind([n,n],idx_global_1, idx_global_2); 
    dWdsigma(ind) = - val;
    
    ind_trans = sub2ind([n,n],idx_global_2, idx_global_1);
    dWdsigma(ind_trans) = - val;
    
    % Now the diagonal
    dWdsigma(all_diag_idx) = dWdsigma(all_diag_idx) + ...
        get_cumulative_val2(idx_global_1, val, n);
    dWdsigma(all_diag_idx) = dWdsigma(all_diag_idx) + ...
        get_cumulative_val2(idx_global_2, val, n);
    
    
    dloglike_dsigma = dloglike_dsigma - sum(z.*ratio1);
end

return;


