function dWdf   =  get_dWdf(f, ind_t, ind_x, sigma, pairs, M, N) 
% dWdf   =  get_dWdf(f, f_idx_global, sigma, all_pairs, M, N)
% 
% Derivatives of the matrix W wrt a  single component in f. This is 
%       useful in computing the implicit derivatives of the marginal
%       likelihood (hyper-parameter learning).
%
% INPUT:
%   - f:  The current vector  f  (mode)   
%   - ind_t:  scalar index of the current user
%   - ind_x:  scalar index of the current item
%   - sigma:  The noise parameter (scale factor of the preferences)
%   - pairs:  Matrix of preference pairs for user indexed by ind_t 

% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last update: 22/05/2012

global all_diag_idx;

n = M*N;
dWdf = zeros(n, n);


% We first find the user and the data-point correspnding to this index

idx_1 = pairs(:,1);
idx_2 = pairs(:,2);

% We simply match those corresponding to the required f
idx_select = find(idx_1==ind_x | idx_2==ind_x);

if (isempty(idx_select))
    return;
end
idx_1  =  idx_1(idx_select);
idx_2  =  idx_2(idx_select);

coeff = ones(size(idx_1));
coeff((idx_2==ind_x)) = -1; %it is negative if f_{o} is on the wrong side of the relationship

idx_global_1 = ind2global(idx_1, ind_t, N);
idx_global_2 = ind2global(idx_2, ind_t, N);


z = ( f(idx_global_1) - f(idx_global_2) )/sigma; 
pdf_val = normpdf(z);
cdf_val = normcdf(z);
    
ratio1   = pdf_val./cdf_val;
val = (1/sigma^3) .* ratio1.* ( 1 - (z + ratio1).*(z + 2.*ratio1) );

val = val.*coeff;
    
%ind = sub2ind([n,n],idx_global_1, idx_global_2)
ind = ind2global(idx_global_1, idx_global_2, n);
dWdf(ind) = - val;
    
%ind_trans = sub2ind([n,n],idx_global_2, idx_global_1);
ind_trans = ind2global(idx_global_2, idx_global_1, n);

dWdf(ind_trans) = - val;
    

% Now the diagonal
dWdf(all_diag_idx) = dWdf(all_diag_idx) + get_cum2(idx_global_1, val, n);
dWdf(all_diag_idx) = dWdf(all_diag_idx) + get_cum2(idx_global_2, val, n);

     

return;

