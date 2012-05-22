function [idx_global_1, idx_global_2] = compute_global_index(all_pairs, N)
% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last update: 22/05/2012

M = length(all_pairs);
idx_global_1  = [];
idx_global_2  = [];
for j = 1 : M
    pairs = all_pairs{j};
    idx_1 = pairs(:,1);
    idx_2 = pairs(:,2);
    
    idx_global_1 = [idx_global_1; ind2global(idx_1, j, N)];
    idx_global_2 = [idx_global_2; ind2global(idx_2, j, N)];
    %
    %idx_global_1 = [idx_global_1; sub2ind([N M], idx_1, j)];
    %idx_global_2 = [idx_global_2; sub2ind([N M], idx_2, j)];
end

return;






