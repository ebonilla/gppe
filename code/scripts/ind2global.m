function  idx_global = ind2global(idx, task_id, N)
% idx_global = ind2global(idx, task_id, N)
% 
% Computes the global index corresponding to item idx and user task_id
%
% INPUT:
%   - idx: Item's index
%   - task_id: Task index 
%   - N: Number of items
%
% OUTPUT:
%   - idx_global: Global index 

% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last update: 22/05/2012

idx_global = (task_id-1)*N + idx;


return;
