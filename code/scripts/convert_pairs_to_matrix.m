function pairs = convert_pairs_to_matrix(cell_pairs)
% pairs = convert_pairs_to_matrix(cell_pairs)
% 
% Converts cell of preference pairs to a matrix
% 
% INPUT:
%   - cell_pairs: Cell array of M elements. Each element is a O_m x 2 matrix 
%       where O_m is the number of preferences observed for the corresponding
%       user. Each row all_pairs{m} contains a preference relation 
%       of the form cell_pairs{m}(1) > cell_pairs{m}(2)     
% OUTPUT:
%   - pairs: nobs x 3 matrix where n is the total number of preferences. The 
%       first column contains the user id and the other two the corresponding 
%       preference relation

% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last update: 22/05/2012

pairs = [];
for i = 1 : length(cell_pairs)
    pairs = [pairs; [repmat(i, size(cell_pairs{i},1), 1), cell_pairs{i}]];
end

