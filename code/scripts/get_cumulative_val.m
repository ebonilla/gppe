function count = get_cumulative_val(idx, val, n)
% count = get_cumulative_val(idx, val, n)
%
% Accumulates the values in val according to the indices given by idx


% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last update: 22/05/2012

count = zeros(n,1);
for i = 1 : length(val)
    count(idx(i)) = count(idx(i)) + val(i);
end

return;


