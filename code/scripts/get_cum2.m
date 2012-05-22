function count = get_cum2(idx, val, n)
% count = get_cum2(idx, val, n)
%
% Accumulates the values in val according to the indices given by idx

% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last update: 22/05/2012

count = zeros(1,n);
for i = 1 : length(val)
    count(idx(i)) = count(idx(i)) + val(i);

end


return;

