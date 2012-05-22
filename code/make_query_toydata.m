function pair = make_query_toydata(query_idx, test_idx, test_t)
% pair = make_query_toydata(query_idx, test_idx, test_t)
% Makes a query on the toy dataset. It assumes toydata contains the data 
% for the curent test user. This is the function that queries the "oracle"
% that contains the true preferences for the test users
%
% INPUT:
%   - query_idx: The index of the pairwise relation queried
%   - test_idx: The index of the test user
%   - test_t: The features of the test user
% OUTPUT:
%   pair: The actual preference (pair) indexed by query_idx on the test
%   user
%
% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last update: 22/05/2012

all_pairs = [];
load('toydata.mat', 'all_pairs'); % test_pairs
% pair = test_pairs(query_idx, :);
test_user_pref = all_pairs{test_idx};
pair           = test_user_pref(query_idx,:);

return;



 