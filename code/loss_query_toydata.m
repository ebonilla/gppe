function [loss stop] = loss_query_toydata(test_user_idx, best_item_idx)
% [loss stop] = loss_query_toydata(test_user_idx, best_item_idx)
% Computes the loss of recommending item best_item_idx on user test_user_idx 
% on the toy dataset. This is the "oracle" function that looks at the true
% preferences on all the users. It used by gppe for evaluation purposes
% 
% INPUT:
%   - test_user_idx: Index of test user
%   - best_item_idx: Index of recommended item  
% OUTPUT:
%   - loss: The actual loss incurred by recommended best_item_idx instead 
%       of the actual best item
%   - stop: a binary flag indicating the the best recommendation has been
%       done (not used by gppe at the moment)

% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last update: 22/05/2012

F = [];
load('toydata.mat', 'F');

ftest = F(:, test_user_idx);
best_val = max(ftest);
pred_val = ftest(best_item_idx);

loss = best_val - pred_val;

if (pred_val == best_val)
    stop = true;
else
    stop = false;
end

return;







