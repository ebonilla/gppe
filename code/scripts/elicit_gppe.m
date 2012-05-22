function loss = elicit_gppe(covfunc_t, covfunc_x, theta, train_t, x, ...
    train_pairs, test_t, test_user_idx, idx_pairs, Maxiter, ...
    ptr_query_func, ptr_loss_func)
% loss = elicit_gppe(covfunc_t, covfunc_x, theta, train_t, x, ...
%     train_pairs, test_t, test_user_idx, idx_pairs, Maxiter, ...
%     ptr_query_func, ptr_loss_func)
% 
% Elicit preferences for a new user. It assumes that that there is an
%   "oracle" given by the function pointed by ptr_query_func that knows
%   the actual preference of the new user
% 
% INPUT:
%   - covfunc_t: Covariance function on user space 
%   - covfunc_x: Covariance function on item space 
%   - theta: [theta_t; theta_x; theta_sigma]: vector of hyperparameters 
%        theta_t and theta_x are the hyperparameters of the covariances. 
%       theta_sigma = log (sigma).
%   - train_t:  Training users' features
%   - x: Items' features
%   - train_pairs: cell array of M elements. Each element is a O_m x 2 matrix 
%       where O_m is the number of preferences observed for the corresponding
%       user. Each row all_pairs{m} contains a preference relation 
%       of the form train_pairs{m}(1) > train_pairs{m}(2)
%   - test_t: Test user's features 
%   - test_user_idx: Index of test user in "oracle"  
%   idx_pairs:  All possible pairs to evaluate on test user
%   Maxiter: Maximum number of iterations (queries) to perform
%   ptr_query_func: Pointer to query function ("oracle") 
%   ptr_loss_func: Pointer to loss function
% 
% OUTPUT:
%   - loss: Vector of losses (one elemement per query)

% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last update: 22/05/2012

covfunc_t = check_covariance(covfunc_t);
covfunc_x = check_covariance(covfunc_x);
[theta_t, theta_x, theta_sigma] = get_gppe_parameters(covfunc_t, covfunc_x, theta, train_t, x);

N = size(x,1);
M = size(train_t,1);
[idx_global_1, idx_global_2] = compute_global_index(train_pairs, N);
idx_global = unique([idx_global_1; idx_global_2]);
[ind_x ind_t] = ind2sub([N M], idx_global); % idx of seen points and tasks

% we augment the dataset with the test user
Mtrain = size(train_t,1);
N = size(x,1);

M = Mtrain + 1;
t = [train_t; test_t];
train_pairs{M} = [];
Npairs = size(idx_pairs,1);
evoi = zeros(Npairs,1);

stop = false;
count = 0;
is_selected = zeros(Npairs,1);

% We use one iteration more 
loss = zeros(Maxiter+1,1);
for iter = 1 : Maxiter+1
     
    % Laplace approximation
    [f Kx, Kinv, W, L] = approx_gppe_laplace_fast(covfunc_t, covfunc_x, ...
        theta, t, x, train_pairs, idx_global, idx_global_1, idx_global_2, ...
        ind_t, ind_x, M, N);
    
    

    [mustar varstar]  = get_predictive_utility_distribution(covfunc_t, ...
        theta_t, f, Kx, Kinv, W, L, t, test_t, 1:N, idx_global);
 
    [foo best_item_idx] = max(mustar);
    
    
    fbest = get_fbest(mustar, f, L, N); % Gets fbest
    
    for i = 1 : Npairs % goes trough all the candidadate pairs
        if (is_selected(i) )
            evoi(i) = -inf; 
            continue;
        end
        test_pair = idx_pairs(i,:);
        evoi(i)   =   expected_voi_gppe(covfunc_t, covfunc_x, theta, f, Kx, Kinv, ...
            W, L, t, x, train_pairs, idx_global, ind_t, ind_x, test_pair, fbest);
    end
     
    
    % Here we compute the pair to query
    % TODO: Need to resolve ties at random!
    [val query_idx]  = max(evoi);
    idx_good = find(evoi == val);
    Lgood = length(idx_good); 
    if ( Lgood > 1) 
        vrand = randperm(Lgood);
        fprintf('Solving clashes at random\n');
        query_idx = idx_good(vrand(1));
    end
     
    is_selected(query_idx) = 1;  % avoid considering any further
    
    new_pair  = feval(ptr_query_func, query_idx, test_user_idx);
    
    % We add the new pair to the current data
    train_pairs{M} = [train_pairs{M}; new_pair];
    [idx_global_1, idx_global_2] = compute_global_index(train_pairs, N);
    idx_global = unique([idx_global_1; idx_global_2]);
    [ind_x ind_t] = ind2sub([N M], idx_global); % indices of "seen" data-points and tasks

    
    % Computes the loss of making a recommendation at this point
    loss(iter)  =  feval(ptr_loss_func, test_user_idx, best_item_idx);


    count = count + 1;
    fprintf('Query%d=[%d %d] done, Recommended Item=%d, loss=%d \n', ...
        count, new_pair(1), new_pair(2), best_item_idx, loss(iter));

end
return;


%%
% To compute the expected loss we first get the expected best-so-far
% fbest  = max(mustar);
%
function fbest = get_fbest(mustar, f, L,  N)
% fbest = max(mustar);

ftest = f(end-N+1: end);
fbest =  max (ftest);

% fprintf('fbest=%.2f', fbest);
% fprintf('\n');

% We can sample from the Gaussian instead
% nsamples = 1;
% fmean = f(end-N+1: end);
% fmean = fmean(~isnan(fmean));
% if (~isempty(fmean))
%    D = size(fmean,1);
%    z = randn(D,nsamples);
%    fsample = repmat(fmean, 1, nsamples) + L(end-D+1:end,end-D+1:end)'*z;
%    fbest = max(fsample);
%end

if ( isnan(fbest) ) % I don't have estimates of ftest yet: Cold start
    fbest  = max(f);
end
%

% fprintf('fbest=%.2f\n',fbest);




