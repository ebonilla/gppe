function toy_example()
% A toy example demonstrating the use of the gppe package
% The most important blocks in the code are elicit_gppe which carries out
% the elicitation process and the blocks corresponding to hyper-parameter
% learning.
% 
% For a new dataset it is necessary to create the corresponding data
% structures (see elicit_gppe) and, more importantly, create new functions 
% for the pointers ptr_query_func and ptr_loss_func. These functions make 
% a query on a test user and compute the corresponding loss of making a
% particular recommendation respectively. See make_query_toydata.m and
% loss_query_toydata.m for more details on these functions.
% 

% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last update: 22/05/2012
%
p = genpath('external'); addpath(p);
p = genpath('scripts');  addpath(p);
p = genpath('utils');  addpath(p);

rand('state',18);
randn('state',20);

%% General Settings ghere
M              = 12;     % Total Number of users (including test users)
N              = 10;     % Number of items
covfunc_t      = 'covSEard';
covfunc_x      = 'covSEard';
D_x            = 3;   % dimensionality of the item features
D_t            = 2;   % Dimesionality of user feaures
sigma          = 0.1; % Noise parameter  
maxiter        = 10;  % Maximum number of queries
ptr_query_func = @make_query_toydata;  % Function that queries the data
ptr_loss_func  = @loss_query_toydata;  % Function that computes the loss 

%% Generate toy data and assign required data structures 
[idx_pairs, train_pairs, theta, train_t, x, test_t, test_idx, ftrue_test, ytrue_test] = ...
    generate_toy_data(M, N, D_t, D_x, covfunc_t, covfunc_x, sigma);
M           = length(train_pairs); % Update number of users to training users
[idx_global_1, idx_global_2] = compute_global_index(train_pairs, N);
idx_global    = unique([idx_global_1; idx_global_2]);
[ind_x ind_t] = ind2sub([N M], idx_global); % idx of seen points and tasks


%% Check gradients of the conditional likelihood wrt f
%check_gradient_f(train_pairs, idx_global_1, idx_global_2, sigma, M, N);
%check_hessian_f(train_pairs, idx_global_1, idx_global_2, sigma, M, N);
 
%% Aproximate posterior
%[f Kx, Kinv, W, L]  = approx_gppe_laplace_fast(covfunc_t, covfunc_x, ...
%theta, train_t, x, train_pairs, idx_global, idx_global_1, idx_global_2, ind_t, ind_x, M, N);

%% make predictions on a new user
%make_predictions_new_user(covfunc_t, covfunc_x, theta, train_t, x, train_pairs, ...
%    idx_global, idx_global_1, idx_global_2, ind_t, ind_x, test_t, idx_pairs, ftrue_test, ytrue_test);


%% Here we test the elicitation stuff
loss = elicit_gppe(covfunc_t, covfunc_x, theta, train_t, x, train_pairs, ...
   test_t, test_idx, idx_pairs,  maxiter, ptr_query_func,   ptr_loss_func);
%plot(1:length(loss), loss); xlabel('Iter'); ylabel('loss');


%% here we check gradients of the marginal log-likelihood
% check_gradients_marginal_loglikelihood(theta, covfunc_t, covfunc_x, ...
%    train_t, x, train_pairs, idx_global, idx_global_1, idx_global_2, ...
%     ind_t, ind_x, M, N);



%% Learn hyper-parameters with scaled conjugate gradient
% theta_learned = learn_gppe_with_netlab(theta, covfunc_t, covfunc_x, ...
%        train_t, x, train_pairs, idx_global, ...
%        idx_global_1, idx_global_2, ind_t, ind_x, M, N);
%theta_learned = theta_learned';    


%% Learn hyper-parameters using Carl Rassmussen's minimize
% theta0 = zeros(size(theta)); %rand(size(theta)); % Initialization is a big issue
% theta0= theta;
% theta_learned = learn_gppe_with_minimize(covfunc_t, covfunc_x, ...
%       theta0, train_t, x, train_pairs, idx_global, ...
%       idx_global_1, idx_global_2, ind_t, ind_x, M, N);
%[theta, theta_learned]


%% Making predictions after learning hyper-parameters
%make_predictions_new_user(covfunc_t, covfunc_x, theta_learned, train_t, x, ...
%     train_pairs, idx_global, idx_global_1, idx_global_2, ind_t, ind_x, ...
%     test_t, idx_pairs, ftrue_test, ytrue_test);

 
 return;
 
 
 



  






















