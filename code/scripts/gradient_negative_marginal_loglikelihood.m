function grad_theta = gradient_negative_marginal_loglikelihood(theta, ...
    covfunc_t, covfunc_x,  t, x, all_pairs, idx_global, idx_global_1, idx_global_2, ...
    ind_t, ind_x, M, N)
% grad_theta = gradient_negative_marginal_loglikelihood(theta, ...
%    covfunc_t, covfunc_x,  t, x, all_pairs, idx_global, idx_global_1, idx_global_2, ...
%    ind_t, ind_x, M, N)
%
% Computes the gradients of the marginal log likelihood wrt all
% hyper-parameters
%
% INPUT:
%   - theta = [theta_t; theta_x; theta_sigma]: vector of hyperparameters
%       theta_t and theta_x are the hyperparameters of the covariences. 
%       theta_sigma = log (sigma).
%   - covfunc_t: Covariance function on user space
%   - covfunc_x: Covariance function on item space 
%   - t: Users' features
%   - x: Items' features
%   - all_pairs: Cell array of M elements. Each element is a O_m x 2 matrix 
%       where O_m is the number of preferences observed for the corresponding
%       user. Each row all_pairs{m} contains a preference relation 
%       of the form all_pairs{m}(1) > all_pairs{m}(2)     
%   - idx_global: The unique global indices of the observed preferences
%   - idx_global_1: The global indices of the first objects in the preferences
%   - idx_global_2: The gobal indices of the second objects in the preferences
%   - ind_t: Indices of seen tasks
%   - ind_x: Indices of seen items
%   - M: The number of users
%   - N: The number of items
%
% OUTPUT:
%   - grad_theta: Vector of gradients wrt theta

% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last update: 22/05/2012


% TODO: consider precomputing (W + Sigma^{-1} )^{-1}
%as I need to solve too many cholesky decompositions!

global all_diag_idx;
n = M*N;
all_diag_idx = sub2ind([n,n],1 : n, 1 :n);


covfunc_t = check_covariance(covfunc_t);
covfunc_x = check_covariance(covfunc_x);

% Get the separate hyper-parameters
[theta_t, theta_x, theta_sigma] = get_gppe_parameters(covfunc_t, ...
    covfunc_x, theta, t, x);
sigma = exp(theta_sigma); 

dtheta_kt = zeros(length(theta_t),1); 
dtheta_kx = zeros(length(theta_x),1); 

% Laplace approximation to the posterior p(f | D)
[fhat Kx, Kinv, W, L] = approx_gppe_laplace_fast(covfunc_t, covfunc_x, theta,...
    t, x, all_pairs, idx_global, idx_global_1, idx_global_2, ind_t, ind_x, M, N);

clear theta; 

deriv_loglike_vis = deriv_log_likelihood_gppe_fast(fhat, sigma, all_pairs, ...
    idx_global_1, idx_global_2, M, N);
deriv_loglike_vis = deriv_loglike_vis(idx_global); % At visible location

Kt = feval(covfunc_t{:}, theta_t, t); 


%% lets precompute things
Cinv = solve_chol(L',Kinv); % (KW + I)^{-1}
fvis = fhat(idx_global);
alpha = fvis'*Kinv;

%% We compute the explicit derivative theta_t and theta_x
L_thetat = length(theta_t);
dK_dthetat = cell(L_thetat);
for i = 1 : L_thetat
    dKt_dthetat = feval(covfunc_t{:}, theta_t, t, i);  
    dK_dthetat{i} = dKt_dthetat(ind_t, ind_t).*Kx(ind_x,ind_x);

    dtheta_kt(i) = -  0.5*trace(Cinv * dK_dthetat{i} * W) + ...
        0.5*alpha * dK_dthetat{i} * alpha';
end
L_thetax = length(theta_x);
dK_dthetax{i} = cell(L_thetax);
for i = 1 : L_thetax
    dKx_dthetax = feval(covfunc_x{:}, theta_x, x, i);  
    dK_dthetax{i} = Kt(ind_t, ind_t).*dKx_dthetax(ind_x,ind_x);
    
    dtheta_kx(i) = -  0.5*trace(Cinv * dK_dthetax{i} * W) + ...
        0.5*alpha * dK_dthetax{i} * alpha';
end
clear dKt_dthetat; clear dKx_dthetax;



%% explicite derivatives wrt sima
% Need to compute dW_sigma (which is actually dW/dtheta_sigma
[dWdsigma dloglike_dsigma] = get_dsigma(fhat, sigma, all_pairs, M, N);
dWdsigma =  dWdsigma(idx_global, idx_global);   % only at visible locations    
dtheta_sigma = -  0.5*trace( solve_chol(L', dWdsigma) ) + dloglike_dsigma;

%% Compute implicit derivatives here
d_dlogp_dsigma = get_dlogp_dsigma(fhat, sigma, all_pairs, M, N);
dfdsigma_vis =  solve_chol(L', d_dlogp_dsigma(idx_global)); % dfdsigma at "observed" values
clear d_dlogp_dsigma;
dtheta_sigma_implicit = 0;
for i = 1 : length(idx_global)
    ptr_ind_t = ind_t(i); ptr_ind_x = ind_x(i); % indices of current f_{o}
    pairs = all_pairs{ptr_ind_t};

    dWdf   =  get_dWdf(fhat, ptr_ind_t, ptr_ind_x, sigma, pairs, M, N);  
    dWdf   = dWdf(idx_global, idx_global);
    tmp_val = -0.5*trace( solve_chol(L', dWdf) );
    
    val = tmp_val*dfdsigma_vis(i); %  tmp_val*dF_dsigma_{i} as dfdsigma only contains "observed values"
    
    dtheta_sigma_implicit = dtheta_sigma_implicit + val; 
    
    for k = 1 : length(theta_t)
        df_dtheta =  Cinv*dK_dthetat{k}*deriv_loglike_vis;
        dtheta_kt(k)  = dtheta_kt(k)  + tmp_val*df_dtheta(i);  % as df_dtheta only contain info at observed values
    end
    
    for k = 1 : length(theta_x)
        df_dtheta =  Cinv*dK_dthetax{k}*deriv_loglike_vis;
        dtheta_kx(k)  = dtheta_kx(k)  + tmp_val*df_dtheta(i);
    end
    
end  
dtheta_sigma = dtheta_sigma + dtheta_sigma_implicit;
grad_theta = - [ dtheta_kt; dtheta_kx; dtheta_sigma];

% DELETE ME
grad_theta = grad_theta';
return;








%%