function [nl grad_theta] = learn_gppe(theta, covfunc_t, covfunc_x, t, x, all_pairs, ...
        idx_global, idx_global_1, idx_global_2, ind_t, ind_x, M, N)
% [nl grad_theta] = learn_gppe(theta, covfunc_t, covfunc_x, t, x, all_pairs, ...
%        idx_global, idx_global_1, idx_global_2, ind_t, ind_x, M, N)
%
% Computes the negative marginal likelihood and its gradients wrt to the
% hyper-paramters. This can be used to learn a gppe model with Carl  
% Rasmussen's minimize function
%
% INPUT:
%   - theta = [theta_t; theta_x; theta_sigma]: vector of hyperparameters
%       theta_t and theta_x are the hyperparameters of the covariences. 
%       theta_sigma = log (sigma)
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
%   - nl: The negative marginal log likelihood
%   - grad_theta: The gradients of the negative marginal log likelihood wrt
%       to the hyper-parameters

% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last update: 22/05/2012

global all_diag_idx;
n = M*N;

all_diag_idx = sub2ind([n,n],1 : n, 1 :n);

%% Things we need for computing marginal loglike and derivatives
covfunc_t = check_covariance(covfunc_t);
covfunc_x = check_covariance(covfunc_x);

% Get the separate hyper-parameters
[theta_t, theta_x, theta_sigma] = get_gppe_parameters(covfunc_t, covfunc_x, theta, t, x);
sigma = exp(theta_sigma); 

% Laplace approximation to the posterior p(f | D)
% L is the cholesky decomposition of (W+Kinv)
[fhat Kx, Kinv, W, L] = approx_gppe_laplace_fast(covfunc_t, covfunc_x, theta, t, ...
    x, all_pairs, idx_global, idx_global_1, idx_global_2, ind_t, ind_x, M, N);
%clear theta;

fvis = fhat(idx_global); % f visible


%% Marginal log-likelihood here
cond_loglike = log_likelihood_gppe(fhat, sigma, all_pairs, idx_global_1, idx_global_2, M, N);

%  -1/2 log det ( Sigma*W + I) - 1/2 f^ Sigma^{-1}f + log p (D | f, theta)
%  log det ( Sigma*W + I) = logdet ( Sigma (W+ Sigma^{-1}) ) 
% = logdet(Sigma) + logdet(W+ Sigma^{-1}) ) 
% = -logdet(Sigma^{-1}) + logdet(W + Sigma^{-1})
% Need to pass LK instead of Kinv
margl = -0.5*( - log(det(Kinv)) + 2*sum(log(diag(L))) ) - 0.5*fvis'*Kinv*fvis ...
    + cond_loglike;


% Negative marginal log-likelihood
nl = - margl;

fprintf('nmll=%.2f\n', nl);
%fprintf('theta= %.2f', theta);
%fprintf('\n');
 

%% Things we need for gradients only
dtheta_kt = zeros(length(theta_t),1); 
dtheta_kx = zeros(length(theta_x),1); 

% I may take this from Laplace function?
deriv_loglike_vis = deriv_log_likelihood_gppe_fast(fhat, sigma, all_pairs, ...
    idx_global_1, idx_global_2, M, N);
deriv_loglike_vis = deriv_loglike_vis(idx_global); % At visible location

Kt = feval(covfunc_t{:}, theta_t, t); 


%% lets precompute things
Cinv = solve_chol(L',Kinv); % (KW + I)^{-1}
alpha = fvis'*Kinv;         % f'K^{-1}

%% We compute the explicit derivative theta_t and theta_x
L_thetat = length(theta_t);
dK_dthetat = cell(L_thetat);
for i = 1 : L_thetat
    dKt_dthetat = feval(covfunc_t{:}, theta_t, t, i);  
    dK_dthetat{i} = dKt_dthetat(ind_t, ind_t).*Kx(ind_x,ind_x);

    dtheta_kt(i) = -  0.5*trace(Cinv * dK_dthetat{i} * W) + ...
        0.5*alpha * dK_dthetat{i} * alpha';
    
    
%   fprintf('dK_dthetat[%d]\n',i); 
%   dK_dthetat{i}

%      tmpMat = Cinv * dK_dthetat{i}; % DELETE ME!
%   fprintf('tmpMat=\n');
%   disp(tmpMat);


end
%dtheta_kt

L_thetax = length(theta_x);
dK_dthetax{i} = cell(L_thetax);
for i = 1 : L_thetax
    dKx_dthetax = feval(covfunc_x{:}, theta_x, x, i);  
    dK_dthetax{i} = Kt(ind_t, ind_t).*dKx_dthetax(ind_x,ind_x);
    
    dtheta_kx(i) = -  0.5*trace(Cinv * dK_dthetax{i} * W) + ...
        0.5*alpha * dK_dthetax{i} * alpha';
    
%       fprintf('dK_dthetax[%d]\n',i); 
%          dK_dthetax{i}

end
%dtheta_kx
%
%pause

clear dKt_dthetat; clear dKx_dthetax;


%% explicit derivatives wrt sima
% Need to compute dW_sigma (which is actually dW/dtheta_sigma
[dWdsigma dloglike_dsigma] = get_dsigma(fhat, sigma, all_pairs, M, N);
dWdsigma =  dWdsigma(idx_global, idx_global);   % only at visible locations    
dtheta_sigma = -  0.5*trace( solve_chol(L', dWdsigma) ) + dloglike_dsigma;

%% Stuff needed for implicit derivatives
d_dlogp_dsigma = get_dlogp_dsigma(fhat, sigma, all_pairs, M, N);
dfdsigma_vis =  solve_chol(L', d_dlogp_dsigma(idx_global)); % dfdsigma at "observed" values

 
clear d_dlogp_dsigma;
dtheta_sigma_implicit = 0;


%% Big loop to compute implicit derivarives
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

return;

