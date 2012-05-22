function [out1, out2, out3, out4] = binaryEPGP(logtheta, covfunc, x, y, xstar);

% binaryEPGP - The Expectation Propagation approximation for binary Gaussian
% process classification. Two modes are possible: training or testing: if no
% test cases are supplied, then the approximate negative log marginal
% likelihood and its partial derivatives wrt the hyperparameters is computed;
% this mode is used to fit the hyperparameters. If test cases are given, then
% the test set predictive probabilities are returned. The program is flexible
% in allowing a multitude of covariance functions.
%
% usage: [nlml dnlml] = binaryEPGP(logtheta, covfunc, x, y);
%    or: [p mu s2 nlml] = binaryEPGP(logtheta, covfunc, x, y, xstar);
%
% where:
%
%   logtheta is a (column) vector of hyperparameters
%   covfunc  is the name of the covariance function (see below)
%   x        is a n by D matrix of training inputs
%   y        is a (column) vector (of size n) of binary +1/-1 targets
%   xstar    is a nn by D matrix of test inputs
%   nlml     is the returned value of the negative log marginal likelihood
%   dnlml    is a (column) vector of partial derivatives of the negative
%               log marginal likelihood wrt each log hyperparameter
%   p        is a (column) vector (of length nn) of predictive probabilities
%   mu       is a (column) vector (of length nn) of predictive latent means
%   s2       is a (column) vector (of length nn) of predictive latent variances
%
% The length of the vector of log hyperparameters depends on the covariance
% function, as specified by the "covfunc" input to the function, specifying the
% name of a covariance function. A number of different covariance function are
% implemented, and it is not difficult to add new ones. See "help covFunctions"
% for the details
%
% The function can conveniently be used with the "minimize" function to train
% a Gaussian process:
%
% [logtheta, fX, i] = minimize(logtheta, 'binaryLaplaceGP', length, 'covSEiso',
%                              x, y);
%
% Copyright (c) 2004, 2005, 2006 Carl Edward Rasmussen, 2006-03-20.

if ischar(covfunc), covfunc = cellstr(covfunc); end % convert to cell if needed
[n, D] = size(x);
if eval(feval(covfunc{:})) ~= size(logtheta, 1)
  error('Error: Number of parameters do not agree with covariance function')
end
persistent best_ttau best_tnu best_lml;   % keep tilde parameters between calls

tol = 1e-3; max_sweep = 10;          % tolerance for when to stop EP iterations
sigmoid = inline('(1+erf(x/sqrt(2)))/2');
gos = inline('sqrt(2/pi)*exp(-x.^2/2)./(1+erf(x/sqrt(2)))');  % gauss / sigmoid

K = feval(covfunc{:}, logtheta, x);                     % the covariance matrix

% A note on naming: variables are given short but descriptive names in 
% accordance with Rasmussen & Williams "GPs for Machine Learning" (2006): mu
% and s2 are mean and variance, nu and tau are natural parameters. A leading t
% means tilde, a subscript _ni means "not i" (for cavity parameters), or _n
% for a vector of cavity parameters.

if any(size(best_ttau) ~= [n 1])     % find starting point for tilde parameters
  ttau = zeros(n,1);            % initialize to zero if we have no better guess
  tnu = zeros(n,1);
  Sigma = K;                    % initialize Sigma and mu, the parameters of ..
  mu = zeros(n, 1);                   % .. the Gaussian posterior approximation
  lml = -n*log(2);
  best_lml = -inf;
else
  ttau = best_ttau;                   % try the tilde values from previous call
  tnu = best_tnu;
  [Sigma, mu, lml] = epComputeParams(K, y, ttau, tnu); 
  if lml < -n*log(2)                                     % if zero is better ..
    ttau = zeros(n,1);                   % .. then initialize with zero instead
    tnu = zeros(n,1); 
    Sigma = K;                  % initialize Sigma and mu, the parameters of ..
    mu = zeros(n, 1);                 % .. the Gaussian posterior approximation
  end
end
lml_old = -inf; sweep = 0;                        % make sure while loop starts

while lml - lml_old > tol & sweep < max_sweep    % converged or maximum sweeps?

  lml_old = lml; sweep = sweep + 1;
  for i = 1:n                                % iterate EP updates over examples

    tau_ni = 1/Sigma(i,i)-ttau(i);      % first find the cavity distribution ..
    nu_ni = mu(i)/Sigma(i,i)-tnu(i);           % .. parameters tau_ni and nu_ni

    z_i = y(i)*nu_ni/sqrt(tau_ni*(1+tau_ni));     % compute the desired moments
    hmu = nu_ni/tau_ni + y(i)*gos(z_i)/sqrt(tau_ni*(1+tau_ni));
    hs2 = (1-gos(z_i)*(z_i+gos(z_i))/(1+tau_ni))/tau_ni;

    ttau_old = ttau(i);                   %  then find the new tilde parameters
    ttau(i) = 1/hs2 - tau_ni;
    tnu(i) = hmu/hs2 - nu_ni;

    ds2 = ttau(i) - ttau_old;                  % finally rank-1 update Sigma ..
    Sigma = Sigma - ds2/(1+ds2*Sigma(i,i))*Sigma(:,i)*Sigma(i,:);
    mu = Sigma*tnu;                                       % .. and recompute mu

  end
  [Sigma, mu, lml, L] = epComputeParams(K, y, ttau, tnu); % recompute Sigma and
   % mu since repeated rank-one updates eventually destroys numerical precision
end

if sweep == max_sweep
  disp('Warning: maximum number of sweeps reached in function binaryEPGP')
end

if lml > best_lml
  best_ttau = ttau; best_tnu = tnu; best_lml = lml; % keep values for next call
end

if nargin == 4                   % return the negative log marginal likelihood?

  out1 = -lml;
  if nargout > 1                                      % do we want derivatives?
    out2 = 0*logtheta;                         % allocate space for derivatives
    b = tnu-sqrt(ttau).*solve_chol(L,sqrt(ttau).*(K*tnu));
    F = b*b'-repmat(sqrt(ttau),1,n).*solve_chol(L,diag(sqrt(ttau)));
    for j=1:length(logtheta)
      C = feval(covfunc{:}, logtheta, x, j);
      out2(j) = -sum(sum(F.*C))/2;
    end
    if nargout == 4, out3 = ttau; out4 = tnu; end     % return the tilde params
  end

else                               % otherwise compute predictive probabilities

  [a b] = feval(covfunc{:}, logtheta, x, xstar);
  mus = b'*(tnu-sqrt(ttau).*solve_chol(L,sqrt(ttau).*(K*tnu)));    % test means
  v = L'\(repmat(sqrt(ttau),1,size(xstar,1)).*b);
  s2s = a - sum(v.*v,1)';                     % latent test predictive variance
  out1 = sigmoid(mus./sqrt(1+s2s));      % return predictive test probabilities
  out2 = mus;                             % return latent test predictive means
  out3 = s2s;                         % return latent test predictive variances
  out4 = -lml;                        % return negative log marginal likelihood

end


% function to compute the parameters of the Gaussian approximation, Sigma and
% mu, and the log marginal likelihood, lml, from the current site parameters,
% ttau and tnu. The function also may return L (useful for predictions).

function [Sigma, mu, lml, L] = epComputeParams(K, y, ttau, tnu); 

sigmoid = inline('(1+erf(x/sqrt(2)))/2');
n = length(y);                                       % number of training cases
ssi = sqrt(ttau);                                        % compute Sigma and mu
L = chol(eye(n)+ssi*ssi'.*K);
V = L'\(repmat(ssi,1,n).*K);
Sigma = K - V'*V;
mu = Sigma*tnu;

tau_n = 1./diag(Sigma)-ttau;              % compute the log marginal likelihood
nu_n = mu./diag(Sigma)-tnu;                      % vectors of cavity parameters
z = y.*nu_n./sqrt(tau_n.*(1+tau_n));
lml = -sum(log(diag(L)))+sum(log(1+ttau./tau_n))/2+sum(log(sigmoid(z))) ...
      +tnu'*Sigma*tnu/2+nu_n'*((ttau./tau_n.*nu_n-2*tnu)./(ttau+tau_n))/2 ...
      -sum(tnu.^2./(tau_n+ttau))/2;
