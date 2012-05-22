function [out1, out2, out3, out4] = binaryLaplaceGP(logtheta, covfunc, lik, x, y, xstar);

% binaryLaplaceGP - Laplace's approximation for binary Gaussian process
% classification. Two modes are possible: training or testing: if no test
% cases are supplied, then the approximate negative log marginal likelihood
% and its partial derivatives wrt the hyperparameters is computed; this mode is
% used to fit the hyperparameters. If test cases are given, then the test set
% predictive probabilities are returned. The program is flexible in allowing
% several different likelihood functions and a multitude of covariance
% functions.
%
% usage: [nlml dnlml] = binaryLaplaceGP(logtheta, covfunc, lik, x, y);
%    or: [p mu s2 nlml] = binaryLaplaceGP(logtheta, covfunc, lik, x, y, xstar);
%
% where:
%
%   logtheta is a (column) vector of hyperparameters
%   covfunc  is the name of the covariance function (see below)
%   lik      is the name of the likelihood function (see below)
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
% function, as specified by the "cov" input to the function, specifying the 
% name of a covariance function. A number of different covariance function are
% implemented, and it is not difficult to add new ones. See "help covFunctions"
% for the details.
%
% The shape of the likelihood function is given by the "lik" input to the
% function, specifying the name of the likelihood function. The two implemented
% likelihood functions are:
%   
%   logistic   the logistic function: 1/(1+exp(-x)) 
%   cumGauss   the cumulative Gaussian (error function)
%
% The function can conveniently be used with the "minimize" function to train
% a Gaussian process:
%
% [logtheta, fX, i] = minimize(logtheta, 'binaryLaplaceGP', length, 'covSEiso',
%                              'cumGauss', x, y);
%
% Note, that the function has a set of persistent variables where the best "a"
% vector so far and the value of the corresponding approximate negative log 
% marginal likelihood is recorded. The Newton iteration is started from this
% guess (if it isn't worse than zero), which should mostly be quite reasonable
% guesses when the function is called repeatedly (from eg "minimize"), when
% finding good values for the hyperparameters.
%
% Copyright (c) 2004, 2005, 2006 by Carl Edward Rasmussen, 2006-03-20.

if ischar(covfunc), covfunc = cellstr(covfunc); end % convert to cell if needed
[n, D] = size(x);
if eval(feval(covfunc{:})) ~= size(logtheta, 1)
  error('Error: Number of parameters do not agree with covariance function')
end
persistent best_a best_value;   % keep a copy of the best "a" and its obj value

tol = 1e-6;                  % tolerance for when to stop the Newton iterations
[n D] = size(x);
K = feval(covfunc{:}, logtheta, x);            % evaluate the covariance matrix

if any(size(best_a) ~= [n 1])      % find a good starting point for "a" and "f"
  f = zeros(n,1); a = f; [lp dlp W] = feval(lik, f, y);         % start at zero
  Psi_new = -n*log(2); best_value = inf;
else
  a = best_a; f = K*a; [lp dlp W] = feval(lik, f, y); % try the best "a" so far
  Psi_new = -a'*f/2 + lp;         
  if Psi_new < -n*log(2)                                  % if zero is better..
    f = zeros(n,1); a = f; [lp dlp W] = feval(lik, f, y);  % ..then switch back
    Psi_new = -a'*f/2 + lp;
  end
end
Psi_old = -inf;                                   % make sure while loop starts

while Psi_new - Psi_old > tol                       % begin Newton's iterations
  Psi_old = Psi_new; a_old = a; 
  sW = sqrt(W);                     
  L = chol(eye(n)+sW*sW'.*K);  
  b = W.*f+dlp;
  a = b - sW.*solve_chol(L,sW.*(K*b));
  f = K*a;
  [lp dlp W d3lp] = feval(lik, f, y);

  Psi_new = -a'*f/2 + lp;
  i = 0;
  while i < 10 & Psi_new < Psi_old               % if objective didn't increase
    a = (a_old+a)/2;                                 % reduce step size by half
    f = K*a;
    [lp dlp W d3lp] = feval(lik, f, y);
    Psi_new = -a'*f/2 + lp;
    i = i + 1;
  end
end                                                   % end Newton's iterations

sW = sqrt(W);                     
L = chol(eye(n)+sW*sW'.*K);  

nlmarglik = a'*f/2 - lp + sum(log(diag(L)));      % approx neg log marginal lik
if nlmarglik < best_value                                   % if best so far...
  best_a = a; best_value = nlmarglik;          % ...then remember for next call
end

if nargin == 5                   % return the negative log marginal likelihood?

  out1 = nlmarglik;    
  if nargout == 2                                     % do we want derivatives?
    out2 = 0*logtheta;                         % allocate space for derivatives
    Z = repmat(sW, 1, n).*solve_chol(L, diag(sW));
    C = L'\(repmat(sW,1,n).*K);                 % FIX: use that L is triangular
    s2 = -0.5*(diag(K)-sum(C.^2,1)').*d3lp;
    for j=1:length(logtheta)
      C = feval(covfunc{:}, logtheta, x, j);
      s1 = a'*C*a/2-sum(sum(Z.*C))/2;
      b = C*dlp;
      s3 = b-K*(Z*b);
      out2(j) = -s1-s2'*s3;
    end
  end

else                               % otherwise compute predictive probabilities

  [a b] = feval(covfunc{:}, logtheta, x, xstar);
  v = L'\(repmat(sW,1,size(xstar,1)).*b);       % FIX: use that L is triangular
  mu = b'*dlp;
  s2 = a - sum(v.*v,1)';
  out1 = feval(lik, mu, s2, []);                       % the real probabilities
  if nargout > 1
    out2 = mu; out3 = s2;
    if nargout == 4
      out4 = lmarglik;
    end
  end
end


% Below are the possible likelihood functions. They have two possible modes. If
% called with two input arguments, the log likelihood (scalar), its derivatives
% (vector), 2nd derivatives (vector) and 3rd derivatives (vector). The 2nd and
% 3rd derivatives are represented as vectors, since "cross-terms" are zero, as
% the likelihood factorizes over cases. If a dummy argument is supplied, then
% the first argument is the mean and the  second argument the variance of a
% Gaussian, and the average likelihood wrt this Gaussian is returned.

function [lp, dlp, d2lp, d3lp] = logistic(f, y, dummy);

if nargin == 2
  s = -f.*y; ps = max(0,s); lp = -sum(ps+log(exp(-ps)+exp(s-ps)));
  s = min(0,f); p = exp(s)./(exp(s)+exp(s-f)); dlp = (y+1)/2-p; 
  d2lp = exp(2*s-f)./(exp(s)+exp(s-f)).^2;
  d3lp = 2*d2lp.*(0.5-p);
else
  lp = erfint(f, y);
end


function [lp, dlp, d2lp, d3lp] = cumGauss(f, y, dummy);

if nargin == 2
  p = (1+erf(f.*y/sqrt(2)))/2 + 1e-10;
  n = exp(-f.^2/2)/sqrt(2*pi);
  lp = sum(log(p));
  dlp = y.*n./p;
  d2lp = +n.^2./p.^2+y.*f.*n./p;
  d3lp = 2*y.*n.^3./p.^3+3*f.*n.^2./p.^2+y.*(f.^2-1).*n./p; 
  d3lp = -d3lp;
else
  lp = (1+erf(f./sqrt(1+y)/sqrt(2)))/2;
end
