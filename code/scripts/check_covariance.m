function covfunc = check_covariance(covfunc)
% covfunc = check_covariance(covfunc)
%
% Converts covariance to cell if needed
%
% Edwin V. Bonilla (edwin.bonilla@nicta.com.au)
% Last update: 22/05/2012

if ischar(covfunc)   % convert to cell if needed
    covfunc = cellstr(covfunc); 
end 
return;
