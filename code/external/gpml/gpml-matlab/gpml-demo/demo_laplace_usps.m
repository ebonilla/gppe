% Demo script to illustrate use of binaryLaplaceGP.m on a binary digit
% classification task. 2006-03-29.

clear;
clc;
path(path,'/home/s0340568/Matlab/gpml-matlab/data/usps_resampled');
[x y xx yy] = loadBinaryUSPS(3, 5);
loghyper = [3.0; 0.0];   % set the log hyperparameters

%[newloghyper logmarglik] = minimize(loghyper, 'binaryLaplaceGP', -20, 'covSEiso', 'cumGauss', x, y);
[newloghyper logmarglik] = minimize(loghyper, 'binaryEPGP', -20, 'covSEiso', x, y);
disp('  [newloghyper'' logmarglik(end)]')
[newloghyper' logmarglik(end)]
%pp = binaryLaplaceGP(newloghyper, 'covSEiso', 'cumGauss', x, y, xx);
pp = binaryEPGP(newloghyper, 'covSEiso', x, y, xx);
plot(pp,'g.')
disp('  sum((pp>0.5)~=(yy>0))')
sum((pp>0.5)~=(yy>0))
disp('  mean((yy==1).*log2(pp)+(yy==-1).*log2(1-pp))+1')
mean((yy==1).*log2(pp)+(yy==-1).*log2(1-pp))+1

