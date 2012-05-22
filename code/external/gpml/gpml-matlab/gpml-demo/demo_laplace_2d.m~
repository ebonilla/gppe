% demonstrate the Laplace approximation on a 2-d classification task.
% 2006-03-23.

clear;
clf;
n1=80; n2=40;
S1 = eye(2); S2 = [1 0.95; 0.95 1];
m1 = [0.75; 0]; m2 = [-0.75; 0];                            
randn('seed',17);
x1 = chol(S1)'*randn(2,n1)+repmat(m1,1,n1);
x2 = chol(S2)'*randn(2,n2)+repmat(m2,1,n2);
x = [x1 x2]';
y = [repmat(-1,1,n1) repmat(1,1,n2)]';
[t1 t2] = meshgrid(-4:0.1:4,-4:0.1:4);
t = [t1(:) t2(:)];
tt = sum((t-repmat(m1',length(t),1))*inv(S1).*(t-repmat(m1',length(t),1)),2);
z1 = n1*exp(-tt/2)/sqrt(det(S1));
tt = sum((t-repmat(m2',length(t),1))*inv(S2).*(t-repmat(m2',length(t),1)),2);
z2 = n2*exp(-tt/2)/sqrt(det(S2));

loghyper = [0; 0];
newloghyper = minimize(loghyper, 'binaryLaplaceGP', -20, 'covSEiso', 'cumGauss', x, y)
%newloghyper = minimize(loghyper, 'binaryEPGP', -20, 'covSEiso', x, y)

[p3 out2 out3 out4] = binaryLaplaceGP(newloghyper, 'covSEiso', 'cumGauss',x, y, t);
% [p3 out2 out3 out4] = binaryEPGP(newloghyper, 'covSEiso', x, y, t);

out4


contour(t1,t2,reshape(p3,size(t1)),[0.1:0.1:0.9]);
hold on
plot(x1(1,:),x1(2,:),'b+')
plot(x2(1,:),x2(2,:),'r+')
