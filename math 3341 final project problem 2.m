% Math 3341, Fall 2021
% Author: Franklin Chapman
%problem 2

set(groot,'defaulttextinterpreter','latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');

A = [3 1 -1; 1 5 2; -1 2 5]
b = [4 -1 1]'
x = [0,0,0];
tol = 10^(-7);
maxiter = 10;

[x, converg] = steepest_decent(A,b,x,tol,maxiter)    