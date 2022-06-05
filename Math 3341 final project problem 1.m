% Math 3341, Fall 2021
% Author: Franklin Chapman
%problem 1
set(groot,'defaulttextinterpreter','latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');

f = @(x) 1./2.*x.^2.*(4-x.^2)-(x.^4)./2;% Function to be integrated
a = -1; 
b = 1; % Limits of integration

fprintf('\n----------------- Final project Problem 1 -----------------\n')
fprintf('Integrating f(x,y) = (x^2*y)*exp(-x^2+y^2) \n\n')


ymin = 1/3;
ymax = 11/3;

%solving the x-direction using Gauss quadrature
for n = 1:5 %computing the approx for N=1-5
I_approx = Gauss_Quad(f, a, b, n);

fprintf('Gauss Quadrature (N = %d): %.12f\n', n, I_approx)
end


%solving the y-direction using Simpson's rule
fy = @(y) (-1/(3.^(1/2)).*y + exp(-(-1/(3.^(1/2)).^2 + y.^2)));
N = 20;
h = (ymax-ymin)/N;
x1 = ymin:h:ymax;

s1 = fy(ymin) + fy(ymax);
s2 = 0;
s4 = 0;
fprintf('\n------------------------------------------------------------\n')
for j = 2:2:length(x1)-1
    s4 = s4+fy(x1(j));
    fprintf('Simpson rule s4 (K = %d): %.12f\n', j, s4)
end
for j = 3:2:length(x1)-2
    s2 = s2+fy(x1(j));
    fprintf('Simpson rule s2 (K = %d): %.12f\n', j, s2)
end
Integral_Simp = h/3*(s1+4*s4+2*s2);
fprintf('Simpson rule final result : %.12f\n', Integral_Simp)
fprintf('\n------------------------------------------------------------\n')
%combine the results from Simpson and Gauss

solution = Integral_Simp*I_approx;
fprintf('Combine Gauss and Simpson (Solution = %d): %.12f\n', solution)




