% Math 3341, Fall 2021
% Author: Franklin Chapman
%problem 3

set(groot,'defaulttextinterpreter','latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');
f = @(t,u) (2+sqrt(u-2.*t+3));
h = 0.05; %dt, or step size
u0 = [1];
t_span = [0,2];
[t,u] = Backward_Euler(f,t_span,u0,h);

u_exact = @(t) 1+4.*t+(t.^2)/4; %exact solution to compare
t_exact = linspace(0, 2); %for plotting the exact solution
%plotting
plot(t,u, '+g',t_exact,u_exact(t_exact),'-b','LineWidth',1.5);
grid on;
xlabel('t')
ylabel('u(t)')
title('$Solution- of-du/dt = 2+(u-2t+3)^{1/2}$')
legend('Backward Euler', 'Exact Solution', 'Location', 'best')
fprintf('u(2) = %f\n',y(end));

