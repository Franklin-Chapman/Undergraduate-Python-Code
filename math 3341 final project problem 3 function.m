function [t,u] = Backward_Euler(f,t_span,u0,h)
%function solves a ODE using the backward Euler method
t0 = t_span(1);
tf = t_span(2);
n = (tf+t0)/h;
u = zeros(length(u0),n+1);
u(:,1) = u0;
t = t0:h:tf;
for i = 1:n
    u(:,i+1) = u(:,i)+h*f(t(i),u(:,i));
    fprintf('Solution : %.12f\n',u)
end
end

