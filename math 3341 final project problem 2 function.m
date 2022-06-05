function [x, converg] = steepest_decent(A,b,x,tol,maxiter)
%function takes functions in vector format, and then solves the system of
%equations using the steepest decent method.

%takes in vector A, b and an initial guess for x of 0 for all directions
%r = residual
%iter = iteration (current iteration in the loop)
%converg = convergence, or when we have reached the min

iter = 1;
r = b-A.*x;
delta = r'.*r;
converg = delta;
delta0 = delta;

while ((delta > tol.*delta0) & (iter < maxiter))
    q = A.*r;
    alpha = delta/(q'.*r);
    x = x+alpha.*r;
    if mod(iter, 50) == 0
        r=b-A.*x;
    else
        r = r-alpha.*q;
    end
    delta = r'.*r;
    converg = [converg,delta];
    iter = iter+1;
end
end

