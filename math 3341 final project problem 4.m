% Math 3341, Fall 2021
% Author: Franklin Chapman
%problem 4

clear; close all; clc;
set(groot,'defaulttextinterpreter','latex');
set(groot, 'defaultAxesTickLabelInterpreter','latex');
set(groot, 'defaultLegendInterpreter','latex');
figure(1); hold on;

f = @(x) abs(x.^2-2)+abs(2.*x+3);
x = linspace(-4,0,10000);
y = f(x);

plot(x,y)


gr = 1/(1.61803398875);
err = 1;
c = 0;
x_bot = -4;
x_top = 0;
tol = 10^-8;

%loop to find the min using Golden search
fprintf('Golden Search solutions : %.12f\n')
fprintf('------------------------------------------------------------\n')
fprintf('h\tx_1\tf(x_1)\tx_2\tf(x_2)\tx_1\tf(x_1)\tx_u\tf(x_u)\terror\n')
fprintf('------------------------------------------------------------\n')

while err>tol
    x1 = x_top-(x_top-x_bot)*gr;
    x2 = x_bot+(x_top-x_bot)*gr;
    if f(x1)<f(x2)
        x_top = x2;
    else
        x_bot=x1;
    end
    err = abs(x_top-x_bot);
    c = c+1;
    count(c) = c;
    error(c) = err;
    x_t(c) = x_top;
    x_b(c) = x_bot;
    fprintf('%d\t%2.3f\t%2.3f\t%2.3f\t%2.3f\t%2.3f\t%2.3f\t%2.3f\t%2.3f\t%2.2f \n'...
    ,c,x1,f(x1),x2,f(x2),x_bot,f(x_bot),x_top,f(x_top),err)
end

x_min = successiveparabolicinterpolation(@(x) abs(x.^2-2)+abs(2.*x+3),-3);
fprintf('------------------------------------------------------------\n')
fprintf('Successive Parabolic solution : %.12f\n',x_min)

hold on
plot(x1,f(x1),'g*')
hold on
plot(x_min,f(x_min),'b*')

title('$\displaystyle f = |x.^2-2|+|2.*x+3|$');
xlabel('$x$');
ylabel('$y$');
legend('x','Golden search','Successive parabolic','Location','best')



    


