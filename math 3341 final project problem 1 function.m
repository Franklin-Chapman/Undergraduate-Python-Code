function I = Gauss_Quad(f,a,b,n)
  
% wi = weights
% xi = nodes
switch(n)
case 1
wi = 2;
xi = 0;
case 2
wi = [1 1];
xi = [0.57735027 -0.57735027];
case 3
wi = [0.5555555 0.88888889 0.55555556];
xi = [0.77459667 0 -0.77459667];
case 4
wi = [0.65214515 0.34785485 0.65214515 0.34785485];
xi = [0.33998104 0.86113631 -0.33998104 -0.86113631];
case 5
wi = [0.23692689 0.47862867 0.568888889 0.47862867 0.23692689];
xi = [0.90617985 0.53846931 0 -0.53846931 -0.90617985];
otherwise
I = 'Invalid value';
return
end

I = 0;
x = a:(b-a)/n:b;
for j = 1:length(x)-1
I = I + sum(wi .* (0.5*(x(j+1)-x(j)) * f(0.5*(x(j+1)-x(j))*xi + 0.5*(x(j+1)+x(j))))); % Gaussian Quadrature Rule
end
end

