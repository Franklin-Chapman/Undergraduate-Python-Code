function [x,fo]=successiveparabolicinterpolation(f,x)
%Input
%f is a function handle of the function to be minimized
%x is the current position
%
%Output
%x is the new location
%fc is the function evaluated at the new location

xtol=1e-8;
%ytol=1e-8;
xc=[x-1 x x+1];
fc=f(xc);
for k=1:10000
  xnew=(fc(1)*(xc(2)^2-xc(3)^2)-fc(2)*(xc(1)^2-xc(3)^2)+fc(3)*(xc(1)^2-xc(2)^2))/(2*(fc(1)*(xc(2)-xc(3))-fc(2)*(xc(1)-xc(3))+fc(3)*(xc(1)-xc(2))));
  if isnan(xnew)||isinf(xnew)
    xnew=xc(1);
  end
  %ynew=f(xnew);
  %if any(abs(xnew-xc)<xtol)||all(abs(ynew-fc)<ytol)
   % break
  %end
  xc=[xnew xc(1:end-1)];
  %fc=[ynew fc(1:end-1)];
end
x=xnew;
%fo=ynew;
end

