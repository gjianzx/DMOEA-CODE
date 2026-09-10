function result = DF3(x)
global t
G=sin(0.5*pi*t);
H=1.5+G;
XII=x(:,2:10);
g=1+sum((XII-G-power(x(:,1:1),H)).^2,2);
result(:,1:1)=x(:,1:1);
result(:,2:2)=(1-power(x(:,1:1)./g,H)).*g;
result=result';