function result = DF1(x)
global t
G=abs(sin(0.5*pi*t));
H=0.75*sin(0.5*pi*t)+1.25;
XII=x(:,2:10);
g=1+sum((XII-G).^2,2);
result(:,1:1)=x(:,1:1);
result(:,2:2)=(1-power(x(:,1:1)./g,H)).*g;
result=result';