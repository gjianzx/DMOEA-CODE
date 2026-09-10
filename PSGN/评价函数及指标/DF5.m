function result = DF5(x)
global t
G=sin(0.5*pi*t);
w=floor(10*G);
XII=x(:,2:10);
g=1+sum((XII-G).^2,2);
result(:,1:1)=g.*(x(:,1:1)+0.02*sin(w*pi*x(:,1:1)));
result(:,2:2)=g.*(1-x(:,1:1)+0.02*sin(w*pi*x(:,1:1)));
result=result';