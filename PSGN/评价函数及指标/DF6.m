function result = DF6(x)
global t
G=sin(0.5*pi*t);
Alfa=0.2+2.8*abs(G);
y=x-G;
YII=y(:,2:10);
g=1+sum((abs(G)*YII.^2-10*cos(2*pi*YII)+10),2);
result(:,1:1)=g.*power(x(:,1:1)+0.1*sin(3*pi*x(:,1:1)),Alfa);
result(:,2:2)=g.*power(1-x(:,1:1)+0.1*sin(3*pi*x(:,1:1)),Alfa);
result=result';