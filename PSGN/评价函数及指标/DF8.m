function result = DF8(x)
global t
Alfa=2.25+2*cos(2*pi*t);
G=sin(0.5*pi*t);
Beta=100*G*G;
XII=x(:,2:10);
g=1+sum(power((XII-(G*sin(4*pi*power(x(:,1:1),Beta)))/(1+abs(G))),2),2);
result(:,1:1)=g.*(x(:,1:1)+0.1*sin(3*pi*x(:,1:1)));
result(:,2:2)=g.*power((1-x(:,1:1)+0.1*sin(3*pi*x(:,1:1))),Alfa);
result=result';