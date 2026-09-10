function result = DF10(x)
global t
H=2.25+2*cos(0.5*pi*t);
G=sin(0.5*pi*t);
XII=x(:,3:end);
g=1+sum(power(XII-(sin(2*pi*(x(:,1:1)+x(:,2:2))))/(1+abs(G)),2),2);
result(:,1:1)=g.*power(sin(0.5*pi*x(:,1:1)),H);
result(:,2:2)=g.*power((sin(0.5*pi*x(:,2:2)).*(cos(0.5*pi*x(:,1:1)))),H);
result(:,3:3)=g.*power((cos(0.5*pi*x(:,2:2)).*(cos(0.5*pi*x(:,1:1)))),H);
result=result';