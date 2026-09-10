function result = DF7(x)
global t
Alfa=5*cos(0.5*pi*t);
XII=x(:,2:10);
g=1+sum(power(XII-power((1+exp(Alfa*(x(:,1:1)-2.5))),-1),2),2);
result(:,1:1)=g.*((1+t)*power(x(:,1:1),-1));
result(:,2:2)=g.*(x(:,1:1)*power((1+t),-1));
result=result';