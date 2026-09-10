function result = DF12(x)
global t
k=10*sin(pi*t);
XII=x(:,3:end);
g=1+sum(power(XII-sin(t*x(:,1:1)),2),2)+abs(sin(floor(k*(2*x(:,1:1)-1))*pi/2)).*abs(sin(floor(k*(2*x(:,2:2)-1))*pi/2));
result(:,1:1)=g.*(cos(0.5*pi*x(:,2:2)).*(cos(0.5*pi*x(:,1:1))));
result(:,2:2)=g.*(sin(0.5*pi*x(:,2:2)).*(cos(0.5*pi*x(:,1:1))));
result(:,3:3)=g.*sin(0.5*pi*x(:,1:1));
result=result';