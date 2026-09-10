function result = DF11(x)
global t
G=abs(sin(0.5*pi*t));
XII=x(:,3:end);
g=1+G+sum(power(XII-0.5*G*x(:,1:1),2),2);
y_1=(pi/6)*G+(pi/2-(pi/3)*G).*x(:,1:1);
y_2=(pi/6)*G+(pi/2-(pi/3)*G).*x(:,2:2);
result(:,1:1)=g.*sin(y_1);
result(:,2:2)=g.*sin(y_2).*cos(y_1);
result(:,3:3)=g.*cos(y_2).*cos(y_1);
result=result';