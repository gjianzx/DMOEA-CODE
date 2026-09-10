function result = DF13(x)
global t
G=sin(0.5*pi*t);
p=floor(6*G);
XII=x(:,3:end);
g=1+sum((XII-G).^2,2);
result(:,1:1)=g.*(cos(0.5*pi*x(:,1:1)).*(cos(0.5*pi*x(:,1:1))));
result(:,2:2)=g.*(cos(0.5*pi*x(:,2:2)).*(cos(0.5*pi*x(:,2:2))));
temp_1=sin(0.5*pi*x(:,1:1)).*sin(0.5*pi*x(:,1:1))+sin(0.5*pi*x(:,1:1)).*cos(p*pi*x(:,1:1)).*cos(p*pi*x(:,1:1));
temp_2=sin(0.5*pi*x(:,2:2)).*sin(0.5*pi*x(:,2:2))+sin(0.5*pi*x(:,2:2)).*cos(p*pi*x(:,2:2)).*cos(p*pi*x(:,2:2));
result(:,3:3)=g.*(temp_1+temp_2);
result=result';