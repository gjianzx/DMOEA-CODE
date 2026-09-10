function result = DF14(x)
global t
G=sin(0.5*pi*t);
XII=x(:,3:end);
g=1+sum((XII-G).^2,2);
y_1=0.5+G*(x(:,1:1)-0.5);
result(:,1:1)=g.*(1-y_1+0.05*sin(6*pi*y_1));
result(:,2:2)=g.*(1-x(:,2:2)+0.05*sin(6*pi*x(:,2:2))).*(y_1+0.05*sin(6*pi*y_1));
result(:,3:3)=g.*(x(:,2:2)+0.05*sin(6*pi*x(:,2:2))).*(y_1+0.05*sin(6*pi*y_1));
result=result';