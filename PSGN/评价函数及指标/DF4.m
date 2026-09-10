function result = DF4(x)
global t
a=sin(0.5*pi*t);
b=1+abs(cos(0.5*pi*t));
H=1.5+a;
XII=x(:,2:10);
for i=1:9
    XII(:,i:i)=XII(:,i:i)-a*(x(:,1:1).*x(:,1:1))./(i+1);
end
g=1+sum((XII).^2,2);
result(:,1:1)=g.*power(abs(x(:,1:1)-a),H);
result(:,2:2)=g.*power(abs(x(:,1:1)-a-b),H);
result=result';