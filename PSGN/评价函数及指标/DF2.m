function result = DF2(x)
global t
G=abs(sin(0.5*pi*t));
r=1+floor((10-1)*G);
XII=[x(:,1:r-1),x(:,r+1:10)];
g=1+sum((XII-G).^2,2);
result(:,1:1)=x(:,r:r);
result(:,2:2)=(1-sqrt(x(:,r:r)./g)).*g;
result=result';