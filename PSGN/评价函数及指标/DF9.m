function result = DF9(x)
global t
N=1+floor(10*abs(sin(0.5*pi*t)));
temp=x;
for i=2:10
    temp(:,i:i)=x(:,i:i)-cos(4*t+x(:,1:1)+x(:,i-1:i-1));
end
temp=temp(:,2:end);
g=1+sum(power(temp,2),2);
result(:,1:1)=g.*(x(:,1:1)+max(0,(1/(2*N)+0.1)*sin(2*N*pi*x(:,1:1))));
result(:,2:2)=g.*(1-x(:,1:1)+max(0,(1/(2*N)+0.1)*sin(2*N*pi*x(:,1:1))));
result=result';