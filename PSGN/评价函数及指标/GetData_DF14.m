function data = GetData_DF14()
global t n_t
CostFunction = @(x) DF14(x);      % Cost Function
G=sin(0.5*pi*(t-1/n_t));
x=zeros(1000,10);
x(:,1:2)=unifrnd(0,1,1000,2);
x(:,3:10)=x(:,3:10)+G;
t=t-1/n_t;
data=CostFunction(x);
t=t+1/n_t;
data = data';
end