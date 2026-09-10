function data = GetData_DF11()
global t n_t
CostFunction = @(x) DF11(x);      % Cost Function
x=zeros(1000,10);
x(:,1:2)=unifrnd(0,1,1000,2);
G=abs(sin(0.5*pi*(t-1/n_t)));
x(:,3:10)=x(:,3:10)+0.5*G*x(:,1:1);
t=t-1/n_t;
data=CostFunction(x);
t=t+1/n_t;
data = data';
end