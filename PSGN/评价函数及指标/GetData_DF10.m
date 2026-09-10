function data = GetData_DF10()
global t n_t
CostFunction = @(x) DF10(x);      % Cost Function
x=zeros(1000,10);
x(:,1:2)=unifrnd(0,1,1000,2);
G=sin(0.5*pi*(t-1/n_t));
x(:,3:10)=x(:,3:10)+(sin(2*pi*(x(:,1:1)+x(:,2:2))))/(1+abs(G));
t=t-1/n_t;
data=CostFunction(x);
t=t+1/n_t;
data = data';
end