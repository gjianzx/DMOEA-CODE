function data = GetData_DF8()
global t n_t
CostFunction = @(x) DF8(x);      % Cost Function
data=zeros(1000,10);
G=sin(0.5*pi*(t-1/n_t));
Beta=100*G*G;
for i=2:1000
    data(i,1)=data(i-1,1)+(1-0)/1000;
end
data(:,2:10)=data(:,2:10)+(G*sin(4*pi*power(data(:,1:1),Beta)))/(1+abs(G));
t=t-1/n_t;
data=CostFunction(data);
t=t+1/n_t;
data = data';
end