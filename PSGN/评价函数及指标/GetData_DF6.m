function data = GetData_DF6()
global t n_t
CostFunction = @(x) DF6(x);      % Cost Function
data=zeros(1000,10);
G=sin(0.5*pi*(t-1/n_t));
for i=2:1000
    data(i,1)=data(i-1,1)+(1-0)/1000;
end
data(:,2:10)=data(:,2:10)+G;
t=t-1/n_t;
data=CostFunction(data);
t=t+1/n_t;
data = data';
end