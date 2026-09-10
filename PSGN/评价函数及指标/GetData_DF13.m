function data = GetData_DF13()
global t n_t
CostFunction = @(x) DF13(x);      % Cost Function
G=sin(0.5*pi*(t-1/n_t));

x=zeros(1000,10);
x(:,1:2)=unifrnd(0,1,1000,2);
x(:,3:10)=x(:,3:10)+G;
t=t-1/n_t;
data=CostFunction(x);
t=t+1/n_t;
data = data';
[FrontValue,~] = NonDominateSort(data,1);
empty_particle.Cost = [];
empty_particle.IsDominated = [];
temp = repmat(empty_particle, 1000, 1);
for i=1:1000
    temp(i).Cost=data(i:i,:);
    if FrontValue(1,i)==1
        temp(i).IsDominated=false;
    else
        temp(i).IsDominated=true;
    end
end
temp = temp(~[temp.IsDominated]);
data=[];
for i=1:numel(temp)
    data(i:i,:)=temp(i).Cost;
end
end