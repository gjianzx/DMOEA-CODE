function data = GetData_DF9()
global t n_t
CostFunction = @(x) DF9(x);      % Cost Function
data=zeros(1000,10);
for i=2:1000
    data(i,1)=data(i-1,1)+(1-0)/1000;
end
for i=2:10
    data(:,i:i)=cos(4*(t-1/n_t)+data(:,1:1)+data(:,i-1:i-1));
end
t=t-1/n_t;
data=CostFunction(data);
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