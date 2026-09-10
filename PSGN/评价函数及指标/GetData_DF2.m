function data = GetData_DF2()
data=zeros(1000,2);
for i=2:1000
    data(i,1)=data(i-1,1)+(1-0)/1000;
end
data(:,2:2)=1-sqrt(data(:,1:1));
end