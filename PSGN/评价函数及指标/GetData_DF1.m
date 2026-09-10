function data = GetData_DF1()
global t n_t
data=zeros(1000,2);
for i=2:1000
    data(i,1)=data(i-1,1)+(1-0)/1000;
end
data(:,2:2)=1-data(:,1:1).^(0.75*sin(0.5*pi*(t-1/n_t))+1.25);
end