function data = GetData_DF7()
global t n_t
data=zeros(1000,2)+(1+(t-1/n_t))/4;
for i=2:1000
    data(i,1)=data(i-1,1)+(1+(t-1/n_t)-((1+(t-1/n_t))/4))/1000;
end
data(:,2:2)=power(data(:,1:1),-1);
end