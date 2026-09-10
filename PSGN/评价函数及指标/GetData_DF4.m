function data = GetData_DF4()
global t n_t
a=sin(0.5*pi*(t-1/n_t));
b=1+abs(cos(0.5*pi*(t-1/n_t)));
H=1.5+a;
data=zeros(1000,2);
for i=2:1000
    data(i,1)=data(i-1,1)+(power(b,H)-0)/1000;
end
data(:,2:2)=power(b-power(data(:,1:1),1/H),H);
end