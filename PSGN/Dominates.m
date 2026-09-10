function b = Dominates(x, y)
% Function: b = Dominates(x, y)
%
% Description: 比较两个个体的函数值
%
%
% Syntax:
%   
%
% Parameters:
%   pop：种群
%
% Return:
%   b：比较结果，为布尔型
%*************************************************************************
    if isstruct(x)
        x = x.Cost;
    end
    
    if isstruct(y)
        y = y.Cost;
    end

    b = all(x <= y) && any(x<y);

end