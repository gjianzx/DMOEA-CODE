function particle = Mutation(particle, pm, VarMin, VarMax)
% Function: particle = Mutation(particle, pm, VarMin, VarMax)
%
% Description: 随机选择一个决策变量进行变异，避免陷入局部最优
%
%
% Syntax:
%   
%
% Parameters:
%   particle：个体决策变量
%   pm：扰乱算子
%   VarMin：变量最小值
%   VarMax：变量最大值
%
% Return:
%   particle：变异后的个体
%*************************************************************************


    nVar = numel(particle);
    p = randi([1 nVar]); % 随机选取一个变异位置 p
    
    x = particle(p);

    d = pm * (VarMax(p) - VarMin(p));
    ub = x + d; 
    lb = x - d;
    ub = min(ub, VarMax(p));
    lb = max(lb, VarMin(p));

    x = unifrnd(lb, ub, 1);

    particle(p) = x;

end