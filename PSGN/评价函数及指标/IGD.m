function Score = IGD(PopObj,PF)

% PopObej：算法求得的非支配解集
% PF：真实 Pareto 解集

% <metric> <min>
% Inverted generational distance

%------------------------------- Reference --------------------------------
% C. A. Coello Coello and N. C. Cortes, Solving multiobjective optimization
% problems using an artificial immune system, Genetic Programming and
% Evolvable Machines, 2005, 6(2): 163-190.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    % 通过 pdist 计算 PF 与 PopObj 两两欧式距离，通过 min 取每一行的最小值作为列向量
    % 最后再取所有距离的平均值，越小说明分布性和收敛性越

    Distance = min(pdist2(PF,PopObj),[],2); 
    Score    = mean(Distance);
end