function particle = FindGridIndex(particle, Grid)
% Function: particle = FindGridIndex(particle, Grid)
%
% Description: 为 rep 中的每个个体划分所在的Subspace，其中 
%              GridSubIndex 表示每维目标所对应的Subspace位置
%              GridIndex 表示将 GridSubIndex 归并为一个数来表示
%
%
% Syntax:
%   
%
% Parameters:
%   particle：当前非支配集中的一个个体
%   Grid：已经划分好的Subspace，用于自适应Subspace算法
%
% Return:
%   particle：划分到对应Subspace后的个体
%*************************************************************************
    nObj = numel(particle.Cost);
    
    nGrid = numel(Grid(1).LB);
    
    particle.GridSubIndex = zeros(1, nObj);
    
    for j = 1:nObj
        
        particle.GridSubIndex(j) = ...
            find(particle.Cost(j)<Grid(j).UB, 1, 'first');
        
    end
    
    % 将高维转化为一维的空间索引
    particle.GridIndex = particle.GridSubIndex(1);
    for j = 2:nObj
        particle.GridIndex = particle.GridIndex-1;
        particle.GridIndex = nGrid*particle.GridIndex;
        particle.GridIndex = particle.GridIndex+particle.GridSubIndex(j);
    end
    
end