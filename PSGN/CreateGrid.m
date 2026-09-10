function Grid = CreateGrid(pop, nGrid, alpha)
% Function: Grid = CreateGrid(pop, nGrid, alpha)
%
% Description: 为每一个目标函数，选出 pop 中个体目标函数值的最小及最大值
%              将最小最大值利用 alpha 参数扩展一定的范围，作为当前维度Subspace的上下限
%
%
% Syntax:
%   
%
% Parameters:
%   pop：rep 即为当前轮的非支配个体
%   nGrid：Subspace的维度大小
%   alpha：自定义参数，用于调整Subspace的最大最小上下线
%
% Return:
%   Grid：定义后的Subspace
%*************************************************************************

    c = [pop.Cost];
    
    cmin = min(c, [], 2);
    cmax = max(c, [], 2);
    
    dc = cmax-cmin;
    cmin = cmin-alpha*dc;
    cmax = cmax+alpha*dc;
    
    nObj = size(c, 1);
    
    empty_grid.LB = [];
    empty_grid.UB = [];
    Grid = repmat(empty_grid, nObj, 1);
    
    for j = 1:nObj
        
        cj = linspace(cmin(j), cmax(j), nGrid+1);
        
        Grid(j).LB = [-inf cj];
        Grid(j).UB = [cj +inf];
        
    end

end