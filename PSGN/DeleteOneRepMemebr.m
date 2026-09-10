function rep = DeleteOneRepMemebr(rep, gamma)
% Function: rep = DeleteOneRepMemebr(rep, gamma)
%
% Description: 利用轮盘赌选出其中一个Subspace，Subspace聚集密度越大，越容易被选中，
%              再在选中的Subspace中随机选出一个个体删除
%
%
% Syntax:
%   
%
% Parameters:
%   rep：当前迭代的非支配个体
%   beta：自定义参数
%
% Return:
%   leader：被选中的 gbest 
%*************************************************************************
    % Grid Index of All Repository Members
    GI = [rep.GridIndex];
    
    % Occupied Cells
    OC = unique(GI);
    
    % Number of Particles in Occupied Cells
    N = zeros(size(OC));
    for k = 1:numel(OC)
        N(k) = numel(find(GI == OC(k)));
    end
    
    % Selection Probabilities
    P = exp(gamma*N);
    P = P/sum(P);
    
    % Selected Cell Index
    sci = RouletteWheelSelection(P);
    
    % Selected Cell
    sc = OC(sci);
    
    % Selected Cell Members
    SCM = find(GI == sc);
    
    % Selected Member Index
    smi = randi([1 numel(SCM)]);
    
    % Selected Member
    sm = SCM(smi);
    
    % Delete Selected Member
    rep(sm) = [];

end