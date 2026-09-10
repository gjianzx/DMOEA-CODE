function leader = SelectLeader(rep, beta)
% Function: leader = SelectLeader(rep, beta)
%
% Description: 利用轮盘赌选出其中一个Subspace，Subspace聚集密度越小，越容易被选中，
%              再在选中的Subspace中随机选出一个个体作为 leader 即 gbest
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
    P = exp(-beta*N);  % 注意这是有负数，表示个体数越多的Subspace被选择的概率越小
    P = P/sum(P); % 转化为和为 1 的概率
    
    % Selected Cell Index
    sci = RouletteWheelSelection(P);
    
    % Selected Cell
    sc = OC(sci);
    
    % Selected Cell Members
    SCM = find(GI == sc);
    
    % Selected Member Index，random selection
    smi = randi([1 numel(SCM)]);
    
    % Selected Member
    sm = SCM(smi);
    
    % Leader
    leader = rep(sm);

end