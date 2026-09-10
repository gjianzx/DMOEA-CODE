function res = DM(Problem,popSize,MaxIt,T_parameter,group,Algs_name)
% DM: 动态多目标优化主函数
% Problem: 问题对象
% popSize: 种群大小
% MaxIt: 每个环境的最大迭代次数 (对应 Python: tau)
% T_parameter: 环境参数

% 参数获取
n_severity = T_parameter(group, 1);  
tau = T_parameter(group, 2);         
total_envs = floor(T_parameter(group,3) / tau); 

res = cell(1, total_envs);
N = popSize;
con=configure();
DIM = con.dec; 


% 初始化记录变量
centroids = zeros(total_envs, DIM); 
use_variance_len = 5;  % 对应 Python: use_variance_len = 5

% 子代数量设置
count1 = floor(N * 0.5);       % Alg1: CPBP
count2 = N;                    % Alg2: Random Noise
count3 = floor(N * 0.5 / DIM); % Alg3: VGAS (Algorithm 3)
count4 = N;                    % DM

% 上一代种群记录 (初始化为 DIM x N)
Last_Pop_X = zeros(DIM, N); 

for T = 1 : total_envs
    t = 1/n_severity * (T-1); 
    fprintf(' %d ', T);
    tic;
    
    %% === 阶段 1: 环境变化瞬间的响应 (DIM x N) ===
    if T == 1
        % 初始环境，随机生成 (DIM x N)
        initPop_Stage1 = generateRandomPoints(N, Problem);
    else
        last_centroids = centroids(1:T-1, :);
        
        % Algorithm 1 (预测) - 输入输出均为 DIM x N
        if T > 2
            X1 = Algorithm_1(last_centroids, Last_Pop_X, count1, Problem);
        else
            X1 = [];
        end
        
        % Algorithm 2 (随机) - 输入输出均为 DIM x N
        X2 = Algorithm_2(Last_Pop_X, count2, Problem);
        
        % 合并 & 修复 (水平拼接 [DIM x N1, DIM x N2])
        hugePop = [Last_Pop_X, X1, X2];
        hugePop = Repair(hugePop, Problem);
        
        % 随机截断到 N (按列选择)
        [~, huge_cols] = size(hugePop);
        if huge_cols > N
            rand_idx = randperm(huge_cols, N);
            initPop_Stage1 = hugePop(:, rand_idx);
        else
            initPop_Stage1 = hugePop;
        end
    end
    
    Current_Pop_X = initPop_Stage1; % 保持 DIM x N
    
    %% === 阶段 2: 逐代运行前 5 代并收集方差 ===
    variance_history = zeros(use_variance_len, DIM); 
    
    % 【修改 1】初始化一个 Cell 数组，用于存储阶段 2 每一代的 POF
    POF_iter_Stage2 = cell(1, use_variance_len);
    
    % 循环 5 次
    for k = 1:use_variance_len
        % 运行 moead 1 代
        % 这里 moead 返回的第三个参数通常是这一代的历史记录（长度为1的cell或数组）
        % 或者我们可以直接用第二个参数 Pareto 来获取当前代的非支配解
        [pop_k, pareto_k, ~] = moead(Problem, N, 1, t, Current_Pop_X);
        
        % 转回 CPU double
        pop_k_X = double(gather(pop_k)); 
        
        % 维度检查 (确保是 DIM x N)
        [r, c] = size(pop_k_X);
        if r == N && c == DIM
             pop_k_X = pop_k_X'; 
        end
        
        % 计算方差
        variance_history(k, :) = var(pop_k_X, 0, 2)';
        
        % 【修改 2】记录这一代的 Pareto Front (适应度值)
        % 假设 pareto_k.F 是 DIM x M 或 M x DIM，统一存入 cell
        % 使用 gather 确保数据在 CPU
        POF_iter_Stage2{k} = double(gather(pareto_k.F));
        
        % 更新当前种群
        Current_Pop_X = pop_k_X;
    end
    
    PopX_Stage1 = Current_Pop_X;
    
    %% === 阶段 3: 深度学习与 Alg3 ===
    if T > 1
        % --- 1. Algorithm 3 ---
        % 输入: DIM x N, 输出: DIM x M
        X_New_Alg3 = Algorithm_3(PopX_Stage1, variance_history, count3, Problem);
        
        % --- 2. VAE + Diffusion Model (需要转置为 N x DIM) ---
        
        % 准备训练数据: 上一代 Pareto (DIM x N) -> 转置为 (N x DIM)
        Train_Data_NxDIM = All_POS{T-1}'; 
        
        dim1 = DIM;      
        dim2 = 12;       
        dim3 = DIM - 1; 
        
        % 训练 VAE (传入 N x DIM, 边界需为行向量 1 x DIM)
        vae = VAE(dim1, dim2, dim3);
        latent_result = train_VAE(vae, Train_Data_NxDIM, Problem.XLow(:)', Problem.XUpp(:)'); 
        
        % 训练 DM
        dm = DiffusionModel(latent_result, -1, 1);
        dm.train(250, 50); 
        
        % 生成新解 (Latent Space -> N x DIM)
        Pred_latent = dm.get_x0(count4); 
        
        % 解码 (N x DIM)
        % 注意: decode_prediction 内部通常期望输入为 N x Latent，输出 N x DIM
        Pred_Pop_DM_NxDIM = vae.decode_prediction(Pred_latent, Problem.XLow(:)', Problem.XUpp(:)');
        
        % 【关键转置】: N x DIM -> DIM x N
        Pred_Pop_DM_DIMxN = double(gather(Pred_Pop_DM_NxDIM))';
        
        % --- 3. 合并 ---
        % 全部是 DIM x N，水平拼接
        Combined_Pop = [PopX_Stage1, X_New_Alg3, Pred_Pop_DM_DIMxN];
        Combined_Pop = Repair(Combined_Pop, Problem);
        
        % 截断到 N
        [~, comb_cols] = size(Combined_Pop);
        if comb_cols > N
            rand_idx = randperm(comb_cols, N);
            initPop_Stage2 = Combined_Pop(:, rand_idx);
        else
            initPop_Stage2 = Combined_Pop;
        end
    else
        initPop_Stage2 = PopX_Stage1;
    end
    
    %% === 阶段 4: 运行剩余代数 ===
    remaining_it = MaxIt - use_variance_len;
    
    if remaining_it > 0
        % 运行剩余代数
        [PopX, Pareto, POF_iter_Stage4] = moead(Problem, N, remaining_it, t, initPop_Stage2);
        
        % 【修改 3】合并两个阶段的记录
        % 假设 POF_iter_Stage4 是一个 Cell 数组 (1 x remaining_it)
        % 如果 moead 返回的不是 cell，请根据 moead 的实际返回类型调整
        if iscell(POF_iter_Stage4)
            POF_iter = [POF_iter_Stage2, POF_iter_Stage4];
        else
            % 如果 moead 返回的是矩阵，可能需要特殊处理，通常是 Cell
            POF_iter = [POF_iter_Stage2, {POF_iter_Stage4}];
        end
    else
        % 如果没有剩余代数了，直接结束
        PopX = initPop_Stage2;
        % 构造 Pareto 结构 (使用阶段2最后一代的结果)
        % 注意：这里需要重新评估一下由 initPop_Stage2 构成的 Pareto，或者直接用上一代的
        Pareto.X = PopX; 
        
        % 计算 F (如果 moead 没跑，需要手动计算一下当前种群的目标值)
        % 假设 Problem 对象支持 cal_objectives 或者你有单独的函数
        % Pareto.F = Problem.CalObj(PopX); % 伪代码
        Pareto.F = []; % 如果没有计算函数，暂时留空或沿用上一代
        
        % 【修改 4】如果剩余代数为0，POF_iter 就是阶段 2 的记录
        POF_iter = POF_iter_Stage2;
    end
    
    % 结果存储 
    All_POS{T} = double(gather(Pareto.X)); 
    All_POF{T} = double(gather(Pareto.F)); 
    
    centroids(T, :) = mean(All_POS{T}, 2)';
    Last_Pop_X = double(gather(PopX));
    
    res{T}.rt = toc;
    res{T}.turePOF = getBenchmarkPOF(Problem.Name, group, T, T_parameter);
    % 【修改 5】现在 POF_iter 包含了完整历史
    res{T}.POF_iter = POF_iter;    
    res{T}.POS = All_POS{T};

end
end


function X1 = Algorithm_1(centroids, pop_X, count, Problem)
    % centroids: [T, DIM] (历史记录)
    % pop_X: [DIM, N] (当前种群)
    % count: 需要生成的数量
    
    [rows, DIM] = size(centroids);
    X1 = zeros(DIM, count); % 输出 DIM x Count
    if rows < 2, return; end
    
    % 计算移动向量 D: 1 x DIM
    D = centroids(end, :) - centroids(end-1, :); 
    D_col = D'; % 转置为 DIM x 1
    
    [~, N] = size(pop_X);
    for i = 1:count
        idx = mod(i-1, N) + 1;
        % 列向量相加
        X1(:, i) = pop_X(:, idx) + D_col;
    end
    X1 = Repair(X1, Problem);
end

function X_temp = Algorithm_2(X, count, Problem)
    % X: [DIM, N]
    [DIM, ~] = size(X);
    X_temp = zeros(DIM, count);
    
    % min(X, [], 2) 沿行求最小值 -> DIM x 1
    min_x = min(X, [], 2);
    max_x = max(X, [], 2);
    
    range = max_x - min_x;
    lowbound = min_x - 0.25 * range;
    uppbound = max_x + 0.25 * range;
    
    for i = 1:count
        % 生成 DIM x 1 的随机向量
        r = rand(DIM, 1);
        X_temp(:, i) = r .* (uppbound - lowbound) + lowbound;
    end
    X_temp = Repair(X_temp, Problem);
end

function X_new = Algorithm_3(X, variance_history, create_ind, Problem)
    % X: [DIM, N]
    % variance_history: [len, DIM] (历史是按时间堆叠的)
    if isempty(variance_history)
        X_new = [];
        return;
    end
    
    [dim, ~] = size(X); % dim在行
    means = mean(X, 2); % DIM x 1
    new_list = {};
    
    len_hist = size(variance_history, 1);
    x_axis = (1:len_hist)';
    
    XLow_Col = Problem.XLow; % DIM x 1
    XUpp_Col = Problem.XUpp; % DIM x 1
    
    for d = 1:dim
        % variance_history 是 T x DIM, 取第 d 列
        y_data = variance_history(:, d);
        
        if length(unique(y_data)) < 2 
            continue; 
        end
        
        mdl = fitlm(x_axis, y_data);
        slope = mdl.Coefficients.Estimate(2);
        p_value = mdl.Coefficients.pValue(2);
        
        low_d = XLow_Col(d);
        upp_d = XUpp_Col(d);
        mean_d = means(d);
        
        if slope < 0 && p_value < 0.01
            % 减小趋势
            scaler_std = sqrt(y_data(end) * 0.25);
            
            % 随机选择 create_ind 个个体 (列)
            rand_indices = randperm(size(X, 2), create_ind);
            % 提取这些个体的所有维度
            base_ind = X(:, rand_indices); 
            
            % 在第 d 维增加噪声
            noise = -scaler_std + (2*scaler_std) .* rand(1, create_ind);
            base_ind(d, :) = mean_d + noise;
            
            new_list{end+1} = base_ind;
            
        elseif slope > 0 && p_value < 0.01
            % 增大趋势
            % 创建均值个体 [DIM, create_ind]
            base_ind = repmat(means, 1, create_ind);
            
            % 第 d 维随机重置
            base_ind(d, :) = low_d + (upp_d - low_d) .* rand(1, create_ind);
            
            new_list{end+1} = base_ind;
        end
    end
    
    if ~isempty(new_list)
        % 水平拼接 [DIM x N1, DIM x N2, ...]
        X_combined = [new_list{:}];
        X_new = Repair(X_combined, Problem);
    else
        X_new = [];
    end
end