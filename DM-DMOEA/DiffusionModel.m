classdef DiffusionModel < handle
    properties
        Parameters      % 存储模型权重 (Weights, Biases, Embeddings)
        
        % 超参数
        num_steps
        input_dim
        num_units
        x_low
        x_upp
        
        % Diffusion 预计算系数 (dlarray格式，用于GPU计算)
        betas
        alphas
        alphas_prod
        alphas_bar_sqrt
        one_minus_alphas_bar_sqrt
    end
    
    methods
        function obj = DiffusionModel(data, x_low, x_upp, num_steps, num_units)
            % 构造函数
            if nargin < 4, num_steps = 10; end
            if nargin < 5, num_units = 32; end
            
            obj.num_steps = num_steps;
            obj.num_units = num_units;
            
            % 确定输入维度 (假设 data 是 [N, Dim])
            obj.input_dim = size(data, 2);
            
            % 处理边界 (确保是行向量)
            if isscalar(x_low), x_low = repmat(x_low, 1, obj.input_dim); end
            if isscalar(x_upp), x_upp = repmat(x_upp, 1, obj.input_dim); end
            obj.x_low = x_low;
            obj.x_upp = x_upp;
            
            % === 1. 初始化超参数 (对应 Python __init__) ===
            % betas = sigmoid(linspace) * ...
            s = linspace(-6, 6, num_steps);
            sig_s = 1 ./ (1 + exp(-s)); % Sigmoid
            betas_val = sig_s * (0.5e-2 - 1e-5) + 1e-5;
            
            alphas_val = 1 - betas_val;
            alphas_prod_val = cumprod(alphas_val);
            
            % 转换为 dlarray 并确保持久化 (如果在 GPU 上训练)
            obj.betas = dlarray(single(betas_val));
            obj.alphas = dlarray(single(alphas_val));
            obj.alphas_prod = dlarray(single(alphas_prod_val));
            obj.alphas_bar_sqrt = sqrt(obj.alphas_prod);
            obj.one_minus_alphas_bar_sqrt = sqrt(1 - obj.alphas_prod);
            
            % === 2. 初始化模型参数 (MLP + Embeddings) ===
            % 结构: Linear -> ReLU -> Linear -> ReLU -> Linear -> ReLU -> Linear
            % 加上 Time Embeddings
            
            % 使用 Glorot/He 初始化
            % Layer 1: Input -> Units
            obj.Parameters.fc1.Weights = dlarray(initializeGlorot([num_units, obj.input_dim]));
            obj.Parameters.fc1.Bias = dlarray(zeros(num_units, 1, 'single'));
            obj.Parameters.emb1 = dlarray(initializeGlorot([num_units, num_steps])); % Embedding
            
            % Layer 2: Units -> Units
            obj.Parameters.fc2.Weights = dlarray(initializeGlorot([num_units, num_units]));
            obj.Parameters.fc2.Bias = dlarray(zeros(num_units, 1, 'single'));
            obj.Parameters.emb2 = dlarray(initializeGlorot([num_units, num_steps])); % Embedding
            
            % Layer 3: Units -> Units
            obj.Parameters.fc3.Weights = dlarray(initializeGlorot([num_units, num_units]));
            obj.Parameters.fc3.Bias = dlarray(zeros(num_units, 1, 'single'));
            obj.Parameters.emb3 = dlarray(initializeGlorot([num_units, num_steps])); % Embedding
            
            % Layer 4: Units -> Output
            obj.Parameters.fc4.Weights = dlarray(initializeGlorot([obj.input_dim, num_units]));
            obj.Parameters.fc4.Bias = dlarray(zeros(obj.input_dim, 1, 'single'));
            
            % 保存标准化后的训练数据
            obj.Data_Normalized = obj.normalize(data);
        end
        
        function train(obj, epochs, batch_size)
            if nargin < 2, epochs = 1000; end
            if nargin < 3, batch_size = 32; end
            
            % 准备数据: [Dim, N] 格式适应 'CB'
            X_train = obj.Data_Normalized'; 
            dlX = dlarray(single(X_train), 'CB');
            
            % 移动到 GPU (如果可用)
            if canUseGPU()
                dlX = gpuArray(dlX);
                obj.moveParamsToGPU();
            end
            
            % 优化器配置
            lr = 1e-3;
            avgG = [];
            avgGS = [];
            
            numObservations = size(dlX, 2);
            
            % === 修复点 1: 处理小样本情况 (Population < BatchSize) ===
            % 确保即使样本少于 BatchSize，也至少训练一次
            if numObservations < batch_size
                actual_batch_size = numObservations;
                numIterationsPerEpoch = 1;
            else
                actual_batch_size = batch_size;
                numIterationsPerEpoch = floor(numObservations / batch_size);
            end
            
            % 防止除以 0
            if numIterationsPerEpoch == 0
                numIterationsPerEpoch = 1;
            end
            
            % fprintf('Starting Diffusion Model Training...\n');
            
            for epoch = 1:epochs
                loss_accum = 0; % 初始为 double
                
                % Shuffle
                idx = randperm(numObservations);
                dlX_shuffled = dlX(:, idx);
                
                for i = 1:numIterationsPerEpoch
                    idxStart = (i-1)*actual_batch_size + 1;
                    
                    % === 修复点 2: 动态计算结束索引，处理无法整除的情况 ===
                    if i == numIterationsPerEpoch
                         idxEnd = numObservations; % 最后一个 Batch 取剩余所有
                    else
                         idxEnd = i*actual_batch_size;
                    end
                    
                    % 防止索引越界
                    if idxStart > numObservations
                        break;
                    end
                    
                    X_batch = dlX_shuffled(:, idxStart:idxEnd);
                    
                    % 计算梯度和损失
                    [loss, gradients] = dlfeval(@obj.model_loss, X_batch, obj.Parameters);
                    
                    % 梯度裁剪 (Clip Grad Norm 1.0)
                    gradients = dlgradientclip(gradients, 1.0);
                    
                    % 更新参数
                    [obj.Parameters, avgG, avgGS] = adamupdate(obj.Parameters, ...
                        gradients, avgG, avgGS, i, lr);
                    
                    loss_accum = loss_accum + loss;
                end
                
                if mod(epoch, 100) == 0 || epoch == 1
                    % === 修复点 3: 增加类型检查，防止 extractdata 报错 ===
                    if isa(loss_accum, 'dlarray')
                        loss_val = extractdata(loss_accum);
                    else
                        loss_val = loss_accum; % 如果是 double，直接使用
                    end
                    
                    avg_loss = loss_val / numIterationsPerEpoch;
                    % fprintf('Epoch %d, Loss: %.4f\n', epoch, avg_loss);
                end
            end
        end
        
        function x0 = get_x0(obj, num_samples)
            % 采样生成新数据
            % num_samples: 需要生成的样本数量 (count4)
            
            % 初始化噪声 x_T ~ N(0, I)
            shape = [obj.input_dim, num_samples];
            x = randn(shape, 'single');
            
            if canUseGPU()
                x = gpuArray(x);
                obj.moveParamsToGPU(); % 确保参数在 GPU
            end
            
            x = dlarray(x, 'CB');
            
            % 反向扩散过程 (从 T 到 1)
            % Python range: reversed(range(num_steps)) -> T-1 ... 0
            % Matlab: num_steps ... 1
            
            for t = obj.num_steps : -1 : 1
                % 生成 t 张量 (全为 t 的向量)
                t_idx = repmat(t, 1, num_samples); 
                
                % 获取当前步的系数
                % 注意: MATLAB 索引从 1 开始
                beta_t = obj.betas(t);
                alpha_t = obj.alphas(t);
                one_minus_alpha_bar_t = obj.one_minus_alphas_bar_sqrt(t);
                
                coeff = beta_t / one_minus_alpha_bar_t;
                
                % 预测噪声 eps
                eps = obj.model_forward(x, t_idx, obj.Parameters);
                
                % 计算均值
                % mean = (x - coeff * eps) / sqrt(alpha_t)
                mean_val = (x - coeff * eps) / sqrt(alpha_t);
                
                if t > 1
                    z = randn(size(x), 'like', x);
                    sigma_t = sqrt(beta_t);
                    x = mean_val + z * sigma_t;
                else
                    x = mean_val;
                end
            end
            
            % 提取数据，反归一化，并限制范围 [0, 1]
            x_raw = extractdata(x)'; % 转置回 [N, Dim]
            x_denorm = obj.denormalize(x_raw);
            
            % Clamp to [0, 1] (根据 Python 代码逻辑)
            x_denorm(x_denorm < 0) = 0;
            x_denorm(x_denorm > 1) = 1;
            
            x0 = x_denorm;
        end
        
        % === 内部核心函数 ===
        
        function [loss, gradients] = model_loss(obj, x0, params)
            % 前向扩散过程并计算损失
            batch_size = size(x0, 2);
            
            % 1. 随机采样时间步 t (1 到 num_steps)
            t = randi([1, obj.num_steps], [1, batch_size]);
            
            % 2. 获取对应的系数 (Gather)
            % a = alphas_bar_sqrt[t]
            a = obj.alphas_bar_sqrt(t); % [1, Batch]
            
            % aml = one_minus_alphas_bar_sqrt[t]
            aml = obj.one_minus_alphas_bar_sqrt(t); % [1, Batch]
            
            % 扩展维度以匹配 x0: [Dim, Batch]
            % 注意：dlarray 的广播机制会自动处理，但明确一点更好
            % 这里的 a 和 aml 是 [1, Batch]，x0 是 [Dim, Batch]
            % MATLAB 会自动沿第一维广播
            
            % 3. 生成噪声 e
            e = randn(size(x0), 'like', x0);
            
            % 4. 加噪: x = x0 * a + e * aml
            x_t = x0 .* a + e .* aml;
            
            % 5. 模型预测噪声
            output = obj.model_forward(x_t, t, params);
            
            % 6. 计算 MSE
            loss = sum((e - output).^2, 'all') / batch_size; % Mean over batch
            
            % 计算梯度
            gradients = dlgradient(loss, params);
        end
        
        function out = model_forward(obj, x, t, params)
            % MLPDiffusion 前向传播
            % x: [Dim, Batch]
            % t: [1, Batch] (时间步索引)
            % params: 权重结构体
            
            % Layer 1
            % Embed: [Units, Steps] 取 t 列 -> [Units, Batch]
            emb1 = params.emb1(:, t); 
            h1 = fullyconnect(x, params.fc1.Weights, params.fc1.Bias);
            h1 = relu(h1 + emb1); % Add embedding
            
            % Layer 2
            emb2 = params.emb2(:, t);
            h2 = fullyconnect(h1, params.fc2.Weights, params.fc2.Bias);
            h2 = relu(h2 + emb2);
            
            % Layer 3
            emb3 = params.emb3(:, t);
            h3 = fullyconnect(h2, params.fc3.Weights, params.fc3.Bias);
            h3 = relu(h3 + emb3);
            
            % Layer 4 (Output)
            out = fullyconnect(h3, params.fc4.Weights, params.fc4.Bias);
        end
        
        function norm_data = normalize(obj, data)
            % 映射 [x_low, x_upp] -> [-1, 1]
            % data: [N, Dim]
            low = obj.x_low;
            upp = obj.x_upp;
            norm_data = 2 * (data - low) ./ (upp - low) - 1;
        end
        
        function denorm_data = denormalize(obj, data)
            % 映射 [-1, 1] -> [x_low, x_upp]
            % data: [N, Dim]
            low = obj.x_low;
            upp = obj.x_upp;
            denorm_data = (data + 1) / 2 .* (upp - low) + low;
        end
        
        function moveParamsToGPU(obj)
            % 辅助函数：将参数移动到 GPU
            fields = fieldnames(obj.Parameters);
            for i = 1:numel(fields)
                if isstruct(obj.Parameters.(fields{i}))
                    subfields = fieldnames(obj.Parameters.(fields{i}));
                    for j = 1:numel(subfields)
                        obj.Parameters.(fields{i}).(subfields{j}) = ...
                            gpuArray(obj.Parameters.(fields{i}).(subfields{j}));
                    end
                else
                    obj.Parameters.(fields{i}) = gpuArray(obj.Parameters.(fields{i}));
                end
            end
            
            obj.betas = gpuArray(obj.betas);
            obj.alphas = gpuArray(obj.alphas);
            obj.alphas_bar_sqrt = gpuArray(obj.alphas_bar_sqrt);
            obj.one_minus_alphas_bar_sqrt = gpuArray(obj.one_minus_alphas_bar_sqrt);
        end
    end
    
    properties (Access = private)
        Data_Normalized
    end
end

% === 辅助函数 (非类方法) ===

function output = fullyconnect(input, weights, bias)
    % 剥离标签以进行矩阵乘法
    x_raw = stripdims(input);
    res = weights * x_raw + bias;
    output = dlarray(res, 'CB');
end

function weights = initializeGlorot(sz)
    numIn = sz(2);
    numOut = sz(1);
    var = 2 / (numIn + numOut);
    weights = sqrt(var) * randn(sz, 'single');
end

function grad = dlgradientclip(grad, clipNorm)
    % 简单的梯度裁剪实现
    globalNorm = 0;
    fields = fieldnames(grad);
    
    % 计算全局范数
    for i = 1:numel(fields)
        val = grad.(fields{i});
        if isstruct(val)
            subfields = fieldnames(val);
            for j = 1:numel(subfields)
                 globalNorm = globalNorm + sum(val.(subfields{j}).^2, 'all');
            end
        else
            globalNorm = globalNorm + sum(val.^2, 'all');
        end
    end
    globalNorm = sqrt(extractdata(globalNorm));
    
    % 缩放
    if globalNorm > clipNorm
        scale = clipNorm / globalNorm;
        for i = 1:numel(fields)
            if isstruct(grad.(fields{i}))
                subfields = fieldnames(grad.(fields{i}));
                for j = 1:numel(subfields)
                    grad.(fields{i}).(subfields{j}) = grad.(fields{i}).(subfields{j}) * scale;
                end
            else
                grad.(fields{i}) = grad.(fields{i}) * scale;
            end
        end
    end
end

function result = canUseGPU()
    try
        result = (gpuDeviceCount > 0);
    catch
        result = false;
    end
end