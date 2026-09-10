function latent_data = train_VAE(vae_model, dataset, XLow, XUpp)
    % train_VAE: 训练 VAE 模型
    
    EPOCH = 200; 
    LR = 5e-3;
    BATCHSIZE = 50; 
    
    % 数据预处理: [N, Dim] -> [Dim, N] (转置以适应 MATLAB DL)
    norm_dataset = 2 * ((dataset - XLow) ./ (XUpp - XLow)) - 1;
    norm_dataset = norm_dataset'; 
    
    % 转换为 dlarray
    if canUseGPU()
        dlX = dlarray(single(norm_dataset), 'CB');
        dlX = gpuArray(dlX); 
         % fprintf('Training VAE on GPU...\n');
    else
        dlX = dlarray(single(norm_dataset), 'CB');
         % fprintf('Training VAE on CPU...\n');
    end
    
    avgG = [];
    avgGS = [];
    
    numObservations = size(norm_dataset, 2);
    
    % === 修复点 1: 处理小样本情况 ===
    % 确保即使 numObservations < BATCHSIZE，也能计算出至少 1 次迭代
    if numObservations < BATCHSIZE
        actual_batch_size = numObservations;
        numIterationsPerEpoch = 1;
    else
        actual_batch_size = BATCHSIZE;
        numIterationsPerEpoch = floor(numObservations / BATCHSIZE);
    end
    
    % 防止空数据导致除以0
    if numIterationsPerEpoch == 0
        numIterationsPerEpoch = 1;
    end
    
    for epoch = 1:EPOCH
        loss_accum = 0; % 初始为 double
        
        % Shuffle
        idx = randperm(numObservations);
        dlX_shuffled = dlX(:, idx);
        
        for i = 1:numIterationsPerEpoch
            idxStart = (i-1)*actual_batch_size + 1;
            
            % 动态计算 idxEnd，处理不能整除或小样本的情况
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
            
            % 计算梯度
            % 注意：这里显式传递 Parameters 和 vae_model
            [loss, gradients] = dlfeval(@modelLoss, vae_model.Parameters, X_batch, vae_model);
            
            % 更新参数
            [vae_model.Parameters, avgG, avgGS] = adamupdate(vae_model.Parameters, ...
                gradients, avgG, avgGS, i, LR);
            
            % 累加 Loss (double + dlarray = dlarray)
            loss_accum = loss_accum + loss;
        end
        
        % 打印日志
        if mod(epoch, 100) == 0 || epoch == 1
             % === 修复点 2: 增加类型检查，修复 extractdata 报错 ===
             if isa(loss_accum, 'dlarray')
                 loss_val = extractdata(loss_accum);
             else
                 loss_val = loss_accum; % 如果是 double (例如 0)，直接使用
             end
             
             avg_loss = loss_val / numIterationsPerEpoch;
             % fprintf('epoch: %2d | Loss: %.4f\n', epoch, avg_loss);
        end
    end
    
    % 训练结束，计算隐空间
    % 使用最新的参数进行编码
    [mu, logvar] = vae_model.encode(dlX, vae_model.Parameters);
    std_dev = exp(0.5 * logvar);
    epsilon = randn(size(mu), 'like', mu);
    z = mu + epsilon .* std_dev;
    
    % 提取数据并转置回 [N, Dim]
    latent_data = extractdata(z)'; 
end

% === 损失函数 ===
function [loss, gradients] = modelLoss(params, x, vae_obj)
    
    % 前向传播 (使用 params)
    [x_recon, mu, logvar] = vae_obj.forward(x, params);
    
    % 1. Reconstruction Loss (MSE)
    MSE = sum((x_recon - x).^2, 'all');
    
    % 2. KL Divergence
    KLD_element = -0.5 * (1 + logvar - mu.^2 - exp(logvar));
    KLD = sum(KLD_element, 'all');
    
    % Total Loss
    batch_size = size(x, 2);
    % 按照 Python 逻辑 (reduction='sum')，最后再除以 batch_size 使得梯度规模正常
    loss = (10 * MSE + KLD) / batch_size;
    
    % 计算梯度
    gradients = dlgradient(loss, params);
end

% === GPU 检查工具 ===
function result = canUseGPU()
    try
        result = (gpuDeviceCount > 0);
    catch
        result = false;
    end
end