classdef VAE
    % VAE 模型类，模仿 PyTorch 的 nn.Module 结构
    % 管理模型的参数（权重和偏置）

    properties
        Parameters % 存储所有的 Learnables (weights, biases)
        Dim1
        Dim2
        Dim3
    end

    methods
        function obj = VAE(dim1, dim2, dim3)
            % 构造函数：初始化权重
            obj.Dim1 = dim1;
            obj.Dim2 = dim2;
            obj.Dim3 = dim3;
            
            % 初始化参数 (使用 He 初始化或 Xavier 初始化)
            % 格式: [OutputSize, InputSize]
            sz1 = [dim2, dim1];
            sz2 = [dim3, dim2];
            sz3 = [dim2, dim3];
            sz4 = [dim1, dim2];
            
            obj.Parameters.fc1.Weights = dlarray(initializeGlorot(sz1)); 
            obj.Parameters.fc1.Bias = dlarray(zeros(dim2, 1, 'single'));
            
            obj.Parameters.fc21.Weights = dlarray(initializeGlorot(sz2));
            obj.Parameters.fc21.Bias = dlarray(zeros(dim3, 1, 'single'));
            
            obj.Parameters.fc22.Weights = dlarray(initializeGlorot(sz2));
            obj.Parameters.fc22.Bias = dlarray(zeros(dim3, 1, 'single'));
            
            obj.Parameters.fc3.Weights = dlarray(initializeGlorot(sz3));
            obj.Parameters.fc3.Bias = dlarray(zeros(dim2, 1, 'single'));
            
            obj.Parameters.fc4.Weights = dlarray(initializeGlorot(sz4));
            obj.Parameters.fc4.Bias = dlarray(zeros(dim1, 1, 'single'));
        end

        function [z, mu, logvar] = encode(obj, x, params)
            % 使用传入的 params 进行计算，以便 dlgradient 跟踪
            h1 = relu(fullyconnect(x, params.fc1.Weights, params.fc1.Bias));
            mu = fullyconnect(h1, params.fc21.Weights, params.fc21.Bias);
            logvar = fullyconnect(h1, params.fc22.Weights, params.fc22.Bias);
            
            std_dev = exp(0.5 * logvar);
            epsilon = randn(size(mu), 'like', mu);
            z = mu + epsilon .* std_dev;
        end

        function x_recon = decode(obj, z, params)
            h3 = relu(fullyconnect(z, params.fc3.Weights, params.fc3.Bias));
            out = fullyconnect(h3, params.fc4.Weights, params.fc4.Bias);
            x_recon = tanh(out); 
        end

        function [x_recon, mu, logvar] = forward(obj, x, params)
            [z, mu, logvar] = obj.encode(x, params);
            x_recon = obj.decode(z, params);
        end
        
        function result = decode_prediction(obj, z, x_low, x_upp)
            % 用于预测的解码接口
            % z: [Latent_Dim, Batch]
            % 将数据移回 CPU 并反归一化
            z_in = z';
            % 确保 z 是 dlarray
            if ~isa(z_in, 'dlarray')
                z_in = dlarray(z_in, 'CB');
            end
            
            % 调用解码网络
            % 注意：这里使用 obj.Parameters，因为预测时不需要计算梯度
            % 输出 x_recon 的形状是 [Dim, N]
            x_recon = obj.decode(z_in, obj.Parameters);
            
            % === 【关键修改 2】提取数据并转置回 [N, Dim] ===
            % extractdata 将 dlarray 转为普通数组
            % 转置 ' 是为了让输出变回 [N, Dim] 格式，方便后续合并
            res_data = extractdata(x_recon)'; 
            
            % 反归一化 (res_data 是 N x Dim, x_low 是 1 x Dim，广播计算)
            result = VAE.inverse_transform(res_data, x_low, x_upp);
        end
    end
    
    methods(Static)
        function tensors = inverse_transform(tensors, x_low, x_upp)
            % 将张量从[-1, 1]转换回[x_low, x_upp]区间
            % 输入 tensors 维度: [N, Dim] (NumPy 风格)
            
            % clip/clamp 操作
            tensors(tensors > 1) = 1;
            tensors(tensors < -1) = -1;
            
            % 线性变换到 [0, 1]
            tensors = (tensors + 1) / 2;
            
            % 缩放到 [x_low, x_upp]
            tensors = tensors .* (x_upp - x_low) + x_low;
        end
    end
end

function output = fullyconnect(input, weights, bias)
    % input: [InDim, Batch]
    % weights: [OutDim, InDim]
    % bias: [OutDim, 1]
    
    % === 修改点：增加类型检查 ===
    % 如果输入是 dlarray，使用 stripdims 剥离标签
    if isa(input, 'dlarray')
        x_raw = stripdims(input);
    else
        % 如果输入已经是 gpuArray 或 double，直接使用，不需要 stripdims
        x_raw = input;
    end
    
    % 执行矩阵乘法
    res = weights * x_raw + bias;
    
    % 重新赋予 'CB' (Channel, Batch) 标签，以便传入下一层
    output = dlarray(res, 'CB');
end

% 辅助函数：权重初始化 (Glorot/Xavier)
function weights = initializeGlorot(sz)
    numIn = sz(2);
    numOut = sz(1);
    var = 2 / (numIn + numOut);
    weights = sqrt(var) * randn(sz, 'single');
end



