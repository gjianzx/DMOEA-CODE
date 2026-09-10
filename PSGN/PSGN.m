function res= PSGN(Problem,popSize,MaxIt,T_parameter,group,Algs_name)
    currentDir = fileparts(mfilename('fullpath'));  
    addpath(genpath(currentDir));  
    global t n_t
    Problem_ID_list = [1 2 3 4 5 6 7 8 9 10 11 12 13 14]; % Benchmark test function DF1~14
    problem_ID = sscanf(Problem.Name, 'DF%d');
    isNativeDF = isscalar(problem_ID) && problem_ID >= 1 && problem_ID <= numel(Problem_ID_list);
    Max_run=1;
    Env=floor(T_parameter(group,3)/T_parameter(group,2)); % 环境变化的次数
    nRep = 100;          % Repository Size
    w = 0.6;              % Inertia Weight
    nGrid = 20;            % Number of Grids
    alpha = 0.1;          % Inflation Rate
    beta = 2;             % Leader Selection Pressure
    gamma = 2;            % Deletion Selection Pressure
    mu = 0.1;             % Mutation4 Rate
    eplision = 1e-5;       % c control Rate
    epolision = 1e3;          % network control Rate
    
    n_t=T_parameter(group,1); % severity of change
    Gamma_t=T_parameter(group,2); %frequency of change
    if isNativeDF
        Problem_ID = Problem_ID_list(problem_ID);
        [CostFunction,GetData,VarMin,VarMax,M] = Get_problem(Problem_ID); %#ok<ASGLU>
    elseif startsWith(Problem.Name,'GTS')
        CostFunction = @(x) EvaluateFrameworkProblem(Problem,x);
        GetData = []; %#ok<NASGU>
        VarMin = Problem.XLow(:)';
        VarMax = Problem.XUpp(:)';
        M = Problem.NObj; %#ok<NASGU>
    else
        error('PSGN:UnsupportedProblem', ...
            'PSGN supports DF1-DF14 and GTS problems, but received %s.',Problem.Name);
    end
    nVar = numel(VarMin);   % Number of Decision Variables
    VarSize = [1 nVar];     % Size of Decision Variables Matrix
    N = popSize;% 种群规模
    it = 1;
    t=fix(it/Gamma_t)/n_t;
    % 初始化种群
    empty_particle.Position = [];
    empty_particle.Velocity = [];
    empty_particle.local_c = 0.5;
    empty_particle.global_c = 0.5;
    empty_particle.Cost = [];
    empty_particle.Best.Position = [];
    empty_particle.Best.Cost = [];
    empty_particle.IsDominated = [];
    empty_particle.GridIndex = [];
    empty_particle.GridSubIndex = [];
    empty_particle.effect = [1 0 0];
    pop = repmat(empty_particle, N, 1);
    for i = 1:N
        pop(i).Position = unifrnd(VarMin, VarMax, VarSize);
        pop(i).Velocity = zeros(VarSize);
        pop(i).Cost = CostFunction(pop(i).Position);
        % Update Personal Best
        pop(i).Best.Position = pop(i).Position;
        pop(i).Best.Cost = pop(i).Cost;
    end
    % 非支配排序
    [FrontValue,~] = NonDominateSort([pop.Cost]',1);
    for i = 1:N
        if FrontValue(1,i)==1
            pop(i).IsDominated=false;
        else
            pop(i).IsDominated=true;
        end
    end
    % 存储非支配解
    rep = pop(~[pop.IsDominated]);
    % 创建目标空间子空间
    Grid = CreateGrid(rep, nGrid, alpha);
    for i = 1:numel(rep)
        rep(i) = FindGridIndex(rep(i), Grid);
    end
    tic;
    %% 算法开始执行
    for it = 1:Gamma_t*Env+1
        t=fix((it-1)/Gamma_t)/n_t;
        %% 如果环境发生变化，对环境变化做出响应
        if mod((it-1),Gamma_t)==0 && it~=1
            fprintf(' %d',fix(it/Gamma_t));
            res{fix(it/Gamma_t)}.rt = toc;
            res{fix(it/Gamma_t)}.turePOF=getBenchmarkPOF(Problem.Name,group,fix(it/Gamma_t),T_parameter );   % 获取当前测试函数的真实Pareto前沿
            res{fix(it/Gamma_t)}.POF_iter=POF_iter;
            if it == Gamma_t*Env+1
                break;
            end
            tic;
            %% 对环境变化做出响应
            for i = 1:N
                if(i<=numel(rep)&&i<=0.8*N) % 80%的初始种群来自存档，20%的初始种群随机生成
                    pop(i).Position=rep(i).Position;
                else
                    pop(i).Position = unifrnd(VarMin, VarMax, VarSize);
                end
                pop(i).Velocity = zeros(VarSize);
                pop(i).Cost = CostFunction(pop(i).Position);
                pop(i).Best.Position = pop(i).Position;
                pop(i).Best.Cost = pop(i).Cost;
            end
            [FrontValue,~] = NonDominateSort([pop.Cost]',1);
            for i = 1:N
                if FrontValue(1,i)==1
                    pop(i).IsDominated=false;
                else
                    pop(i).IsDominated=true;
                end
            end
            rep = pop(~[pop.IsDominated]);
            Grid = CreateGrid(rep, nGrid, alpha);
            for i = 1:numel(rep)
                rep(i) = FindGridIndex(rep(i), Grid);
            end
        end
        % 环境变化响应完毕
        % 如果环境没有发生变化，则产生子代
        if numel(rep)==0
            global_leader = pop(1);
        else
            global_leader = SelectLeader(rep, beta);
        end
        ielm  = IELM('numberOfInputNeurons', nVar, 'numberOfHiddenNeurons',nGrid);
        X=[];
        Y=[];
        for i = 1:N
            X = [X;pop(i).Position];
            Y = [Y;pop(i).effect];
        end
        ielm  = ielm.train(X,Y);
        Y_pred = ielm.predict(X);
        %增枝
        while(norm(Y_pred-Y,2)>epolision)
            ielm_new  = IELM('numberOfInputNeurons', nVar, 'numberOfHiddenNeurons',ielm.numberOfHiddenNeurons+nGrid);
            ielm_new  = ielm_new.train(X,Y);
            ielm=ielm_new;
            Y_pred = ielm.predict(X);
        end
        %剪枝
        while(norm(Y_pred-Y,2)<1/epolision)
            ielm_new  = IELM('numberOfInputNeurons', nVar, 'numberOfHiddenNeurons',ielm.numberOfHiddenNeurons-1);
            ielm_new  = ielm_new.train(X,Y);
            ielm=ielm_new;
            Y_pred = ielm.predict(X);
        end
        for i = 1:N
            % 选择学习目标
            if numel(rep)==0
                local_leader = pop(i).Best;
            else
                flag = false;
                pop_index = FindGridIndex(pop(i), Grid).GridIndex;
                temp_rep = [];
                for j = 1:numel(rep)
                    if(rep(j).GridIndex==pop_index)
                         temp_rep = [temp_rep rep(j)];
                         flag = true;
                    end
                end
                if(flag)
                    local_leader = SelectLeader(temp_rep, beta);
                else
                    local_leader = SelectLeader(rep, beta);
                end
            end
            %控制加速系数
            output = ielm.predict(pop(i).Position);
            o = find(output == max(output));
            if(o(1)==1)
                pop(i).local_c = pop(i).local_c*(1+eplision);
                pop(i).global_c = 1-pop(i).local_c;
            elseif(o(1)==2)
                pop(i).local_c = pop(i).local_c*(1-eplision);
                pop(i).global_c = 1-pop(i).local_c;
            else
            end
            % 粒子速度更新
            pop(i).Velocity = w*pop(i).Velocity ...
                +pop(i).global_c*(global_leader.Position-pop(i).Position) ...
                +pop(i).local_c.*(local_leader.Position-pop(i).Position);
            % 粒子位置更新
            pop(i).Position = pop(i).Position + pop(i).Velocity;
            pop(i).Position = max(pop(i).Position, VarMin);
            pop(i).Position = min(pop(i).Position, VarMax);
            pop(i).Cost = CostFunction(pop(i).Position);
            pm=(1)^(1/mu);
            if rand<pm
                NewSol.Position=Mutation(pop(i).Position,pm,VarMin,VarMax);
                NewSol.Cost=CostFunction(NewSol.Position);
                if Dominates(NewSol,pop(i))
                    pop(i).Position=NewSol.Position;
                    pop(i).Cost=NewSol.Cost;
                elseif Dominates(pop(i),NewSol)
                else
                    if rand<0.5
                        pop(i).Position=NewSol.Position;
                        pop(i).Cost=NewSol.Cost;
                    end
                end
            end
            % Record evolution effect
            if Dominates(pop(i), pop(i).Best)
                pop(i).Best.Position = pop(i).Position;
                pop(i).Best.Cost = pop(i).Cost;
                pop(i).effect = [1 0 0];
            elseif Dominates(pop(i).Best, pop(i))
                % Do Nothing
                pop(i).effect = [0 0 1];
            else
                pop(i).effect = [0 1 0];
                if rand<0.5
                    pop(i).Best.Position = pop(i).Position;
                    pop(i).Best.Cost = pop(i).Cost;
                end
            end
        end
        POF_iter{mod(it-1,10)+1} = [pop.Cost];
        if mod(it,Gamma_t)==0
            res{fix(it/Gamma_t)}.POS=vertcat(pop.Position)';
        end
        % Add Non-Dominated Particles to REPOSITORY
        [FrontValue,~] = NonDominateSort([pop.Cost]',1);
        for i = 1:N
            if FrontValue(1,i)==1
                pop(i).IsDominated=false;
            else
                pop(i).IsDominated=true;
            end
        end
        rep = [rep
            pop(~[pop.IsDominated])];
        % Determine Domination of New Resository Members
        [FrontValue,~] = NonDominateSort([rep.Cost]',1);
        for i = 1:numel(rep)
            if FrontValue(1,i)==1
                rep(i).IsDominated=false;
            else
                rep(i).IsDominated=true;
            end
        end
        % Keep only Non-Dminated Memebrs in the Repository
        rep = rep(~[rep.IsDominated]);
        % Update Grid
        Grid = CreateGrid(rep, nGrid, alpha);
        % Update Grid Indices
        for i = 1:numel(rep)
            rep(i) = FindGridIndex(rep(i), Grid);
        end
        % Check if Repository is Full
        if numel(rep)>nRep
            Extra = numel(rep)-nRep;
            for e = 1:Extra
                rep = DeleteOneRepMemebr(rep, gamma);
            end
        end
    end
end

function F = EvaluateFrameworkProblem(Problem,x)
global t
[F,~] = Problem.FObj(x,t);
F = real(F(:));
if numel(F) ~= Problem.NObj
    error('PSGN:InvalidObjectiveCount', ...
        '%s returned %d objectives; PSGN expected %d.', ...
        Problem.Name,numel(F),Problem.NObj);
end
end
