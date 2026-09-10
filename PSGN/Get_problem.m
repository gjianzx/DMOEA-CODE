function [CostFunction,GetData,VarMin,VarMax,M] = Get_problem(Problem_ID)
switch Problem_ID
    case 1
        CostFunction = @(x) DF1(x);      % Cost Function
        GetData = @() GetData_DF1();     % Real Pareto Front
        VarMin = [0 0 0 0 0 0 0 0 0 0];          % Lower Bound of Variables
        VarMax = [1 1 1 1 1 1 1 1 1 1];          % Upper Bound of Variables
        M = 2;
    case 2
        CostFunction = @(x) DF2(x);      % Cost Function
        GetData = @() GetData_DF2();     % Real Pareto Front
        VarMin = [0 0 0 0 0 0 0 0 0 0];          % Lower Bound of Variables
        VarMax = [1 1 1 1 1 1 1 1 1 1];          % Upper Bound of Variables
        M = 2;
    case 3
        CostFunction = @(x) DF3(x);      % Cost Function
        GetData = @() GetData_DF3();     % Real Pareto Front
        VarMin = [0 -1 -1 -1 -1 -1 -1 -1 -1 -1]; % Lower Bound of Variables
        VarMax = [1 2 2 2 2 2 2 2 2 2];          % Upper Bound of Variables
        M = 2;
    case 4
        CostFunction = @(x) DF4(x);      % Cost Function
        GetData = @() GetData_DF4();     % Real Pareto Front
        VarMin = [-2 -2 -2 -2 -2 -2 -2 -2 -2 -2]; % Lower Bound of Variables
        VarMax = [2 2 2 2 2 2 2 2 2 2];          % Upper Bound of Variables
        M = 2;
    case 5
        CostFunction = @(x) DF5(x);      % Cost Function
        GetData = @() GetData_DF5();     % Real Pareto Front
        VarMin = [0 -1 -1 -1 -1 -1 -1 -1 -1 -1]; % Lower Bound of Variables
        VarMax = [1 1 1 1 1 1 1 1 1 1];        % Upper Bound of Variables
        M = 2;
    case 6
        CostFunction = @(x) DF6(x);      % Cost Function
        GetData = @() GetData_DF6();     % Real Pareto Front
        VarMin = [0 -1 -1 -1 -1 -1 -1 -1 -1 -1]; % Lower Bound of Variables
        VarMax = [1 1 1 1 1 1 1 1 1 1];        % Upper Bound of Variables
        M = 2;
    case 7
        CostFunction = @(x) DF7(x);      % Cost Function
        GetData = @() GetData_DF7();     % Real Pareto Front
        VarMin = [1 0 0 0 0 0 0 0 0 0];          % Lower Bound of Variables
        VarMax = [4 1 1 1 1 1 1 1 1 1];          % Upper Bound of Variables
        M = 2;
    case 8
        CostFunction = @(x) DF8(x);      % Cost Function
        GetData = @() GetData_DF8();     % Real Pareto Front
        VarMin = [0 -1 -1 -1 -1 -1 -1 -1 -1 -1]; % Lower Bound of Variables
        VarMax = [1 1 1 1 1 1 1 1 1 1];        % Upper Bound of Variables
        M = 2;
    case 9
        CostFunction = @(x) DF9(x);      % Cost Function
        GetData = @() GetData_DF9();     % Real Pareto Front
        VarMin = [0 -1 -1 -1 -1 -1 -1 -1 -1 -1]; % Lower Bound of Variables
        VarMax = [1 1 1 1 1 1 1 1 1 1];        % Upper Bound of Variables
        M = 2;
    case 10
        CostFunction = @(x) DF10(x);      % Cost Function
        GetData = @() GetData_DF10();     % Real Pareto Front
        VarMin = [0 0 -1 -1 -1 -1 -1 -1 -1 -1];  % Lower Bound of Variables
        VarMax = [1 1 1 1 1 1 1 1 1 1];          % Upper Bound of Variables
        M = 3;
    case 11
        CostFunction = @(x) DF11(x);      % Cost Function
        GetData = @() GetData_DF11();     % Real Pareto Front
        VarMin = [0 0 0 0 0 0 0 0 0 0];          % Lower Bound of Variables
        VarMax = [1 1 1 1 1 1 1 1 1 1];          % Upper Bound of Variables
        M = 3;
    case 12
        CostFunction = @(x) DF12(x);      % Cost Function
        GetData = @() GetData_DF12();     % Real Pareto Front
        VarMin = [0 0 -1 -1 -1 -1 -1 -1 -1 -1];  % Lower Bound of Variables
        VarMax = [1 1 1 1 1 1 1 1 1 1];          % Upper Bound of Variables
        M = 3;
    case 13
        CostFunction = @(x) DF13(x);      % Cost Function
        GetData = @() GetData_DF13();     % Real Pareto Front
        VarMin = [0 0 -1 -1 -1 -1 -1 -1 -1 -1];  % Lower Bound of Variables
        VarMax = [1 1 1 1 1 1 1 1 1 1];          % Upper Bound of Variables
        M = 3;
    case 14
        CostFunction = @(x) DF14(x);      % Cost Function
        GetData = @() GetData_DF14();     % Real Pareto Front
        VarMin = [0 0 -1 -1 -1 -1 -1 -1 -1 -1];  % Lower Bound of Variables
        VarMax = [1 1 1 1 1 1 1 1 1 1];          % Upper Bound of Variables
        M = 3;
    otherwise
        disp('other value')
end
end

