function [sampley] = F_gp(x_tr, y_tr, popsize, finalgen, x_tt)
%function to train and sample a GP model
%
% This wrapper adds lightweight sanitization and stable fallbacks around
% the original GPML call so ill-conditioned training data in dynamic
% environments does not break prediction.

    %#ok<INUSD>
    ns = size(x_tt,1);
    sampley = zeros(ns,1);
    if ns == 0
        return;
    end

    x_tr = real(x_tr);
    y_tr = real(y_tr(:));
    x_tt = real(x_tt);

    validTrain = all(isfinite(x_tr),2) & isfinite(y_tr);
    validTest = all(isfinite(x_tt),2);
    x_tr = x_tr(validTrain,:);
    y_tr = y_tr(validTrain);

    if isempty(x_tr)
        return;
    end

    sampley(~validTest) = mean(y_tr);
    if ~any(validTest)
        return;
    end

    x_tt_valid = x_tt(validTest,:);
    [x_tr, y_tr] = localDeduplicateTrainSet(x_tr,y_tr);

    if size(x_tr,1) < 2 || max(y_tr) - min(y_tr) < 1e-10
        sampley(validTest) = repmat(mean(y_tr),size(x_tt_valid,1),1);
        return;
    end

    [x_tr_scaled, x_tt_scaled] = localScaleFeatures(x_tr,x_tt_valid);

    meanfunc = @meanZero;
    covfunc = @covLIN;
    likfunc = @likGauss;
    hyp.mean = [];
    hyp.cov = [];

    gpSucceeded = false;
    noiseLevels = [0.01, 0.05, 0.1, 0.5, 1.0];
    for k = 1:numel(noiseLevels)
        hyp.lik = log(noiseLevels(k));
        try
            [m,s2] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, ...
                x_tr_scaled, y_tr, x_tt_scaled);
            if all(isfinite(m)) && all(isfinite(s2))
                gpSucceeded = true;
                break;
            end
        catch
        end
    end

    if gpSucceeded
        if finalgen == 1
            y_tt = m;
        else
            y_tt = m + rand()*sqrt(max(s2,0)).*normrnd(0,1,[size(x_tt_scaled,1),1]);
        end
    else
        y_tt = localLinearFallback(x_tr_scaled,y_tr,x_tt_scaled);
    end

    y_tt(~isfinite(y_tt)) = mean(y_tr);
    sampley(validTest) = y_tt;
end

function [xUnique,yUnique] = localDeduplicateTrainSet(x_tr,y_tr)
    [xUnique,~,groupIdx] = unique(x_tr,'rows','stable');
    yUnique = accumarray(groupIdx,y_tr,[],@mean);
end

function [xTrainScaled,xTestScaled] = localScaleFeatures(xTrain,xTest)
    mu = mean(xTrain,1);
    sigma = std(xTrain,0,1);
    sigma(~isfinite(sigma) | sigma < 1e-8) = 1;
    xTrainScaled = (xTrain - mu)./sigma;
    xTestScaled = (xTest - mu)./sigma;
end

function yPred = localLinearFallback(xTrain,yTrain,xTest)
    nFeat = size(xTrain,2);
    designTrain = [ones(size(xTrain,1),1), xTrain];
    designTest = [ones(size(xTest,1),1), xTest];
    ridge = 1e-8;
    beta = (designTrain'*designTrain + ridge*eye(nFeat+1)) \ (designTrain'*yTrain);
    yPred = designTest*beta;
end
