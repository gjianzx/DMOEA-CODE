clc;clear;
load iris_dataset.mat
X      = irisInputs';
Y      = irisTargets';

X=rand(150,4)-0.5;
Y=sin(sum(X,2));
oselm  = OSELM('numberOfInputNeurons', 4, 'numberOfHiddenNeurons',9);
tic;
oselm  = oselm.train(X, Y);
Yhat   = oselm.predict(X);
zzzz=sqrt(mean(mean((Y - Yhat).^2)));
oselm1  = OSELM('numberOfInputNeurons', 4, 'numberOfHiddenNeurons',9);
tic;
for i =1:10:size(X,1)
    oselm1  = oselm1.train(X(i:i+9,:), Y(i:i+9,:));
    Yhat(i:i+9,:)   = oselm1.predict(X(i:i+9,:));
    sqrt(mean(mean((Y(i:i+9,:) - Yhat(i:i+9,:)).^2)))g
end
Yhat   = oselm1.predict(X);
zzzz1=sqrt(mean(mean((Y - Yhat).^2)));