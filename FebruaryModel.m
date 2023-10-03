clear all; clc; close all;

%only importing Februray 2019 data
file='February 2019.xlsx';
opts1 = detectImportOptions(file);
opts1.SelectedVariableNames = {'Var1','Irradiation_W_m2_','ModuleTemperature___C_','AmbientTemperature___C_','TotalPac'};

Feb = readtable(file,opts1);
[filterFeb ia] = rmmissing(Feb);
FebCheck = ia(ia(:,1)==1); % expecting 27x3 = 81 rows to be removed

h = hour(filterFeb.Var1);
m = minute(filterFeb.Var1);

tabFeb = filterFeb((h >= 7) & (h < 20), :);
matrixFeb = tabFeb(:,2:end);
datasetFeb = table2array(matrixFeb);
[row col] = size(datasetFeb);

% normalize between range 0 to 1
dataNormalized = normalize(datasetFeb,"range"); 
maxValues = max(datasetFeb);
minValues = min(datasetFeb);

% extracting the features from the dataset
irradiation = dataNormalized(:,1);
moduleTemp = dataNormalized(:,2);
ambientTemp = dataNormalized(:,3);
totalPower = dataNormalized(:,end);

all_days = tabFeb.Var1.Day;

%to sort the data based on the dates
 for i = 1:max(all_days)
     ind = (all_days ==i);
     rows = find(any(ind==1,2));
     days{i} = tabFeb(rows(1):rows(end),:);
 end
 
 numTimeStepsDay = height(days{1});
%%
% database creation for LSTM inputs
numDays = 4; % number of days
sequenceLength = numDays*numTimeStepsDay; % number of time steps used in each sample
database = {};
powerDatabase = {};
resultDatabase = [];

count = 0;
for i = 1:1:row-sequenceLength
    count = count+1;
    a = irradiation(i:i+sequenceLength-1)'; % first feature (most important) which is also desired output
    b = moduleTemp(i:i+sequenceLength-1)'; % second feature
    %c = ambientTemp(i:i+sequenceLength-1)'; % third feature
    database{count,1} = [a;b];
    % database{count,1}=[a;b];
end

[rt ct] = size(database);

% database creation for LSTM outputs
count = 0;
for i = 1:1:row-sequenceLength
    count = count+1;
    d = totalPower(i:i+sequenceLength-1)'; % solar PV power
    powerDatabase{count,1}=[d];
end

count = 0;
for i=1:rt-1
    count = count+1;
    e = powerDatabase{count+1,1};
    resultDatabase(count,1) = e(1,end);
end
%%
% segmenting the database into training, validation and testing sets
% training : 85%
% validation : 15%
% testing : 1 days

numTimeStepsTest = numTimeStepsDay*1;
numTimeStepsTrain = round(0.85*(rt-numTimeStepsTest));
numTimeStepsVal = rt-numTimeStepsTest-numTimeStepsTrain;

%%
XTrain = {};
YTrain = [];

for i=1:numTimeStepsTrain+1
    XTrain{i,1}=database{i,1}; % training set predictors (inputs)
end

YTrain = resultDatabase(1:numTimeStepsTrain+1); % training set responses (outputs)


% LSTM Network Creation
numFeatures = 2;
numHiddenLayers = 50;
numResponses = 1;

Layers = [sequenceInputLayer(numFeatures),...
    lstmLayer(numHiddenLayers,'OutputMode','sequence'),...
    dropoutLayer(0.6),...
    lstmLayer(numHiddenLayers,'OutputMode','last'),...
    dropoutLayer(0.6),...
    fullyConnectedLayer(numResponses),...
    regressionLayer];

miniBatchSize = 16;
Epoch = 3;

options = trainingOptions ('adam',...
    'ExecutionEnvironment','auto',...
    'MaxEpochs',Epoch,...
    'MiniBatchSize',miniBatchSize,...
    'InitialLearnRate',0.01,...
    'Plots','training-progress');
%%
% initialises the network to predict t+1 response given the inputs (only 1
% timestep forward output)
Casedrop60 = trainNetwork(XTrain,YTrain,Layers,options);

if exist('Casedrop60.mat','file') 
    fprintf('The Variables Exist. Loading...\n'); 
else fprintf('Variables Do Not Exist. Creating Variables...\n'); 
    save Casedrop60
end
%%
% validation stage

% update network state with observed values
count = 0;
for i=numTimeStepsTrain+1:numTimeStepsVal+numTimeStepsTrain
    count = count+1;
    XValidation{count,1} = database{i,1}; % testing set predictors (inputs)
end

YValidation = resultDatabase(numTimeStepsTrain+2:numTimeStepsVal+numTimeStepsTrain);

[UpdatedNet1 YPredVal] = predictAndUpdateState(Casedrop60,XValidation,'MiniBatchSize',16,'ExecutionEnvironment','cpu');

% Ypred predicts an additional timestep in the future past that of the last
% time of Feb

% unnormalize the prediction
YPredValActual = YPredVal.*(maxValues(end)-minValues(end)) + minValues(end);
YValidationActual = YValidation.*(maxValues(end)-minValues(end)) + minValues(end);
YTrainActual = YTrain.*(maxValues(end)-minValues(end)) + minValues(end);

n = length(YValidation);

% performance metric results:

% normalised
nMAE = abs((sum(YValidation-YPredVal(1:end-1)))/n)
nMSE = (sum((YValidation - YPredVal(1:end-1)).^2))/n
nRMSE = sqrt(nMSE)
nMaxError =  max(YValidation-YPredVal(1:end-1))
ei = YValidation-YPredVal(1:end-1);
SSR = sum(ei.^2);
SST = sum((YValidation-mean(YValidation)).^2);

% un-normalised
MAE = abs((sum(YValidationActual-YPredValActual(1:end-1)))/n)
MSE = (sum((YValidationActual - YPredValActual(1:end-1)).^2))/n
RMSE = sqrt(MSE)
MaxError =  max(YValidationActual-YPredValActual(1:end-1))
%ei = YValidationActual-YPredValActual(1:end-1);
%SSR = sum(ei.^2);
%SST = sum((YValidationActual-mean(YValidationActual)).^2);
%nRSquare = 1-(SSR/SST)
RSquare = 1-(SSR/SST)

% graphical results:

% plot the comparison between actual and forecasted values
figure 
hold on
plot(YValidationActual,'b')
plot(YPredValActual,'r')
xlabel("Timesteps")
ylabel("Solar PV Power")
title("Forecast on Test Data Un-normalised")
legend(["Observed" "Forecast"])
hold off

% plot the RMSE graph
figure
stem(YValidationActual - YPredValActual(1:end-1))
xlabel("Timestep")
ylabel("Error")
title("RMSE")
title("RMSE = " + RMSE)

% plot the MSE graph
figure
stem((YPredValActual(1:end-1) - YValidationActual).^2)
xlabel("Timestep")
ylabel("Error")
title("MSE")
title("MSE = " + MSE)

% plot the R^2 graph
figure
plot(YValidation,YPredVal(1:end-1),'*')
hold on
xlabel('Target')
ylabel('Output')
p = polyfit(YValidation,YPredVal(1:end-1),1);
f = polyval(p,YValidation);
plot(YValidation,f,'r')
title(sprintf('Regression line y = %0.2f*Target + %0.2f',p(1),p(2)))
hold off
%%
% % Testing Stage
% 
% update network state with observed values
count = 0;
for i=numTimeStepsTrain+numTimeStepsVal+1:rt
    count = count+1;
    XTest{count,1} = database{i,1}; % testing set predictors (inputs)
end

YTest = resultDatabase(numTimeStepsTrain+numTimeStepsVal+1:rt-1);

[UpdatedNet2 YPredTest] = predictAndUpdateState(Case50,XTest,'MiniBatchSize',16,'ExecutionEnvironment','cpu')

% Ypred predicts an additional timestep in the future past that of the last
% time of Feb

% unnormalize the prediction
YPredTestActual = YPredTest.*(maxValues(end)-minValues(end)) + minValues(end);
YTestActual = YTest.*(maxValues(end)-minValues(end)) + minValues(end);

n = length(YTest);

% performance metric results:

% normalised
nMAE = abs((sum(YTest-YPredTest(1:end-1)))/n)
nMSE = (sum((YTest - YPredTest(1:end-1)).^2))/n
nRMSE = sqrt(nMSE)
nMaxError =  max(YTest-YPredTest(1:end-1))
ei = YTest-YPredTest(1:end-1);
SSR = sum(ei.^2);
SST = sum((YTest-mean(YTest)).^2);


% un-normalised
MAE = abs((sum(YTestActual-YPredTestActual(1:end-1)))/n)
MSE = (sum((YTestActual - YPredTestActual(1:end-1)).^2))/n
RMSE = sqrt(MSE)
MaxError =  max(YTestActual-YPredTestActual(1:end-1))
% ei = YTestActual-YPredTestActual(1:end-1);
% SSR = sum(ei.^2);
% SST = sum((YTestActual-mean(YTestActual)).^2);
% nRSquare = 1-(SSR/SST)
RSquare = 1-(SSR/SST)

% graphical results:

% plot the comparison between actual and forecasted Testues
figure 
hold on
plot(YTestActual,'b')
plot(YPredTestActual,'r')
xlabel("Timesteps")
ylabel("Solar PV Power")
title("Forecast on Test Data Un-normalised")
legend(["Observed" "Forecast"])
hold off

% plot the RMSE graph
figure
stem(YTestActual - YPredTestActual(1:end-1))
xlabel("Timestep")
ylabel("Error")
title("RMSE")
title("RMSE = " + RMSE)

% plot the MSE graph
figure
stem((YPredTestActual(1:end-1) - YTestActual).^2)
xlabel("Timestep")
ylabel("Error")
title("MSE")
title("MSE = " + MSE)

% plot the R^2 graph
figure
plot(YTest,YPredTest(1:end-1),'*')
hold on
xlabel('Target')
ylabel('Output')
p = polyfit(YTest,YPredTest(1:end-1),1);
f = polyval(p,YTest);
plot(YTest,f,'r')
title(sprintf('Regression line y = %0.2f*Target + %0.2f',p(1),p(2)))
hold off
% %%
% % % plot the training time series with the forecasted values
% % % (Validation+Test)
% % figure
% % trainData = datasetFeb(1:numTimeStepsTrain+1,end);
% % plot(trainData)
% % hold on
% % %temp1 = length(trainData)
% % %idx = temp+1:temp+numTimeStepsVal
% % idxValidation = (numTimeStepsTrain+2:numTimeStepsVal+numTimeStepsTrain+1);
% % plot(idxValidation,YPredValActual(1:end),'m-')
% % idxTest = (numTimeStepsTrain+numTimeStepsVal+2:rt);
% % plot(idxTest,YPredTestActual(1:end),'g-')
% % hold off
% % legend('training set','validation set','test set')