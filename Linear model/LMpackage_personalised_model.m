
clc
close all
clear

% Basic example demonstrating how to use the LM package to fit a linear
% backward model reconstructing the stimulus based on EEG data.
%
tic


allSID = {'P1','P2','P3','P4','P5','P6','P7','P8','P9','P10','P11','P12','P13'};

parts = 1:15;
Fs = 50;

nChan = 64;
 


% Time region in which to derive the decoder. Time lag is understood as lag of
% predictor (here EEG) with respect to predicted data (here stimulus).
% Hence, here positive time lags correspond to the causal part of the
% decoder (response after stimulus).
minLagT = -500e-3;
maxLagT = 500e-3;

% estimate performance (CC & MSE) on windows of this duration
% (negative = use all available data)
tWinPerf = [2]; % in seconds

% stimOpt and EEGopt are arbitray multi-dimensional variables (e.g.
% matrices, cell, structures) containing the required information to load
% stimulus and EEG data respectively. Each element of stimOpt and EEGopt
% will be passed to user-defined loading function (see below) that will
% load the data accordingly. Hence arbitray parameters can be passed to the
% loading functions.
%
% In this example, we will simply store the path of the files to load in
% cell arrays.
nParts = numel(parts);
nSub = numel(allSID);
stimOpt = cell(nParts,1);
% EEGopt have to be of size: [size(stimOpt),nSub], i.e. each stimulus file
% corresponds to 1 EEG recording per subject.
EEGopt = cell(nParts,nSub);

% load channel location
chanLocs = LM_example_loadChanLocs();
% define a channel order to be used for all data
chanOrder = {chanLocs(:).labels};
% 
% for iPart = 1:nParts
%     envFileName = sprintf('env_Fs-%i-%s-%s_%s_%i.mat',Fs,procEnv,typeEnv,condition,iPart);
%     stimOpt{iPart} = fullfile(pwd(),'envelopes',envFileName);
%     
%     for iSub = 1:nSub
%         EEGFolder = fullfile(pwd(),'EEGdata',allSID{iSub});
%         EEGFileName = sprintf('%s-Fs-%i-%s_%s_%i.set',procEEG,Fs,allSID{iSub},condition,iPart);
%         
%         EEGopt{iPart,iSub} = {EEGFolder,EEGFileName,chanOrder};
%     end
% end

for iPart = 1:nParts
    envFileName = sprintf('Envelope_both_%i.mat',iPart);
    stimOpt{iPart} = fullfile('\Desktop\Project\data\Envelope',envFileName);%fullfile('\rds\general\user\as2719\home\Envelope',envFileName);
    
    for iSub = 1:nSub
        EEGFolder = fullfile('\Desktop\Project\data\EEGdata',allSID{iSub});%fullfile('\rds\general\user\as2719\home\EEGdata',allSID{iSub});
        EEGFileName = sprintf('New _Participant_%i_%d_EEG_both.mat',iSub,iPart);
        
        EEGopt{iPart,iSub} = {EEGFolder,EEGFileName,chanOrder};
    end
end



% options passed to the call to get the appropriate matrices to fit the
% linear model
opt = struct();
opt.nStimPerFile = 1;
% These are loading function taking one element of stimOpt and EEGopt
% respectively as input, and loading stimulus / EEG data.
opt.getStimulus = @LoadFeature;
% This function should return as 1st output a [nPnts x nChan] data matrix,
% and as 2nd outut a vector of indices (size nStimPerFile x 1) indicating
% where each stimulus begins in the data. These indices should be sorted in
% the same order as the stimuli returned by opt.getStimulus.
opt.getResponse = @LoadEEG;

% nb of features describing each stimulus
opt.nFeatures = 1;
% nb of channels in the EEG data
opt.nChan = nChan;

% converting lags for time to indices
opt.minLag = floor(minLagT * Fs);
opt.maxLag = ceil(maxLagT * Fs);

opt.sumSub = false;
opt.sumStim = false; % does not matter here
opt.sumFrom = 1; % sum over parts

% false: the predictor data (here stimulus) will be zeros padded at its
% edges. true: no padding.
opt.unpad.do = false;
% removing means = fitting models without offsets
opt.removeMean = true;

% convert to samples
opt.nPntsPerf = ceil(tWinPerf*Fs)+1;


nLags = opt.maxLag - opt.minLag + 1;

% options to fit the model
trainOpt = struct();
trainOpt.method.name = 'ridge-eig-XtX'; % use ridge regression
% regularisation coefficients for which we'll fit the model
trainOpt.method.lambda = 10.^(-6:0.1:6);
trainOpt.method.normaliseLambda = true;
trainOpt.accumulate = true; % the input is XtX & Xty, and not X & y

nLambda = numel(trainOpt.method.lambda);
nPerfSize = numel(tWinPerf);

% We will fit subject specific models using a leave-one-part-out
% cross-validation procedure. For each subject, a model will be fitted over
% all the data bar one part. The excluded part subject will be used as
% testing data.

% Testing value sets for each window duration in tWinPerf, each data part
% and each subject (number of values in each set will depend on the length
% of each data part, and duration of each testing window).
CC = cell(nPerfSize,nParts,nSub);
MSE = cell(nPerfSize,nParts,nSub);

for iTestPart = 1:nParts
    
    iTrainParts = [1:(iTestPart-1),(iTestPart+1):nParts];
    
    for iSub = 1:nSub

        % model fitted using only training parts for iSub
        [XtX_train,Xty_train] = LM_crossMatrices(stimOpt(iTrainParts),EEGopt(iTrainParts,iSub),opt,'backward');

        model_train = LM_fitLinearModel(XtX_train,Xty_train,trainOpt);
        model_train = model_train.coeffs;
        
        % testing on the remaining part
        stim_test = stimOpt(iTestPart);
        EEG_test = EEGopt(iTestPart,iSub);
        
 
        [ CC(:,iTestPart,iSub),...
            MSE(:,iTestPart,iSub)] = LM_testModel(model_train,stim_test,EEG_test,opt,'backward');
    end
end
save('supersafe','CC','MSE')
save('realCC','CC','MSE','stand','maxCC')
toc
%%
% looking at the data using 10s slices
dur0 = 2;
iDur0 = find(tWinPerf == dur0,1);
C1 = CC(iDur0,:,:);


%method to find standard deviation for specific way mentioned in the paper
for i = 1:13
    ct = C1(:,:,i); 
    for j = 1:15
        ctt = ct{1,j};
        mxvl = ctt(:,:,iMax(i));
        avg(j) = mean(mxvl);
    stand(i) = std(avg);
    end
end
        

% pooling all the testing results for windows of duration dur0 
CC0 = vertcat(CC{iDur0,:});
nWin = size(CC0,1) / nSub;
CC0 =  reshape(CC0,[nWin,nSub,nLambda]);
mCC = squeeze(mean(CC0,1))';
[maxCC,iMax] = max(mCC,[],1);


% regularisation curve for each subject
figure;
ax = axes();
plot(trainOpt.method.lambda ,mCC);
ax.XAxis.Scale = 'log';
ax.XAxis.Label.String = '\lambda_n';
ax.YAxis.Label.String = 'Correlation coefficient';
ax.Title.String = 'Regularisation curve for each subject';
legend(subs)
subs = []
for i = 1:13
    butt = sprintf("Subject %d", i);
    subs = [subs, butt];
end


% best CC for each subject
[maxCC,iMax] = max(mCC,[],1);
stdCC = arrayfun(@(iSub,iMax) std(CC0(:,iSub,iMax),[],1),1:nSub,iMax);

[maxCC,iSort] = sort(maxCC,'ascend');
stdCC = stdCC(iSort);

figure;
ax = axes();
errorbar(1:nSub,maxCC,stand,'ko');
ax.XAxis.Label.String = 'Subject #';
ax.YAxis.Label.String = 'Correlation coefficient';
ax.Title.String = 'Sorted best correlation coefficients';
ax.XAxis.Limits = [0,nSub+1];
% The decoder obtained at e.g. the best CC for each subject could then be
% used on some held out data.
%
%