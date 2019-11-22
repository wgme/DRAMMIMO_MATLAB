close all; clear; clc; format short g;
%% Below is an example of using the DRAMMIMO package.

% Two simple linear models y1 = a * x1 + b and y2 = a * x2 + b are considered.
% These two models share parameters a and b, such that the Maximum Entropy
% method can be taken advantage of.

%% Load the data.

disp('Loading data...');

% Two sets of ficticious data are generated using the models with Gaussian 
% noise, i.e. y1 = a1 * x1 + b + epsilon1 and y2 = a2 * x2 + b + epsilon2.
% A discrepancy is introduced between a1 = 0.8 and a2 = 1.2, as well as 
% different magnitudes of noise, epsilon1 = 0.05 and epsilon2 = 0.10.
inputData1 = linspace(0, 1, 101)';
inputData2 = linspace(0, 1, 101)';
outputData1 = 0.8 * inputData1 + 0.05 * randn(101, 1);
outputData2 = 1.2 * inputData2 + 0.10 * randn(101, 1);

%% Set up the DRAMMIMO.

disp('Setting DRAMMIMO...');

% Set the data struct.
% Add however many sets of data here. Just make sure .xdata and .ydata
% have the same length.
% This example has two data sets.
% .xdata contains the input data.
data.xdata = {inputData1, inputData2};
% .ydata contains the output data.
data.ydata = {outputData1, outputData2};

% Set the model struct.
% Add however many number of models here. Just make sure the number matches 
% the number of data sets.
% This example has two models.
% .fun contains the functions that can generate model predictions.
model.fun = {@getModelResponse, @getModelResponse};
% .errFun contains the functions that compare model predictions with data.
model.errFun = {@getModelResponseError, @getModelResponseError};

% Set the modelParams struct.
% .table = {parameter name, initial value, lower limit, upper limit}.
modelParams.table = {{'a', 1, -inf, inf}, ...
                     {'b', 0, -inf, inf}};
% .extra can pass extra parameter values that are not being estimated to 
% each model. Empty cells if not necessary.
modelParams.extra = {{0}, {0}};

% Set the DRAMParams struct.
% Number of iterations that are already done.
DRAMParams.numIterationsDone = 1;
% Number of iterations that are expected to be done.
DRAMParams.numIterationsExpected = 10000;
% Every X number of iterations, display the parameter values at this 
% iteration in the command window.
DRAMParams.numIterationsDisplay = 200;
% Every X number of iterations, save the estimation chains up to this 
% iteration to a .mat file in current folder.
DRAMParams.numIterationsSave = 1000;
% For initial run, the .previousResults struct is empty.
DRAMParams.previousResults.prior.psi_s = [];
DRAMParams.previousResults.prior.nu_s = [];
DRAMParams.previousResults.chain_q = [];
DRAMParams.previousResults.last_cov_q = [];
DRAMParams.previousResults.chain_cov_err = [];

%% Run the DRAMMIMO.

% The uncertainty quantification results consist of three parts:
% 1. Estimation chains.
% 2. Posterior densities.
% 3. Credible and prediction intervals.

disp('Running DRAMMIMO...');

% Get the estimation chains.
[prior, chain_q, last_cov_q, chain_cov_err] = ...
    getDRAMMIMOChains(data, model, modelParams, DRAMParams);

% The estimation chains can be obtained in multiple runs.
% Uncomment the following portion of code to have a 2nd run.
% Remember to adjust the DRAMParams struct accordingly.
% -------------------------------------------------------------------------
% DRAMParams.numIterationsDone = 5000;
% DRAMParams.numIterationsExpected = 10000;
% DRAMParams.numIterationsDisplay = 200;
% DRAMParams.numIterationsSave = 1000;
% DRAMParams.previousResults.prior.psi_s = prior.psi_s;
% DRAMParams.previousResults.prior.nu_s = prior.nu_s;
% DRAMParams.previousResults.chain_q = chain_q;
% DRAMParams.previousResults.last_cov_q = last_cov_q;
% DRAMParams.previousResults.chain_cov_err = chain_cov_err;
% [prior, chain_q, last_cov_q, chain_cov_err] = ...
%     getDRAMMIMOChains(data, model, modelParams, DRAMParams);
% -------------------------------------------------------------------------

% Get the posterior densities.
% Assume the second half of the chains are in steady-state, but this 
% number is not necessarily to be true.
num = round(size(chain_q,1)/2)+1;
[vals,probs] = getDRAMMIMODensities(chain_q(num:end, :));

% Get the credible and prediction intervals.
% 500 is the rule of thumb number, and this number is suggested to be fixed.
nSample = 500;
[credLims,predLims] = ...
    getDRAMMIMOIntervals(data, model, modelParams, ...
                         chain_q(num:end,:),chain_cov_err(:,:,num:end),...
                         nSample);

%% Display the results.

disp('Presenting results...');

% Mean parameter estimation.
disp('Mean Parameter Estimation = ');
disp(mean(chain_q(num:end,:)));

% Plot the results.
figNum = 0;

% Raw data.
figNum = figNum+1;
fh = figure(figNum);
set(fh,'outerposition',96*[2,2,7,6]);
hold on;
h(1) = plot(inputData1,outputData1,'bo');
h(2) = plot(inputData2,outputData2,'ro');
hold off;
box on;
set(gca,'fontsize',24,'xlim',[0,1],'ylim',[-0.1,1.3]);
xlabel('x');
ylabel('y');
legend(h,'Data I','Data II','location','nw');
legend boxoff;

% Estimation chains.
figNum = figNum+1;
fh = figure(figNum);
set(fh,'outerposition',96*[2,2,7,6]);
subplot(2,1,1);
hold on;
plot(1:1:size(chain_q,1),chain_q(:,1),'b.');
hold off;
set(gca,'fontsize',24,'xtick',[],...
    'xlim',[0,DRAMParams.numIterationsExpected],'ylim',[min(chain_q(:,1)),max(chain_q(:,1))]);
box on;
ylabel('a');
subplot(2,1,2);
hold on;
plot(1:1:size(chain_q,1),chain_q(:,2),'b.');
hold off;
set(gca,'fontsize',24,'xtick',[],...
    'xlim',[0,DRAMParams.numIterationsExpected],'ylim',[min(chain_q(:,2)),max(chain_q(:,2))]);
box on;
ylabel('b');
xlabel('Iterations');

% Posterior densities.
figNum = figNum+1;
fh = figure(figNum);
set(fh,'outerposition',96*[2,2,7,6]);
subplot(1,2,1);
hold on;
plot(vals(:,1),probs(:,1),'k','linewidth',3);
hold off;
set(gca,'fontsize',24,'xlim',[min(vals(:,1)),max(vals(:,1))],'ytick',[]);
box on;
xlabel('a');
ylabel('Posterior Density');
subplot(1,2,2);
hold on;
plot(vals(:,2),probs(:,2),'k','linewidth',3);
hold off;
set(gca,'fontsize',24,'xlim',[min(vals(:,2)),max(vals(:,2))],'ytick',[]);
box on;
xlabel('b');

% Credible and prediction interval for Data I.
figNum = figNum+1;
fh = figure(figNum);
set(fh,'outerposition',96*[2,2,7,6]);
set(gca,'fontsize',24,'xlim',[0,1],'ylim',[-0.7,2]);
hold on;
h(1) = patch([data.xdata{1}',fliplr(data.xdata{1}')],...
      [predLims(1,:,1),fliplr(predLims(3,:,1))],...
      [1,0.75,0.5],'linestyle','none');
h(2) = patch([data.xdata{1}',fliplr(data.xdata{1}')],...
      [credLims(1,:,1),fliplr(credLims(3,:,1))],...
      [0.75,1,0.5],'linestyle','none');
h(3) = plot(inputData1,credLims(2,:,1),'k');
h(4) = plot(inputData1,outputData1,'bo');
hold off;
box on;
lh = legend(h,'95% Pred Interval','95% Cred Interval','Model','Data I','location','nw');
lh.FontSize = 18;
legend boxoff;
xlabel('x');
ylabel('y_1');

% Credible and prediction interval for Data II.
figNum = figNum+1;
fh = figure(figNum);
set(fh,'outerposition',96*[2,2,7,6]);
set(gca,'fontsize',24,'xlim',[0,1],'ylim',[-0.7,2]);
hold on;
h(1) = patch([data.xdata{2}',fliplr(data.xdata{2}')],...
      [predLims(1,:,2),fliplr(predLims(3,:,2))],...
      [1,0.75,0.5],'linestyle','none');
h(2) = patch([data.xdata{2}',fliplr(data.xdata{2}')],...
      [credLims(1,:,2),fliplr(credLims(3,:,2))],...
      [0.75,1,0.5],'linestyle','none');
h(3) = plot(inputData1,credLims(2,:,2),'k');
h(4) = plot(inputData2,outputData2,'ro');
hold off;
box on;
lh = legend(h,'95% Pred Interval','95% Cred Interval','Model','Data II','location','nw');
lh.FontSize =18;
legend boxoff;
xlabel('x');
ylabel('y_2');