% function demoExtremelyFastNCut_MNIST

% A demonstration of FastESC algorithm as described in [1]. To run this
% code, please:
%
% 1. download EMNIST data set
% http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip
%
% 2. Unzip and set pathEMNIST as your path of the file emnist-digits.mat
% 
% 3. Run this demo
%
% [1] Li He, Nilanjan Ray and Hong Zhang, Fast Large-Scale Spectral 
% Clustering via Explicit Feature Mapping, submitted to IEEE Trans.
% Cybernetics.
%
% Li He, heli@gdut.edu.cn

clear
clc
close all

%% 0. Initialization
% path of emnist-digits.mat
pathEMNIST = 'E:\MatlabWorks\DataSet\EMNIST\matlab\emnist-digits.mat';

% load data
load(pathEMNIST); % dataset
data = double(dataset.train.images);
labels = dataset.train.labels;
clear dataset;

% number of clusters
numCentrs = length(unique(labels));

% set sigma as the mean distance of the leading 3K data points
dis = pdist2_my(data(1:3000,:),data(1:3000,:));
dis = sqrt(dis);
sigma = mean(dis(:));

% for plot
colorList = 'rgbm';
labelList = [0 1 2 9];

%% 1. FastESC
D = 1200; % dimension of EFM
flagGPU = false; % true for using GPU
ts = tic;
v = FastESC_LargeScale(data, sigma, D, numCentrs, flagGPU);
tFastESC = toc(ts);
disp(['FastESC time cost: ' num2str(tFastESC) ' seconds']);

%% 2. Plot
figure(1);hold on;
set(gca,'Position',[0 0 1 1]);axis off

% plot digits 0, 1, 2 and 9
for i=1:4
    idx = labels==labelList(i);
    % the 1st eigvec in v is the trivial solution in NCut and is abandoned
    plot(v(idx,2),v(idx,3),'.','Color',colorList(i),'MarkerSize',4);
end
legend('0','1','2','9');