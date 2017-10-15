function demoFastESC
% A demonstration of FastESC algorithm as described in [1]. FastESC is an
% analogue to Normlized Cuts (NCut) and generates the approximated
% eigenvectors that NCut outputs.
% 
% [1] Li He, Nilanjan Ray and Hong Zhang, Fast Large-Scale Spectral 
% Clustering via Explicit Feature Mapping, submitted to IEEE Trans.
% Cybernetics.
%
% Li He, heli@gdut.edu.cn

clear
clc
close all

% load data
load ./Iris.mat; % Iris data set: http://archive.ics.uci.edu/ml/datasets/Iris

% number of clusters
numCentrs = length(unique(labels));

% set sigma
dis = pdist2_my(data,data);
dis(dis<0) = 0;
sigma = mean(sqrt(dis(:)));

% demension of EFM, see [1] for details
D = 500;

%% 1. FastESC
vESC = FastESC(data, sigma, D, numCentrs);
%% 2. NCut
vNCut = myNCut(data, sigma, numCentrs);

%% 3. Approximate Error of FastESC to NCut
% make sure the sign on each dimenison is aligned
for i=1:numCentrs
    if vESC(:,i)'*vNCut(:,i)<0
        vESC(:,i) = -vESC(:,i);
    end
end
err = norm(vESC-vNCut,'fro');

disp(['Eigenvector approximation error: ' num2str(err/numCentrs*100) '%'])

