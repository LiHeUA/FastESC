function v = FastESC_LargeScale(inputData, sigma, D, k, flagGPU)
% Fast Explicit Spectral Clustering, large-scale data set.
% Details of this algorithm can be found in [1].
% 
% [1] Li He, Nilanjan Ray and Hong Zhang, Fast Large-Scale Spectral 
% Clustering via Explicit Feature Mapping, submitted to IEEE Trans.
% Cybernetics.
%
% Input:
%       inputData   n*d     input data. n data, each data with d dimension
%       sigma       scalar  Gaussian parameter, Kij=exp(-|x-y|^2/2/sigma^2)
%       D           scalar  desired dimension of explicit features
%       k           scalar  number of clusters
%       flagGPU     bool    true for using GPU
% Output:
%       v           n*k     approximate NCut eigenvectors
% Parameters:
%       X           n*DStar	data after explicit feature mapping
%       W           n*1     diagonal matrix, degree matrix in NCut
%       Y           n*D     X * W^{-1/2}
%       G           n*n     XW^{-1}X^T, or YY^T
%
% Li He, heli@gdut.edu.cn

%% 0. Initialization
if nargin<5
    flagGPU = false;
end

% path of Extended Basic Matrix Multiplication algorithm
addpath ./EBMM_Release;

% using GPU or not
if flagGPU
    data = gpuArray(inputData);
else
    data = inputData;
end
clear inputData;

% D should be even
if mod(D,2)
    D = D+1;
end

% number of data points
n = size(data,1);

%% 1. Learning w by Alg. 2 in [1]
DStar = D+1000; % DStar: D^star in [1]
[X0, w0] = ExplicitFeatureMapping_RFF(data(1:min(1000,n),:), DStar, sigma, flagGPU);
X0 = X0'; % So X0 after is DStar-by-n

% idxSelected: indices of selected w0
[~, ~, idxSelected] = EBMM(X0',X0,DStar/2,2,D/2);

w = w0(:,idxSelected);

%% 2. Get X by learned w
[X, ~] = ExplicitFeatureMapping_RFF(data, D, sigma, flagGPU, w);
X = X';
clear data;

%% 3. Calculate Y = X*D^{-1/2}
% mean of X
barX = sum(X,2);

% W, the degree matrix in NCut
W = barX'*X;
% abandon small W
W(W<0) = 0;

invW2 = 1./sqrt(W); % invW2 = W^{-1/2}, 1-by-n

% Y = X*W^{-1/2}
Y = bsxfun(@times,X,invW2);

%% 4. Get G and beta
% Get G incrementally 
beta = IG(Y, k, flagGPU);

%% 5. Complete NCut
% distance of data to cutting plane
vec = Y'*beta(:,1:k); % n-by-k
% normalization factor
nm = 1./sqrt(sum(vec.^2,1));
% normalized distance of data to cutting plane, or the appr eigvec
vec = bsxfun(@times,vec,nm); % n by k

% in NCut, y=D^{-1/2}*z
v = bsxfun(@times, vec, invW2');

if flagGPU
    v = gather(v);
end