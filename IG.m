function beta = IG(Y, k, flagGPU)
% Increamentally learn beta in FastESC, as described in [1]. 
% Step 7 to 11 in  Alg. 3 in [1].
%
% [1] Li He, Nilanjan Ray and Hong Zhang, Fast Large-Scale Spectral 
% Clustering via Explicit Feature Mapping, submitted to IEEE Trans.
% Cybernetics.
%
% Input:
%           Y           D*n         EFM of data
%           k           scalar      number of clusters
%           flagGPU     bool        true for using GPU
% Output:
%           beta        D*k         beta in Alg. 3 in [1]
%
% Li He, heli@gdut.edu.cn


n = size(Y,2); % size of data set
nGroup = floor(n/50); % size of each group
idx = randperm(n); % random select data
nUsedData = 0; % used data in each iteration

% build G of the first group
G = Y(:,idx(nUsedData+1:nUsedData+nGroup))*Y(:,idx(nUsedData+1:nUsedData+nGroup))';
nUsedData = nUsedData+nGroup;

% compute eigvecs
if flagGPU
    cpuG = gather(G);
    [beta, ~] = eigs(cpuG,k,'LM');
else
    [beta, ~] = eigs(G,k,'LM');
end

preBeta = beta; % previous beta
err = inf; % beta approx error

thr_beta = k*0.02; % threshold on beta approx error

% repeat until all data points are used or error is less than the threshold
while nUsedData+nGroup<n && err>thr_beta
    
    % in each iteration, G = G + Y_new*Y_new^T
    G = G+Y(:,idx(nUsedData+1:nUsedData+nGroup))*Y(:,idx(nUsedData+1:nUsedData+nGroup))';
    nUsedData = nUsedData+nGroup; % update No. of used data
    
    [beta,~] = eigs(gather(G),k,'LM'); % solve eigvecs
    % sign alignment
    for t=1:k
        if (beta(:,t)'*preBeta(:,t))<0
            beta(:,t) = -beta(:,t);
        end
    end
    
    % beta approx error
    err = norm(preBeta-beta,'fro');
    preBeta = beta;
end

if flagGPU
    beta = gpuArray(beta);
end