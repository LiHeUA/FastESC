function [z, w] = ExplicitFeatureMapping_RFF(data, D, sigma, flagGPU, w0)
% Explicit feature mapping by Random Fourier Features [1]. We follow [2] in
% building z.
%
% [1] A. Rahimi and B. Recht, Random features for large-scale kernel
%     machines.
% [2] D. J. Sutherland and J. Schneider, On the error of random fourier
%     features.
%
% Input:
%       data        n*d         input data. n data points in d dimension.
%       D           1*1         desired output data dimension.
%       sigma       1*1         Gaussian scale parameter, 
%                               kernel(x,y)=exp(-|x-y|^2/sigma^2).
% Output:
%       z           n*D         output data in D dimension.
%       w           d*(D/2)     'w' in Algorithm 1 of [1].
% Notice:
% Applied on Gaussian Kernel only
%
% Li He, heli@gdut.edu.cn

[n,d] = size(data);
%% Algorithm 1 in [1]

% make sure D is an even number
if mod(D,2)
    D = D+1;
end
halfD = floor(D/2);

if flagGPU
    cst = gpuArray(sqrt(2/D));
    if nargin<5
        % randomly sample on frequence
        w = gpuArray.randn(d,halfD) / sigma; % d: input dimension, D: desired dimension
    else
        w = gpuArray(w0);
    end
    
    z = gpuArray.zeros(n,D);
else
    cst = sqrt(2/D);
    
    if nargin<5
        % randomly sample on frequence
        w = randn(d,halfD) / sigma; % d: input dimension, D: desired dimension
    else
        w = w0;
    end
    
    z = zeros(n,D);
end


wTdata = data * w;
clear data;

z(:,1:2:D-1) = cos( wTdata );
z(:,2:2:D) = sin( wTdata );
z = z * cst;
