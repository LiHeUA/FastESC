function [C, R, idxSelected] = EBMM(A,B,N,T,c)
% Extended Basic Matrix Multiplication algorithm. Given two large matrices
% A and B, we want to build two smaller matrices C and R so that 
% AB \approx CR. 
%
% Details of this algorithm can be found in Alg. 2 in [1].
% 
% [1] Li He, Nilanjan Ray and Hong Zhang, Fast Large-Scale Spectral 
% Clustering via Explicit Feature Mapping, submitted to IEEE Trans.
% Cybernetics.
%
% Input:
%       A           p*NT            matrix A
%       B           NT*q            matrix B
%       N        	scalar          choose c from N
%       T           scalar          # of submatrices in A and B
%       c           scalar          choose c from N
%
% Output:
%       C           p*cT            matrix C
%       R           cT*q            matrix R
%       idxSelected c*1             indices of sampled columns (rows) in
%                                   the first submatrix 
%
% Notation:     
% A^(t):    the t-th column in matrix A
% B_(t):    the t-th row in matrix B
% 
% Notice:
% A should be structured as A = [A[1], A[2], ..., A[T]], where A[i] is a
% p*N matrix. And 
%     [B[1]]
% B = [B[2]]
%       ...
%     [B[T]]
% where B[i] is an N*q matrix.
%
% Main idea:
%
% 1. Split A into T submatrices, titled A[1], A[2],..., A[T], 
% A = [A[1], A[2], ..., A[T]]
% and
%     [B[1]]
% B = [B[2]]
%       ...
%     [B[T]]
%
% 2. Randomly with replacement pick the t-th index i_t \in {1,...,N} with
% probability Prob[i_t=k] = p_k, k=1,...,N.
%
% 3. For t=1,...,c, if i_t==k, then select the k-th columns in A[1],
% A[2],...,A[T], scaled by 1/sqrt(c*p_k) and form a new matrix C[t],
% C[t]=[A[1]^(k), A[2]^(k),...,A[T]^(k)]/sqrt(c*p_k). And
%        [B[1]_(k)]
% R[t] = [B[2]_(k)]  /sqrt(c*p_k)
%          ...
%        [B[T]_(k)]
%
% 4. Build C=[C[1],C[2],...,C[T]], and 
%     [R[1]]
% R = [R[2]]
%       ...
%     [R[T]]
%
% 5. Then, proven in [1] that E[CR]=AB. 
% 
% 6. For i=1,...,N, define 
%
% H[i] = A[1]^(i)*B[1]_(i) + A[2]^(i)*B_(i) +...+ A[T]^(i)*B_(i)
% 
% If
%
% p_i = ||H[i]||_F/sum(||H[i']||_F)
%
% Then, proven in [1] that E[||AB-CR||_F^2] is minimal. 
%
% Li He, heli@gdut.edu.cn

if nargin~=5
    %% 0. Demo
    N = 30; % # of columns in one submatrix
    T = 20; % # of submatrix
    c = 20; % # of sampled columns in one sampled submatrix
    A = rand(100,N*T);
    B = rand(N*T,200);
end

%% 1. Get the Optimal Sampling Probabilities
prob_opt = EBMM_OptProb(A, B, N, T);

%% 2. Sample with Replacement c Columns from N with prob_opt 
replacement = true;
idxSelected = randsample(N,c,replacement,prob_opt);

%% 3. Build C and R according to the Extended Basic Matrix Multiplication Algorithm
p = size(A,1);
q = size(B,2);
C = zeros(p,c*T);
R = zeros(c*T,q);

for i=1:c
    C(:,i:c:end) = A(:,idxSelected(i):N:end)/sqrt(c*prob_opt(idxSelected(i)));
    R(i:c:end,:) = B(idxSelected(i):N:end,:)/sqrt(c*prob_opt(idxSelected(i)));
end

if nargin<5
    %% 4. Display
    err = norm(A*B-C*R,'fro')/norm(A*B,'fro');
    disp(['|AB-CR|_F/|AB|_F = ' num2str(err)]);
end
