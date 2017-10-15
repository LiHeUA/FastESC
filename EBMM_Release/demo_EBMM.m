function demo_EBMM
% Demot of Extended Basic Matrix Multiplication algorithm. 
% Select cT columns (or rows) from A (or B) to form C (or R) so that
% AB\approx CR. 
% Also verify Theorem 1 in [1].
%
% Details of this algorithm can be found in Alg. 2 in [1].
% 
% [1] Li He, Nilanjan Ray and Hong Zhang, Fast Large-Scale Spectral 
% Clustering via Explicit Feature Mapping, submitted to IEEE Trans.
% Cybernetics.
%
% Parameter:
%       A           p*NT            matrix A
%       B           NT*q            matrix B
%       N        	scalar          choose c from N
%       T           scalar          # of submatrices in A and B
%       c           scalar          choose c from N
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
% A[2],...,A[T], scale by 1/sqrt(c*p_k) and form a new matrix C[t],
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
% 5. Then, E[CR]=AB. 
% 
% 6. For i=1,...,N, define 
%
% H[i] = A[1]^(i)*B[1]_(i) + A[2]^(i)*B_(i) +...+ A[T]^(i)*B_(i)
% 
% If
%
% p_i = ||H[i]||_F/sum(||H[i']||_F)
%
% Then, E[||AB-CR||_F^2] is minimal. 
%
% Li He, heli@gdut.edu.cn

%% 0. Initialization
clc

N = 6;
T = 2;
c = 3;
% randomly generate A and B
A = rand(100,N*T);
B = rand(N*T,200);

p = size(A,1);
q = size(B,2);

% randomly generate the sampling probabilities; for arbitraty prob_col.,
% E(CR)=AB should hold true
prob_col = rand(1,N);
prob_col = prob_col/sum(prob_col);

%% 1. Verification of E(CR)=AB with Arbitrary Probabilities
% we have in total N^c possible C (and R); exhaustively calculate all

disp('Exp 1: E(CR)=AB with arbitrary sampling probabilities')

% generate all N^c possible indices
indices = nchoosek_with_replacement(N,c);

% probability of one C (or R) to appear
prob_matrix = prod( prob_col(indices), 2 );

% build C and R, and brute-forcely check E(CR)

% ECR = sum( prob_matrix*C*R )
ECR = zeros(p,q);
% E[||AB-CR||_F^2]
ECRF = 0;
% ground truth AB
AB = A*B;

C = zeros(p,c*T);
R = zeros(c*T,q);
for i=1:N^c
    % chosen columns (rows) among N^c possible choices
    index = indices(i,:);
    
    % build C and R
    for t=1:c
        ind = index(t); % index of one chosen column
        C(:,t:c:end) = A(:,ind:N:end)/sqrt(c*prob_col(ind));
        R(t:c:end,:) = B(ind:N:end,:)/sqrt(c*prob_col(ind));
    end
    
    % E(CR)
    ECR = ECR + prob_matrix(i)*C*R;
    % E(|AB-CR|_F^2)
    ECRF = ECRF + prob_matrix(i)*norm(C*R-AB,'fro')^2;
end

disp(['||E(CR) - AB||_F = ' num2str(norm(ECR-AB,'fro'))])

%% 2. Optimal Sapmling
% if using the optimal sampling, then E[||AB-CR||_F^2] should be minimum
disp(' ');
disp('Exp 2: the optimal sampling will minimize E(|AB-CR|_F^2)')

% get the optimal sampling probabilities
prob_opt = EBMM_OptProb(A, B, N, T);

% probability of one C (or R) to appear
prob_matrix_opt = prod( prob_opt(indices), 2 );
% ECR = sum( prob_matrix*C*R )
ECR_opt = zeros(p,q);
% E[||AB-CR||_F^2]
ECRF_opt = 0;


C = zeros(p,c*T);
R = zeros(c*T,q);
for i=1:N^c
    % chosen columns (rows) among N^c possible choices
    index = indices(i,:);
    
    % build C and R
    for t=1:c
        ind = index(t); % index of one chosen column
        C(:,t:c:end) = A(:,ind:N:end)/sqrt(c*prob_opt(ind));
        R(t:c:end,:) = B(ind:N:end,:)/sqrt(c*prob_opt(ind));
    end
    
    ECR_opt = ECR_opt + prob_matrix_opt(i)*C*R;
    ECRF_opt = ECRF_opt + prob_matrix_opt(i)*norm(C*R-AB,'fro')^2;
end

disp(['||E(CR_opt) - AB||_F = ' num2str(norm(ECR_opt-AB,'fro'))])

% compare the F-norm error of CR_optimal with the CR in Experiment 1
disp(['E[||CR_opt - AB||_F^2 = ' num2str(ECRF_opt) ', E[||CR_Exp1 - AB||_F^2 = ' num2str(ECRF)])



function indices = nchoosek_with_replacement(n,k)
indices = cell(1,k);
[indices{:}] = ndgrid(1:n);
indices = indices(end:-1:1);
indices = cat(k+1, indices{:});
indices = reshape(indices, [n^k, k]);