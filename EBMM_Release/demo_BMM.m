function demo_BMM(A, B, N, c)
% Demo of the Basic Matrix Multiplication algorithm [1]. Select c columns 
% (or rows) from A (or B) to form C (or R) so that AB\approx CR
%
% [1] P. Drineas, R. Kannan, and M. W. Mahoney, “Fast monte carlo 
% algorithms for matrices i: Approximating matrix multiplication,” SIAM 
% Journal on Computing, vol. 36, no. 1, pp. 132–157, 2006.
%
% Input:
%       A           p*N             matrix A
%       B           N*q          	matrix B
%       N        	scalar      	choose c from N
%       c           scalar       	choose c from N
%
% Notation:     
% A^(t):    the t-th column in matrix A
% B_(t):    the t-th row in matrix B
% 
% Main idea:
% 1. Randomly with replacement pick the t-th index i_t \in {1,...,N} with
% probability prob_col[i_t=k] = p_k, k=1,...,N.
%
% 2. For t=1,...,c, if i_t==k, then select the k-th columns in A, scale by 
% 1/sqrt(c*p_k) and form a new matrix C,
% C^(t)=A^(k)/sqrt(c*p_k). And R_(t) = B(k)/sqrt(c*p_k).
% 
% 3. Then, for arbitraty probabilities p_i, E[CR]=AB. 
% 
% 4. For i=1,...,N, define 
% If
%
% p_i = |A^(i)|*|B_(i)|/sum(|A^(i')|*|B_(i')|)
%
% Then, E[||AB-CR||_F^2] is minimal. 
%
% Li He, heli@gdut.edu.cn

%% 0. Initialization
clc

N = 6;
c = 3;
% randomly generate A and B
A = rand(100,N);
B = rand(N,200);

p = size(A,1);
N = size(A,2);
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

C = zeros(p,c);
R = zeros(c,q);
for i=1:N^c
    % chosen columns (rows) among N^c possible choices
    index = indices(i,:);
    
    % build C and R
    for t=1:c
        ind = index(t); % index of one chosen column
        C(:,t) = A(:,ind)/sqrt(c*prob_col(ind));
        R(t,:) = B(ind,:)/sqrt(c*prob_col(ind));
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

prob_opt = sqrt(sum(A.^2,1)) .* sqrt(sum(B.^2,2))';
prob_opt = prob_opt/sum(prob_opt);
% probability of one C (or R) to appear
prob_matrix_opt = prod( prob_opt(indices), 2 );
% ECR = sum( prob_matrix*C*R )
ECR_opt = zeros(p,q);
% E[||AB-CR||_F^2]
ECRF_opt = 0;


C = zeros(p,c);
R = zeros(c,q);
for i=1:N^c
    % chosen columns (rows) among N^c possible choices
    index = indices(i,:);
    
    % build C and R
    for t=1:c
        ind = index(t); % index of one chosen column
        C(:,t) = A(:,ind)/sqrt(c*prob_opt(ind));
        R(t,:) = B(ind,:)/sqrt(c*prob_opt(ind));
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
