function prob = EBMM_OptProb(A, B, N, T)
% The optimal sampling probabilities in the Extended Basic Matrix 
% Multiplication algorithm. Select cT columns (or rows) from A (or B) to 
% form C (or R) so that AB\approx CR. 
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
%
% Output:
%       prob        N*1             probabilities of one colum to be
%                                   chosen
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


if nargin~=4
    %% 0. Demo
    T = 3; % number of submatrices
    N = 200;
    
    % randomly generate A[1],...,A[T] and B[1],...,B[T]
    A = rand(300,N*T);
    B = rand(N*T,400);
end

%% 1. Calculate the Optimal Sampling Probabilities prob
% use a) F-norm or b) Fast and, if T=2, c) Fast for T=2
% a) is easy to understand but b) is much faster

% % a) F-norm style
% Hf = zeros(N,1); % F-norm of H[1],...,H[N]
% for i=1:N
%     Ai = A(:,i:N:end); % the i-th column in A[1]...A[T]
%     Bi = B(i:N:end,:); % the i-th row in B[1]...B[T]
%     
%     % build H[i]
%     Hi = Ai*Bi;
%     % F-norm of H[i]
%     Hf(i) = norm(Hi,'fro');
% end
% % prob. of columns to be chosen
% prob = Hf/sum(Hf);

% b) Fast
ss = zeros(N,1);
for i=1:N
    P = A(:,i:N:end);
    Q = B(i:N:end,:);
    
    ss(i) = sqrt( trace((P'*P)*(Q*Q')) );
end
prob = ss'/sum(ss);


% % c) Fast when T=2
% if T~=2
%     disp('T must be 2!')
%     return;
% end
% % see https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm
% % |A+B|_F^2 = |A|_F^2 + |B|_F^2 + 2*<A,B>_F
% A1 = A(:,1:N);A2 = A(:,N+1:end);
% B1 = B(1:N,:);B2 = B(N+1:end,:);
% a = sum(A1.^2).*sum(B1.^2,2)';
% b = sum(A2.^2).*sum(B2.^2,2)';
% ab = sum(A1.*A2) .* sum(B1.*B2,2)';
% 
% s = sqrt(a+b+2*ab);
% 
% % prob. of columns to be chosen
% prob = s'/sum(s);


