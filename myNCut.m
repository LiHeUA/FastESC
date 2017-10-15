function vec = myNCut(data, sigma, k)
dis = pdist2(data,data);
K = exp(-dis.^2/2/sigma^2);

% degree matrix D
D = sum(K);
% D^{-1/2}
invD2 = 1./sqrt(D);

% D^{-1/2}K
nK = bsxfun(@times, K, invD2');
% D^{-1/2}KD^{-1/2}
nK = bsxfun(@times, nK, invD2);

if size(nK,1)<3000
    [vec, val] = eig(nK);
    [~, idx] = sort(diag(val),'descend');
    vec = vec(:,idx(1:k));
else
    [vec, ~] = eigs(nK, k, 'LM');
end

% y = D^{-1/2}*z
vec = bsxfun(@times, vec, invD2');