function d = pdist2_my(a,b)
% Compute square distance matrix between ma-by-n matrix a and mb-by-n
% matrix b. Matrix d is ma-by-mb, d(i,j)=||a(i,:)-b(j,:)||_2^2 

sa = sum(a.^2,2);
sb = sum(b.^2,2);
d = bsxfun(@minus,sa,2*a*b');
d = bsxfun(@plus,sb',d);
