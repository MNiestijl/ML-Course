function d = minDist(A, B)
% Maximal distance between two arrays
d = max(arrayfun(@(ai) max(arrayfun(@(bj) norm(A(ai,:)-B(bj,:)),1:size(B,1))),1:size(A,1)));
end
