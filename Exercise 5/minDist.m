function d = minDist(A, B)
% Minimal distance between two arrays
d = min(arrayfun(@(ai) min(arrayfun(@(bj) norm(A(ai,:)-B(bj,:)),1:size(B,1))),1:size(A,1)));
end
