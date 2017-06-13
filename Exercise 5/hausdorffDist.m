function d = hausdorffDist(A, B)
    d = max(helper(A,B), helper(B,A));
end

function val = helper(A, B)
   val = max(arrayfun(@(ai) min(arrayfun(@(bj) norm(A(ai,:)-B(bj,:)),1:size(B,1))),1:size(A,1)));
end
