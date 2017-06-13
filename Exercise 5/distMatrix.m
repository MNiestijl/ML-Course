function D = distMatrix( bags, varargin )
% Compute distance between bags and store in matrix D.
% Default distance: Hausdorff
    dist = @(A,B) hausdorffDist(A,B);
    if ~isempty(varargin)
        dist=varargin{1};
    end
    N = size(bags,1);
    D = zeros(N,N);
    for i=1:N
        D(i,:) = bagEmbedDist(bags{i},bags,dist);
    end
end

