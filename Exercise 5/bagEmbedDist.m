function distances = bagEmbedDist(bag,bags,varargin)
% Compute distance of bag to bags using some bag distance.
% Default distance: Hausdorff
    dist = @(A,B) hausdorffDist(A,B);
    if ~isempty(varargin)
        dist=varargin{1};
    end
    distances = arrayfun(@(i) dist(bag,bags{i}), 1:size(bags,1));
end

