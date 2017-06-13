function [train, test] = splitData( trainInds, testInds, M, labels, varargin)
% get the score on test bags using a MILES classifier trained on the
% training bags. A proper subset of the feature matrix M is used in
% training/testing.
%
% INPUT:
% TrainInds: Indices of bags to be used as training
% TestInds: Indices of bags to be used as testing
% M: Feature matrix on complete dataset (both training and testing).
% labels: labels of the bags
% milBag(OPTIONAL): array containing the corresponding bag for each feature.
%
% OUTPUT:
% train and test prdataset.

%% definitions
ix = 1:size(M,1);
milBag = 1:size(M,2);
if length(varargin)==1
    milBag = varargin{1};
end

%% separate data
Mtrain = M(ismember(ix, trainInds),ismember(milBag, trainInds));
trainLabels = labels(ismember(ix, trainInds));
train = prdataset(Mtrain, trainLabels);

Mtest = M(ismember(ix, testInds),ismember(milBag, trainInds));
testLabels = labels(ismember(ix, testInds));
test = prdataset(Mtest, testLabels);
end

