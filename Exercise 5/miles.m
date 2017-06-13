function [A,W] = miles( MILdataset, labels)
%%Settings
sigma = 20;
C = 10;

%% Train classifier
M = getFeatureMatrix( MILdataset, sigma );
A = prdataset(M,labels);
W = liknonc(A,C);
end

