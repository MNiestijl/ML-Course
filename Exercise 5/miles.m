function miles( MILdataset, labels)
% Perform the MILES algorithm on the dataset using a L1 support vector
% classifier (LIKNON).

%%Settings
sigma = 20;
C = 10;

%% Create feature matrix
data = getdata(MILdataset);
nBags = max(struct(MILdataset).ident.milbag);
nInst = size(data,1);
M = zeros(nBags, nInst);
for i=1:nBags
    instances = findident(MILdataset,i, 'milbag');
    bag = data(instances,:);
    M(i,:) = bagembed(bag,data,sigma);
end

%% Make PRDataset
A = prdataset(M,labels);

%% Train classifier
W = liknonc(A,C);

%% Analyse result
error = testc(A*W);
disp('error:')
disp(error)


end

