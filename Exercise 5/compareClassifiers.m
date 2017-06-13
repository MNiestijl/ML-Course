function scores = compareClassifiers(bags, labels, MILdataset)
% Computes the mean and std scores of various classifiers.
    N=length(labels);
    
    %% Settings
    Kfold = N;                      %  K-fold cross-validation
    sigma = 20;                     % MILES width parameter
    C = 1;                          % MILES regularization parameter
    classifiers1 = {@(X) liknonc(X,C), @(X) fisherc(X),@(X) knnc(X,5)};
    distFunc = @(A,B) hausdorffDist(A,B);
    
    %% Definitions
    nClassifiers = length(classifiers1);
    
    %% Get feature matrices
    disp('Calculating feature matrices');
    M = getFeatureMatrix(MILdataset, sigma);
    D = distMatrix(bags,distFunc);

    %% Perform cross validation
    result = zeros(nClassifiers,2,Kfold);
    indices = crossvalind('KFold',N,Kfold);
    fprintf('\nPerforming %i fold cross-validation\n\n', Kfold);
    for i=1:Kfold
       if mod(i,5)==0
           disp(i)
       end
       trainInds = find(indices~=i);
       testInds = find(indices==i);
       for k=1:nClassifiers
           milbag = struct(MILdataset).ident.milbag;
           [train1, test1] = splitData(trainInds,testInds,M,labels, milbag);
           W1 = classifiers1{k}(train1);
           result(k,1,i) = 1-testd(test1,W1);
           
           [train2, test2] = splitData(trainInds,testInds,D,labels);
           W2 = classifiers1{k}(train2);
           result(k,2,i) = 1-testd(test2,W2);
       end
    end
    
    %% Summarize result
    scores = zeros(nClassifiers,2);
    for i=1:nClassifiers
        scores(i,1) = mean(result(i,1,:));
        scores(i,2) = mean(result(i,2,:));
    end
end


