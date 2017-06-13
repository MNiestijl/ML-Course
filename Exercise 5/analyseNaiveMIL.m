function error = analyseNaiveMIL(MILdataset,labels)
% Perform a naive MIL classification on the MILdataset using a Fischer
% classifier and analyse the result.

    %% Definitions
    data = getdata(MILdataset);
    nLab = getnlab(MILdataset);
    milbag = struct(MILdataset).ident.milbag;
    bagIds = unique(milbag);
    N = length(bagIds);
    Kfold = 120;

    %% Cross-validation
    indices = crossvalind('KFold',N,Kfold);
    error = zeros(1,Kfold);
    for i=1:Kfold
           testBags = bagIds(indices==i);
           
           % Training 
           trainMask = arrayfun( @(k) ~ismember(milbag(k), testBags),(1:size(MILdataset,1))');
           trainLab = nLab(trainMask);
           train = prdataset(data(trainMask,:),trainLab);
           W = fisherc(train);

           % Testing
           testLabels = labels(testBags);
           predictions = zeros(length(testBags),1);
           for b=1:length(testBags)
               testInstances = data(arrayfun(@(k) milbag(k), 1:size(data,1))==testBags(b),:);
               predLabels = testInstances*W*labeld;
               predictions(b) = combineinstlabels(predLabels, 'majority');
           end
           error(i) = mean(predictions ~= testLabels);
    end
end

