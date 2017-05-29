function analyseNaiveMIL(MILdataset,labels)
% Perform a naive MIL classification on the MILdataset using a Fischer
% classifier and analyse the result.
data = getdata(MILdataset);

%% training the classifier
W = fisherc(MILdataset);

%% Predict the bags
nBags = max(struct(MILdataset).ident.milbag);
predictions = zeros(nBags,1);
for i=1:nBags
   instances = findident(MILdataset,i, 'milbag');
   pred_labels = data(instances,:)*W*labeld;
   predictions(i) = combineinstlabels(pred_labels,'majority');
end

%% Analyse result
appleError = sum(predictions(labels==1)~=1);
bananaError = sum(predictions(labels==2)~=2);

disp('Apples misclassified as banana')
disp(appleError)
disp('Bananas misclassified as apple')
disp(bananaError)

error = mean(predictions ~= labels);
disp('Classification error:')
disp(error)
end

