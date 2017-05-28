%INDNN Find indices of nearest neighbors
%
%   IND = INDNN(TEST,TRAIN)
%   IND = TEST*INDNN([],TRAIN) for batch processing
%   IND = INDNN(TEST,CLASSF)
%   IND = TEST*INDNN([],CLASSF) for batch processing
% 
% INPUT
%   TEST   Double array or PRDATASET with object feature vectors
%   TRAIN  Double array or PRDATASET with object feature vectors
%   CLASSF Trained PRTools classifier, either KNNC or NMC.
%
% OUTPUT
%   IND   Column vector with indices pointing for every object in TEST to
%         the nearest neighbor of a trainset.
%
% DESCRIPTION
% This routine is optimised for speed and size to perform a nearest
% neighbor search by the standard PRTools distance routine DISTM. The
% training set may be given explicitly by TRAIN or implicitly by a trained
% KNNC or NMC classifier. The latter stores class means which are useful in
% a kmeans clustering, see PRKMEANS
%
% SEE ALSO
% DATASETS, MAPPINGS, KNNC, NMC, PRKMEANS

% Copyright: R.P.W. Duin, r.p.w.duin@37steps.com

function ind = indnn(testset,trainset)

  if ismapping(trainset) % possible knnc
    if strcmp(getmapping_file(trainset),'knn_map')
      trainset = getdata(trainset,1);
    elseif strcmp(getmapping_file(trainset),'normal_map')
      trainset = getdata(trainset,'mean');
    else
      error('Unsupported data type')
    end
  end
  if isempty(testset)
    ind = prmapping('indnn','fixed',trainset);
  else
    batch = ceil(prmemory/(20*size(trainset,1)));
    testset  = +testset;
    trainset = +trainset;
    n = 0;
    m = size(testset,1);
    ind = zeros(m,1);
    t = sprintf('Finding nearest neighbors for %i objects: ',m);
    prwaitbar(m,t)
    for i=1:floor(m/batch)
      [~,ind(n+1:n+batch)] = min(distm(testset(n+1:n+batch,:),trainset),[],2);
      n = n+batch;
      prwaitbar(m,n,[t num2str(n)]);
    end
    [~,ind(n+1:m)] = min(distm(testset(n+1:m,:),trainset),[],2);
    prwaitbar(0);
        
  end

return