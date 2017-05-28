%FNNC Fast Nearest Neighbor Classifier
%
%   W = A*FNNC
%   W = FNNC(A)
%
% DESCRIPTION
% This is a fast version of KNNC([],1), especially useful for many-class
% problems. For reasons of speed it will NOT return proper confidences on
% evaluation. D = B*W will be a 0/1 dataset. Classification results are
% otherwise identical to KNNC([],1).
%
% SEE ALSO
% MAPPINGS, DATASETS, KNNC

% Copyright: R.P.W. Duin, r.p.w.duin@37steps.com

function out = fnnc(a,v)

  if nargin < 1 || isempty(a)
    out = prmapping(mfilename);
    out = setname(out,'1-NN');
  elseif nargin == 1 && isdataset(a)
    out = trained_classifier(a,{a});
  elseif ismapping(v)
    trainset = getdata(v,1);
    nlab = getnlab(trainset);
    L = indnn(a,trainset);
    out = zeros(size(a,1),getsize(trainset,3));
    for i=1:size(a,1)
      out(i,nlab(L(i))) = 1;
    end
    out = setdat(a,out,v);
  else
    error('Illegal input')
  end

return
