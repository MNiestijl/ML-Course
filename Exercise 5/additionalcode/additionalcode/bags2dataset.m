%BAGS2DATASET    convert bags to a Prtools dataset
%
%    A = BAGS2DATASET(BAGS,BAGLAB)
%
% INPUT
%   BAGS    cell-array of bags, containing a variable number of
%           instances in a bag
%   BAGLAB  class label for each bag
%
% OUTPUT
%   A       Prtools dataset
%
% DESCRIPTION
% Convert a collection of bags, stored in the cell-array BAGS, to a
% Prtools dataset. Each instance in a bag becomes one object in dataset
% A. The label for the objects are copied from the bag label.
% The bag indices are stored in the identifier 'milbag', so you can
% retrieve them using:
% >> bagid = getident(A,'milbag')
%
function a = bags2dataset(bags,baglab)

if length(bags)~=size(baglab,1)
   error('Number of cells in BAGS has to be equal to the number of labels.');
end
B = length(bags);
dat = [];
lab = [];
id = [];
for i=1:B
   n = size(bags{i},1);
   dat = [dat;bags{i}];
   lab = [lab; repmat(baglab(i,:),n,1)];
   id = [id; repmat(i,n,1)];
end

a = prdataset(dat,lab);
a = setident(a,id,'milbag');

