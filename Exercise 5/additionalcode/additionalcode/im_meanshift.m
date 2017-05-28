%
%    LAB = IM_MEANSHIFT(IM,WIDTH)
%
% Perform a Mean-Shift on image IM, with width-parameter WIDTH.
% The output is a label image LAB, where each pixel has the segment
% label.
% This is basically a wrapper around the function MeanShiftCluster.m

% Copyright: D.M.J. Tax, D.M.J.Tax@prtools.org
% Faculty EWI, Delft University of Technology
% P.O. Box 5031, 2600 GA Delft, The Netherlands
  
function lab = im_meanshift(im,width)

sz = size(im);
if length(sz)<3
    sz = [sz 1];
end
dat = double(reshape(im,sz(1)*sz(2),sz(3)));
[proto,I] = MeanShiftCluster(dat',width,0);
lab = reshape(I,sz(1),sz(2));
end
