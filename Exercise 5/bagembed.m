function features = bagembed(bag, data,varargin)
% ”MILES: Multiple-instance learning via embedded
% instance selection.” by Chen, Yixin, Jinbo Bi, and James Ze Wang,
% IEEE Transactions on Pattern Analysis and Machine Intelligence, (2006):
% 1931-1947.
% input:
%   bag: array (Bag instances × features)
%   data: (Total instances × features)
%   sigma > 0 (OPTIONAL, default=1).
% output:
%   feature vector

% Set optional parameter sigma
if length(varargin)==1 && varargin{1}>0
    sigma = varargin{1};
else
    sigma = 1;
end

% Define functions
K = @(x,y) exp(-norm(x-y)^2/(sigma^2));
s = @(Bi,y) max(arrayfun(@(j) K(Bi(j,:),y), 1:size(Bi,1)));

% Calculate feature vector
features = arrayfun(@(j) s(bag,data(j,:)),1:size(data,1));
end

