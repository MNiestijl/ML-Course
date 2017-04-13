function [w] = kmm(X,Z)
% Estimate weights using Kernel Mean Matching
% Paper: Huang, Smola, Gretton, Borgwardt, Schoelkopf (2006)
% Correcting Sample Selection Bias by Unlabeled Data
%
% Input:
%       X       = source data (n x 1)
%       Z       = target data (m x 1)
% Output:
%       w       = weights (n x 1)
%
% Author: Wouter Kouw
% Last update: 28-03-2017

%% Initialization
% Sizes
n = size(X,1);
m = size(Z,1);

% Optimization options
options = optimoptions('quadprog', 'Display', 'final', ...
    'Algorithm', 'interior-point-convex',...
    'TolX', 1e-5, ...
    'maxIter', 1e2);

% |mean(w)-1|<eps
eps = 0.001; 

% regularization
lambda = 1;

% RBF kernel
K = @(x1,x2) exp(-1/2*norm(x1-x2));

% Create Km matrix and km vector
Kmat = zeros(n,n);
k = zeros(m,1);
for i=1:n
    for j=1:m
        Kmat(i,j) = 2/(n^2) * K(X(i,:),X(j,:));
    end
    k(i) = -2/(m*n) * sum(arrayfun(@(j) K(X(i,:),Z(j,:)),1:m));
end
Kmat = Kmat + 2/(n^2).*lambda.*eye(n);

%% Solve quadratic program minimize w.r.t. w:
%       1/n^2 w'Kw - 2/(m*n) w'k 
%       s. t. |mean(w)-1| = epsilon and w >= 0

% define constraints
A = 1./n .* [ones(1,n); -ones(1,n);zeros(n-2,n)];
b = [1+eps; eps-1; zeros(n-2,1)]; 
lb = zeros(n,1);
ub = n*ones(n,1);
x0 = ones(n,1);

% solve program
w = quadprog(Kmat,k,A,b,[],[],lb,ub,x0,options);
end
