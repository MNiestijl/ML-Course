%LIKNONC Liknon or 1-norm SVM classifier
%
%     W = LIKNONC(X, C)
%
% Train the Liknon, or actually the 1-norm support vector classifier, with
% tradeoff parameter C on dataset X.
%
function [w,C] = liknonc(x, C)
prtrace(mfilename);

if (nargin < 2)
	prwarning(3,'Lambda set to one');
	C = 1; 
end
if (nargin < 1) | (isempty(x))
	w = prmapping(mfilename,{C});
	w = setname(w,'LIKNON classifier');
	return
end

if ~ismapping(C)   % train the mapping

	if isnan(C) % we want to optimize the C parameter...
		defaultval = {1};
		%vals = logspace(-7,2,10);
		vals = [1e-7, 1e2];
		%[w,C] = regoptc(x, mfilename, {C}, defaultval,[1],vals,testc([],'soft'));
        [w,C] = regoptc(x, mfilename, {C}, defaultval,[1],vals,testd);
		w = setname(w,'LIKNON classifier');
		return;
	end

	% Unpack the dataset.
	islabtype(x,'crisp');
	isvaldset(x,1,2); % at least 1 object per class, 2 classes
	[n,k,c] = getsize(x); 

	% Is this necessary??
	%wsc = scalem(x,'variance');
	%x = x*wsc;

	X = +x;

	if c ~= 2  % multi-class classifier:

		w = mclassc(x, prmapping(mfilename,{C}),'single');
		w = setname(w,'LIKNON classifier');
		return
	end

	% first create the target values:
	y = 2*getnlab(x)-3;

	%---create f
	f = [ones(2*k + n,1); zeros(2,1)];

	%---create A
	A = -[repmat(y,1,k).*X, repmat(-y,1,k).*X, ...
		sparse(diag(repmat(1/C,1,n))), -y, y];

	%---generate b
	b = -ones(n,1);

	%---lower bound constraints
	lb = zeros(size(f));

	%---solve linear program
	if (exist('glpkmex')==3)
		% Use glpkmex
		[z,dummy,status,xtra] = ...
			glpkmex(1,f,A,b,repmat('U',n,1),lb,[],repmat('C',size(f,1),1));
		if ~isfield(xtra,'lambda')
			alpha = [];
		else
			alpha = xtra.lambda;
		end
	elseif (exist('glpkcc')==3)
		% Use glpkcc
		[z,dummy,status,xtra] = ...
			glpk(f,A,b,lb,[],repmat('U',n,1),repmat('C',size(f,1),1),1);
		alpha = xtra.lambda;
	else
		% Use linprog
		opts = optimset('display','off');
		[z,fmin,exitflag,outp,alpha] = linprog(f,A,b,[],[],lb,[],[],opts);
		alpha = alpha.ineqlin;
	end

	%---extract parameters
	u = z(1:k); u = u(:);
	v = z(k+1:2*k); v = v(:);
	zeta = z(2*k+1:2*k+n); zeta = zeta(:);
	bp = z(end-1);
	bm = z(end);

	% now find out how sparse the result is:
	%nr = sum(beta>1e-6);
	nr = sum(abs(u-v)>0);
	
	% and store the results:
	%W.wsc = wsc;
	W.u = u; %the ultimate weights
	W.v = v;
	W.bp = bp;
	W.bm = bm;
	W.alpha = alpha;
	W.zeta = zeta;
	W.nr = nr;
   W.C = C;
	w = prmapping(mfilename,'trained',W,getlablist(x),size(x,2),c);
	w = setname(w,'LIKNON classifier');
	
else
	% Evaluate the classifier on new data:
	W = getdata(C);
	n = size(x,1);

	% linear classifier:
	out = x*(W.u-W.v) - (W.bp-W.bm);

	% and put it nicely in a prtools dataset:
	w = setdat(x,sigm([-out out]),C);

end
		
return
