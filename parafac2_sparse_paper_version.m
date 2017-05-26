function [H,A,C,P,fit,ITER_TIME,AddiOutput]=parafac2_sparse_paper_version(X,F,Constraints,Options,H,A,C,P)
% This code contains the core part of the fitting algorithm for the PARAFAC2 model.
% It is parameterized to run either the SPARTan or the baseline approach.
% We accredit the dense PARAFAC2 implementation by Rasmus Bro (http://www.models.life.ku.dk/algorithms), from where we have adapted many functionalities.

ShowFit  = 5; % Show fit every 'ShowFit' iteration
NumRep   = 10; %Number of repetead initial analyses
NumItInRep = 80; % Number of iterations in each initial fit
if ~(length(size(X))==3||iscell(X))
    error(' X must be a three-way array or a cell array')
end

if nargin < 4
    Options = zeros(1,5);
end
if length(Options)<5
    Options = Options(:);
    Options = [Options;zeros(5-length(Options),1)];
end
BASELINE = ~Options(6);% Choice of algorithm
PARFOR_FLAG = Options(7);

% Convergence criterion
if Options(1)==0
    ConvCrit = 1e-7;
else
    ConvCrit = Options(1);
end
if Options(5)==0
    disp(' ')
    disp(' ')
    disp([' Convergence criterion        : ',num2str(ConvCrit)])
end

% Maximal number of iterations
if Options(2)==0
    MaxIt = 2000;
else
    MaxIt = Options(2);
end

% Initialization method
initi = Options(3);

if nargin<3
    Constraints = [0 0];
end
if length(Constraints)~=2
    Constraints = [0 0];
    disp(' Length of Constraints must be two. It has been set to zeros')
end
ConstraintOptions=[ ...
    'Fixed                     ';...
    'Unconstrained             ';...
    'Non-negativity constrained';...
    'Orthogonality constrained ';...
    'Unimodality constrained   ';...
    'Not defined               ';...
    'Not defined               ';...
    'Not defined               ';...
    'Not defined               ';...
    'Not defined               ';...
    'Not defined               ';...
    'GPA                       '];

if Options(5)==0
    disp([' Maximal number of iterations : ',num2str(MaxIt)])
    disp([' Number of factors            : ',num2str(F)])
    disp([' Loading 2. mode, A           : ',ConstraintOptions(Constraints(1)+2,:)])
    disp([' Loading 3. mode, C           : ',ConstraintOptions(Constraints(2)+2,:)])
    disp(' ')
end


% Make X a cell array if it isn't
if ~iscell(X)
    for k = 1:size(X,3)
        x{k} = X(:,:,k);
    end
    X = x;
    clear x
end
I = size(X{1}, 2);
K = max(size(X));
for k=2: K
    assert(size(X{k}, 2)==I);
end

% Initialize
if nargin<5
    assert(initi~=0);
    if initi==1 % Initialize by SVD
        if Options(5)==0
            disp('SVD based initialization')
        end
        
        XtX=full(X{1}'*X{1});
        if (PARFOR_FLAG)
            parfor k=2: K
                XtX = XtX + X{k}'*X{k};
            end
        else
            for k=2: K
                XtX = XtX + X{k}'*X{k};
            end
        end
        sumdiag_xtx = sum(diag(XtX));
        
        [A,~,~]=svd(XtX, 0);
        if (Constraints(1)==1)
            A = abs(A);
        end
        A=A(:,1:F);
        C=ones(K,F)+randn(K,F)/10;
        if (Constraints(2)==1)
            C = abs(C);
        end
        H = eye(F);
    elseif initi==2
        if Options(5)==0
            disp(' Random initialization')
        end
        A = rand(I,F);
        C = rand(K,F);
        H = eye(F);
    else
        error(' Options(2) wrongly specified')
    end
end

if initi~=1
    diag_xtx = cellfun(@(x) full(diag(x'*x)), X, 'UniformOutput',false);
    sumdiag_xtx = sum(cellfun(@sum, diag_xtx));
end
fit    = sumdiag_xtx;%sum(diag(XtX));
oldfit = fit*2;
fit0   = fit;
it     = 0;

if Options(5)==0
    disp(' ')
    disp(' Fitting model ...')
    disp(' Loss-value      Iteration     %VariationExpl')
end

tic;
% Iterative part
while abs(fit-oldfit)>oldfit*ConvCrit && it<MaxIt && fit>1000*eps
    oldfit = fit;
    it   = it + 1;
    
    % Update P
    P = cell(K, 1);
    YY = cell(K, 1);
    if (PARFOR_FLAG)
        parfor k = 1:K
            Qk = H*diag(C(k,:))*(X{k}*A)';%(A'*X{k}');
            P{k} = Qk'*psqrt(Qk*Qk');
            YY{k} = sparse(P{k}'*X{k});
        end
    else
        for k = 1:K
            Qk = H*diag(C(k,:))*(X{k}*A)';%(A'*X{k}');
            P{k} = Qk'*psqrt(Qk*Qk');
            YY{k} = sparse(P{k}'*X{k}); 
        end
    end
    
    if (BASELINE) %creation of the sptensor is necessary only for the baseline
        spY = permute( sptensor( sptenmat(cell2mat(YY), [1 2], 3, [F, K, I]) ) , [1, 3, 2]); % assert(isequal(double(tensor(spY)), Y));
    end
    
    % Update A,H,C using PARAFAC-ALS
    if (BASELINE)
        spCP = cp_als_for_parafac2_baseline(spY, F, 'init', {[], A, C}, 'maxiters', 1, 'printitn', 0, 'Constraints', Constraints, 'PARFOR_FLAG', PARFOR_FLAG);
        %[Bro_H, Bro_A, Bro_C, Bro_ff]=parafac(reshape(Y,F,I*K),[F I K],F,1e-4,[ConstB Constraints(1) Constraints(2)],H, A, C, 1); %caution, has to adjusted/permuted
        %Bro_CP = ktensor({Bro_H, Bro_A, Bro_C});
    else
        spCP = cp_als_for_parafac2(YY, F, 'init', {[], A, C}, 'maxiters', 1, 'printitn', 0, 'Constraints', Constraints, 'PARFOR_FLAG', PARFOR_FLAG);
    end
    
    spCP = arrange(spCP, 1);
    H = spCP.U{1}; A = spCP.U{2}; C = spCP.U{3};
    fit = parafac2_fit(X,H,A,C,P,K, PARFOR_FLAG);
    
    if rem(it,ShowFit)==0||it == 1 % Print interim result
        if Options(5)==0
            fprintf(' %12.10f       %g        %3.4f \n',fit,it,100*(1-fit/fit0));
        end
    end
end
ITER_TIME = toc;

if rem(it,ShowFit)~=0 %Show final fit if not just shown
    if Options(5)==0
        fprintf(' %12.10f       %g        %3.4f \n',fit,it,100*(1-fit/fit0));
    end
end


function X = psqrt(A,tol)
% Produces A^(-.5) even if rank-problems

[U,S,V] = svd(A,0);
if min(size(S)) == 1
    S = S(1);
else
    S = diag(S);
end
if (nargin == 1)
    tol = max(size(A)) * S(1) * eps;
end
r = sum(S > tol);
if (r == 0)
    X = zeros(size(A'));
else
    S = diag(ones(r,1)./sqrt(S(1:r)));
    X = V(:,1:r)*S*U(:,1:r)';
end



