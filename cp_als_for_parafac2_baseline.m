function [P,Uinit,output] = cp_als_for_parafac2_baseline(X,R,varargin)
% Tensor Toolbox implementation of CP-ALS used in the baseline approach,
% adapted to handle non-negative constraints.


%% Extract number of dimensions and norm of X.
N = ndims(X);
%normX = norm(X);

%% Set algorithm parameters from input or by using defaults
params = inputParser;
params.addParamValue('tol',1e-4,@isscalar);
params.addParamValue('maxiters',50,@(x) isscalar(x) & x > 0);
params.addParamValue('dimorder',1:N,@(x) isequal(sort(x),1:N));
params.addParamValue('init', 'random', @(x) (iscell(x) || ismember(x,{'random','nvecs'})));
params.addParamValue('printitn',1,@isscalar);
params.addParamValue('Constraints',[0 0],@(x) length(x)==2);
params.addParamValue('PARFOR_FLAG', 1, @isscalar);
params.parse(varargin{:});

%% Copy from params object
fitchangetol = params.Results.tol;
maxiters = params.Results.maxiters;
dimorder = params.Results.dimorder;
init = params.Results.init;
printitn = params.Results.printitn;
Constraints = params.Results.Constraints;
PARFOR_FLAG = params.Results.PARFOR_FLAG;

assert(maxiters == 1); % parafac2 specific (no check for convergence)


%% Set up and error checking on initial guess for U.
if iscell(init)
    Uinit = init;
    if numel(Uinit) ~= N
        error('OPTS.init does not have %d cells',N);
    end
    for n = dimorder(2:end);
        if ~isequal(size(Uinit{n}),[size(X,n) R])
            error('OPTS.init{%d} is the wrong size',n);
        end
    end
else
    % Observe that we don't need to calculate an initial guess for the
    % first index in dimorder because that will be solved for in the first
    % inner iteration.
    if strcmp(init,'random')
        Uinit = cell(N,1);
        for n = dimorder(2:end)
            Uinit{n} = rand(size(X,n),R);
        end
    elseif strcmp(init,'nvecs') || strcmp(init,'eigs')
        Uinit = cell(N,1);
        for n = dimorder(2:end)
            Uinit{n} = nvecs(X,n,R);
        end
    else
        error('The selected initialization method is not supported');
    end
end

%% Set up for iterations - initializing U and the fit.
U = Uinit;
% fit = 0;

if printitn>0
    fprintf('\nCP_ALS:\n');
end


if (isa(X,'sptensor') || isa(X,'tensor')) && (exist('cpals_core','file') == 3)
    
    %fprintf('Using C++ code\n');
    [lambda,U] = cpals_core(X, Uinit, fitchangetol, maxiters, dimorder);
    P = ktensor(lambda,U);
    
else
    
    UtU = zeros(R,R,N);
    for n = 1:N
        if ~isempty(U{n})
            UtU(:,:,n) = U{n}'*U{n};
        end
    end
    %% Main Loop: Iterate until convergence
    for iter = 1:maxiters
        
        % fitold = fit;
        
        % Iterate over all N modes of the tensor
        for n = dimorder(1:end)
            
            % Calculate Unew = X_(n) * khatrirao(all U except n, 'r').
            Unew = mttkrp(X,U,n);
            
            % Compute the matrix of coefficients for linear system
            Y = prod(UtU(:,:,[1:n-1 n+1:N]),3);
            
            if ((n==2 && Constraints(1)==1) || (n==3 && Constraints(2)==1))
                Yt = Y';
                if (PARFOR_FLAG)
                    parfor row=1: size(X, n)
                        Unew(row, :) = fastnnls(Yt, Unew(row, :)');%lsqnonneg(Yt, Unew(row, :)'); %Yt \ Unew(row, :)';
                    end
                else
                    for row=1: size(X, n)
                        Unew(row, :) = fastnnls(Yt, Unew(row, :)');%lsqnonneg(Yt, Unew(row, :)'); %Yt \ Unew(row, :)';
                    end
                end
            else
                Unew = Unew / Y;
            end
            
            if issparse(Unew)
                Unew = full(Unew);   % for the case R=1
            end
            
            % Normalize each vector to prevent singularities in coefmatrix
            if iter == 1
                lambda = sqrt(sum(Unew.^2,1))'; %2-norm
            else
                lambda = max( max(abs(Unew),[],1), 1 )'; %max-norm
            end
            
            Unew = bsxfun(@rdivide, Unew, lambda');
            
            U{n} = Unew;
            UtU(:,:,n) = U{n}'*U{n};
            %end
        end
        
        P = ktensor(lambda,U);
        %         if normX == 0
        %             fit = norm(P)^2 - 2 * innerprod(X,P);
        %         else
        %             normresidual = sqrt( normX^2 + norm(P)^2 - 2 * innerprod(X,P) );
        %             fit = 1 - (normresidual / normX); %fraction explained by model
        %         end
        %         fitchange = abs(fitold - fit);
        %
        %         % Check for convergence
        %         if (iter > 1) && (fitchange < fitchangetol)
        %             flag = 0;
        %         else
        %             flag = 1;
        %         end
        %
        %         if (mod(iter,printitn)==0) || ((printitn>0) && (flag==0))
        %             fprintf(' Iter %2d: f = %e f-delta = %7.1e\n', iter, fit, fitchange);
        %         end
        %
        %         % Check for convergence
        %         if (flag == 0)
        %             break;
        %         end
    end
end


%% Clean up final result
% Arrange the final tensor so that the columns are normalized.
P = arrange(P);
% Fix the signs
P = fixsigns(P);

% if printitn>0
%     if normX == 0
%         fit = norm(P)^2 - 2 * innerprod(X,P);
%     else
%         normresidual = sqrt( normX^2 + norm(P)^2 - 2 * innerprod(X,P) );
%         fit = 1 - (normresidual / normX); %fraction explained by model
%     end
%   fprintf(' Final f = %e \n', fit);
% end

output = struct;
output.params = params.Results;
output.iters = iter;
