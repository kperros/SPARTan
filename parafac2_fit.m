function fit=parafac2_fit(X,H,A,C,P,K, PARFOR_FLAG)
% computing the current fit for convergence checking purposes
fit = 0;
if (PARFOR_FLAG)
    parfor k = 1:K
        fit = fit + sum(sum((X{k} - (P{k}*H)*diag(C(k,:))*A').^2));
    end
else
    for k = 1:K
        fit = fit + sum(sum((X{k} - (P{k}*H)*diag(C(k,:))*A').^2));
    end
end
