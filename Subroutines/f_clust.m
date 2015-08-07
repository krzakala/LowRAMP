function [MEAN,VAR,logZ] = f_clust(A,B)
    RANK=size(A,1);n=size(B,1);
    AA=repmat(diag(A),1,n);
    Prob=-0.5*AA+B';
    KeepMax=max(Prob);
    Prob=exp(Prob-repmat(KeepMax,RANK,1));
    Norm=repmat(sum(Prob),RANK,1);        
    Tokeep=sum(log(sum(Prob/RANK)));    
    Prob=Prob./Norm;
    MEAN=Prob';
    VAR=diag(sum(MEAN)/n)-Prob*Prob'/n;       
    logZ=sum(KeepMax)+Tokeep;
end