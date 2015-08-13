function [MEAN,VAR,logZ] = f_gaussbernoulli_global(A,B,rho)
    %Gauss part
    VAR=inv(eye(size(A))+A);
    MEAN=B*VAR;
    
    %True Mean
    MEAN=MEAN*rho;
    VAR=VAR*rho;
    
    logZ=0;
    for i=1:size(B,1)
        logZ=logZ+0.5*B(i,:)*VAR*B(i,:)';
    end    
    logZ=-0.5*log(det(eye(size(A))+A))*size(B,1)+logZ ;   
    
    %True log Z
    %log(Z)= log((1-rho)*1 + rho*Z)
    
end
