function [MEAN,VAR,logZ] = f_gauss(A,B)
    VAR=inv(eye(size(A))+A);
    MEAN=B*VAR;            
    logZ=0;
    for i=1:size(B,1)
        logZ=logZ+0.5*B(i,:)*VAR*B(i,:)';
    end    
    logZ=-0.5*log(det(eye(size(A))+A))*size(B,1)+logZ ;          
end
