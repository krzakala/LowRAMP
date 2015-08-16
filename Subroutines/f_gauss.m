function [MEAN,VAR,logZ] = f_gauss(A,B)
    VAR=inv(eye(size(A))+A);
    MEAN=B*VAR;   
    logZ=trace(0.5*B'*B*VAR);
    logZ=-0.5*log(det(eye(size(A))+A))*size(B,1)+logZ ;          
end
