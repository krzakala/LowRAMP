function [MEAN,VAR,logZ] = f_gauss(A,B,var_gauss)
    %Rank K Gaussian prior
    %var_gauss is the a priori variance of the prior
   if (nargin <= 2)
        var_gauss=1; % Use default  parameters
   end
   VAR=inv((1./var_gauss)*eye(size(A))+A);
   MEAN=B*VAR;   
   logZ=trace(0.5*B'*B*VAR);
   logZ=-0.5*log(det(var_gauss*eye(size(A))+A))*size(B,1)+logZ ;          
end
