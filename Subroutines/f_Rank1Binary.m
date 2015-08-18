function [MEAN,VAR,logZ] = f_Rank1Binary(A,B,rho)
   %Rank 1 0/1 prior
   %rho is the a priori fraction of 1
   if (nargin <= 2)
        rho=0.5; % Use default  parameters
   end   
   Weight=-0.5*A+B;
   pos=find(Weight>0);
   neg=setdiff([1:size(B,1)],pos);
   MEAN=zeros(size(B));
   MEAN(neg)=rho*exp(-0.5*A+B(neg))./(1-rho+rho*exp(-0.5*A+B(neg)));
   MEAN(pos)= rho./(rho+(1-rho)*exp(0.5*A-B(pos)));
   VAR=mean2(MEAN.*(1-MEAN));
   logZ=sum(log(1-rho+rho*exp(-0.5*A+B(neg))));   
   logZ=logZ+sum(-0.5*A+B(pos)+log(rho+(1-rho)*exp(0.5*A-B(pos))));
end
