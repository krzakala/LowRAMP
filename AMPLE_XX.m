function [ x ] = AMPLE_XX( S, Delta , RANK,opt)
% AMP Lowrank Estimation (AMPLE) is a Belief-Propagation based solver for XX' matrix factorization
% SYNTAX:
% [x ] = AMPLE_XX(S, Delta, RANK,opt)

% Inputs :
% S                     NxN matrix
% Delta                  Estimated noise
% opt -    structure containing  option fields
%    Details of the option:
%   .nb_iter            max number of iterations [default : 1000]
%   .verbose_n          print results every n iteration (0 -> never) [1]
%   .conv_criterion     convergence criterion [10^(-8)]
%   .signal             a solution to compare to while running
%   .init_sol           0 (zeros) 1 (random) 2 (SVD) 3 (solution) [1]
%   .damping            damping coefficient of the learning [-1]
%                       damping=-1 means adaptive damping, otherwise fixed
%   .prior              prior on the data [Community]
%                       One can use 'Gauss' of 'Community'
%
% Outputs :
% x                     final signal estimate as a column vector

    path(path,'./Subroutines');
    % Reading parameters
    if (nargin <= 3)
        opt = AMPLE_XX_Opt(); % Use default  parameters
    end        
    [m,n]=size(S);m=n;        

    % Definition of the prior
    switch     opt.prior;
      case    {'Community'}  
        disp    (['Community Clustering Prior'])
        Fun_a=@f_clust;
      case    {'Gauss'}  
        disp    (['Gaussian Prior'])
        Fun_a=@f_gauss;
      otherwise
        disp    (['unknown prior'])
        return;
    end
    
    % Initialize 
    x=zeros(n,RANK);
    switch     opt.init_sol
        case          1
            x=randn(n,RANK);
        case          2
            PR=sprintf('Use SVD as an initial condition ');
            [V,D] = eigs(S,RANK);
             x=V(:,1:RANK);
        case          3
            x=opt.signal+1e-4*randn(n,RANK);        
    end    
    x_old=zeros(n,RANK);
    x_V=zeros(RANK,RANK);
 
    A=zeros(RANK,RANK);
    B=zeros(n,RANK);
    diff=1;
    t=0;
    
    if (max(size(opt.signal)) < 2)
                PR=sprintf('T  Delta  diff    Free Entropy damp');
    else
                PR=sprintf('T  Delta  diff    Free Entropy damp    Error');
    end
    disp(PR);
    old_free_nrg=-realmax('double');

    while ((diff>opt.conv_criterion)&&(t<opt.nb_iter))    
        %Keep old variable
        A_old=A;
        B_old=B;
        
        %AMP iteration
        B_new=(S*x)/sqrt(n)-x_old*x_V/(Delta);
        A_new=x'*x/(n*Delta);

        %Keep old variables
        x_old=x;
        
        %Iteration with fixed damping or learner one
        pass=0;
        if (opt.damping==-1)
            damp=1;
        else
            damp=opt.damping;
        end
            
        while (pass~=1) 
            if (t>0)
                A=1./((1-damp)./A_old+damp./A_new);
                B=(1-damp)*B_old+damp*B_new;
            else
                A=A_new;
                B=B_new;
            end
            
            [x,x_V,logZ] = Fun_a(A,B);%Community prior                
            
            %Compute the Free Entropy
            minusDKL=logZ+0.5*n*trace(A*x_V)+trace(0.5*A*x'*x)-trace(x'*B)   ;  
            term_x=-trace((x'*x)*x_V)/(2*Delta);
            term_xx=sum(sum((x*x'.*S)))/(2*sqrt(n))-trace((x'*x)*(x'*x))/(4*n*Delta);
            free_nrg=(minusDKL+term_x+term_xx)/n;

            if (t==0) break;end
            if (opt.damping>0) break;end
            %Otherwise adapative damping
            if (free_nrg>old_free_nrg)
                old_free_nrg=free_nrg;
                pass=1;
            else
                 damp=damp/2;
                 if damp<1e-4;      break;end;
            end
        end
        
        diff=mean2(abs(x-x_old));
        if ((t==0)||(mod(t,opt.verbose_n)==0))
            PR=sprintf('%d %f %f %f %f',[t Delta diff free_nrg damp]);    
            if (~(max(size(opt.signal)) < 2))
                PR2=min(mean2((x-opt.signal).^2),mean2((-x-opt.signal).^2));
                PR=[PR PR2];
            end
            disp(PR);
        end
        t=t+1;
    end
end

