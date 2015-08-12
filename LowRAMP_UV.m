function [u,v] = LowRAMP_UV( S, Delta , RANK,opt)
% LowRAMP is a Low Rank factorization Belief-Propagation based solver for UV' matrix factorization
% SYNTAX:
% [u,v] = LowRAMP_UV(S, Delta, RANK,opt)

% Inputs :
% S                     NxM matrix
% Delta                  Estimated noise
% opt -    structure containing  option fields
%    Details of the option:
%   .nb_iter            max number of iterations [default : 1000]
%   .verbose_n          print results every n iteration (0 -> never) [1]
%   .conv_criterion     convergence criterion [10^(-8)]
%   .signal_u           a solution to compare to while running
%   .signal_v           a solution to compare to while running
%   .init_sol           0 (zeros) 1 (random) 2 (SVD) 3 (solution) [1]
%   .damping            damping coefficient of the learning [-1]
%                       damping=-1 means adaptive damping, otherwise fixed
%   .prior_u              prior on the data [gauss]
%   .prior_v              prior on the data [Community]
%                       One can use 'Gauss' of 'Community'
%
% Outputs :
% u                     final signal estimate as a column vector
% v                     final signal estimate as a column vector

    path(path,'./Subroutines');
    % Reading parameters
    if (nargin <= 3)
        opt = LowRAMP_UV_Opt(); % Use default  parameters
    end        
    [m,n]=size(S);
    
    % Definition of the prior
    switch     opt.prior_u;
      case    {'Community'}  
        disp    (['U: Community Clustering Prior'])
        Fun_u=@f_clust;
      case    {'Gauss'}  
        disp    (['U: Gaussian Prior'])
        Fun_u=@f_gauss;
      otherwise
        disp    (['unknown prior'])
        return;
    end
    switch     opt.prior_v;
      case    {'Community'}  
        disp    (['V: Community Clustering Prior'])
        Fun_v=@f_clust;
      case    {'Gauss'}  
        disp    (['V: Gaussian Prior'])
        Fun_v=@f_gauss;
      otherwise
        disp    (['unknown prior'])
        return;
    end
    
    % Initialize
    u=zeros(m,RANK);  
    v=ones(n,RANK)/RANK;  
    switch     opt.init_sol
        case          1
            u=randn(m,RANK);
            v=rand(n,RANK);    
        case          2
            PR=sprintf('Use SVD as an initial condition ');
          [U,SS,V] = svds(S,RANK);
           u=U(:,1:RANK);
           v=V(:,1:RANK);
        case          3
           u=u_truth+1e-4*randn(m,RANK);
           v=v_truth+1e-4*randn(n,RANK);    
    end   
           
    u_old=zeros(m,RANK);
    v_old=zeros(n,RANK);
    u_var=zeros(RANK,RANK);u_var_old=zeros(RANK,RANK);
    v_var=zeros(RANK,RANK);v_var_old=zeros(RANK,RANK);

    A_u=zeros(RANK,RANK);
    B_u=zeros(m,RANK);
    A_v=zeros(RANK,RANK);
    B_v=zeros(n,RANK);
    
    diff=1;
    t=0;
    
    if (max(size(opt.signal_u)) < 2)
                PR=sprintf('T  Delta  diff    Free Entropy damp');
    else
                PR=sprintf('T  Delta  diff    Free Entropy damp    Error_u Error_ v');
    end
    disp(PR);
    old_free_nrg=-realmax('double');delta_free_nrg=0;
    

    while ((diff>opt.conv_criterion)&&(t<opt.nb_iter))    
        %Keep old variable
        A_u_old=A_u;        A_v_old=A_v;
        B_u_old=B_u;        B_v_old=B_v;      
        
        %AMP iteration
        B_u_new=(S*v)/sqrt(n)-u_old*v_var_old/(Delta);
        A_u_new=v'*v/(n*Delta);
        B_v_new=(S'*u)/sqrt(n)-v_old*(m*u_var_old/n)/(Delta);
        A_v_new=u'*u/(n*Delta);
        
        %Keep old variables
        u_old=u;u_var_old=u_var;
        v_old=v;v_var_old=v_var;
        
        %Iteration with fixed damping or learner one
        pass=0;
        if (opt.damping==-1)
            damp=1;
        else
            damp=opt.damping;
        end
         while (pass~=1) 
            if (t>0)
                %here should be corrected with ACTUAL matrix inversion!
                A_u=(1-damp)*A_u_old+damp*A_u_new;
                A_v=(1-damp)*A_v_old+damp*A_v_new;
                B_u=(1-damp)*B_u_old+damp*B_u_new;
                B_v=(1-damp)*B_v_old+damp*B_v_new;
            else
                A_u=A_u_new;                A_v=A_v_new;
                B_u=B_u_new;                B_v=B_v_new;
            end
                
            [u,u_var,logu] = Fun_u(A_u,B_u);
            [v,v_var,logv] = Fun_v(A_v,B_v);            
            
  
            %Compute the Free Entropy
            minusDKL_u=logu+0.5*m*trace(A_u*u_var)+trace(0.5*A_u*u'*u)-trace(u'*B_u);   
            minusDKL_v=logv+0.5*n*trace(A_v*v_var)+trace(0.5*A_v*v'*v)-trace(v'*B_v);   
            term_u=-trace((u'*u)*v_var)/(2*Delta);
            term_v=-(m/n)*trace((v'*v)*u_var)/(2*Delta);%this is such that A_u and B_u gets a factor m/n
            term_uv=sum(sum((u*v'.*S)))/(sqrt(n))-trace((u'*u)*(v'*v))/(2*n*Delta);
            free_nrg=(minusDKL_u+minusDKL_v+term_u+term_v+term_uv)/n;
                      
            if (t==0)  delta_free_nrg=old_free_nrg-free_nrg;old_free_nrg=free_nrg; break; end
            if (opt.damping>=0)  delta_free_nrg=old_free_nrg-free_nrg;old_free_nrg=free_nrg; break;end
            %Otherwise adapative damping
            if (free_nrg>old_free_nrg)
                delta_free_nrg=old_free_nrg-free_nrg;
                old_free_nrg=free_nrg;
                pass=1;
            else
                 damp=damp/2;
                 if damp<1e-4;   delta_free_nrg=old_free_nrg-free_nrg;old_free_nrg=free_nrg;   break;end;
            end                                 
        end
        
        diff=mean2(abs(v-v_old))+mean2(abs(u-u_old));
        if ((t==0)||(mod(t,opt.verbose_n)==0))
        PR=sprintf('%d %f %f %f %f',[t Delta diff free_nrg damp]);              
            if (~(max(size(opt.signal_u)) < 2))
                PR2=sprintf(' %e %e',[min(mean2((u-opt.signal_u).^2),mean2((-u-opt.signal_u).^2)) min(mean2((v-opt.signal_v).^2),mean2((-v-opt.signal_v).^2))]);
                PR=[PR PR2];
            end
            disp(PR);
        end
        if (abs(delta_free_nrg/free_nrg)<opt.conv_criterion)        
            break;
        end
        t=t+1;   
    end
end


