function [u,v] = LowRAMP_UV_completion( S, Delta , S_sup,RANK,opt)
% LowRAMP is a Low Rank factorization Belief-Propagation based solver for UV' matrix factorization
% This version is for completion when a part of S is actually missing    
% SYNTAX:
% [u,v] = LowRAMP_UV_completion(S, Delta, S_sup, RANK,opt)

% Inputs :
% S                     NxM matrix
% Delta                  Estimated noise
% S                     NxM matrix, support
% opt -    structure containing  option fields
%    Details of the option:
%   .nb_iter            max number of iterations [default : 1000]
%   .verbose_n          print results every n iteration (0 -> never) [1]
%   .conv_criterion     convergence criterion [10^(-8)]
%   .signal_U           a solution to compare to while running
%   .signal_V           a solution to compare to while running
%   .init_sol           0 (zeros) 1 (random) 2 (SVD) 3 (solution) [1]
%   .damping            damping coefficient of the learning [-1]
%                       damping=-1 means adaptive damping, otherwise fixed
%   .prior              prior on the data [Community]
%                       One can use 'Gauss' of 'Community'
%
% Outputs :
% u                     final signal estimate as a column vector
% v                     final signal estimate as a column vector

    path(path,'./Subroutines');
    % Reading parameters
    if (nargin <= 4)
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
    u_var=zeros(RANK,RANK);
    v_var=zeros(RANK,RANK);
    u_var_all=zeros(m,RANK,RANK);
    v_var_all=zeros(n,RANK,RANK);
    
    A_u_new=zeros(m,RANK,RANK);
    A_v_new=zeros(m,RANK,RANK);
    A_u=zeros(m,RANK,RANK);%I have m of them
    B_u=zeros(m,RANK);
    A_v=zeros(n,RANK,RANK);%I have n of them
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
        B_u_new=(S*v)/sqrt(n)-u_old*v_var/(Delta);
        B_v_new=(S'*u)/sqrt(n)-v_old*(m*u_var/n)/(Delta);
        
        for i=1:m  
            thisv=repmat(S_sup(i,:)',1,RANK).*v;
            A_u_new(i,:,:)=thisv'*thisv/(n*Delta);      
        end
        for i=1:n  
            thisu=repmat(S_sup(:,i),1,RANK).*u;
            A_v_new(i,:,:)=thisu'*thisu/(n*Delta);            
        end
        
        %Keep old variables
        u_old=u;
        v_old=v;
        
        %Iteration with fixed damping or learner one
        pass=0;
        if (opt.damping==-1)
            damp=1;
        else
            damp=opt.damping;
        end
         while (pass~=1) 
            if (t>0) 
                A_u=(1-damp)*A_u_old+damp*A_u_new;
                A_v=(1-damp)*A_v_old+damp*A_v_new;
                B_u=(1-damp)*B_u_old+damp*B_u_new;
                B_v=(1-damp)*B_v_old+damp*B_v_new;
            else
                A_u=A_u_new;                A_v=A_v_new;
                B_u=B_u_new;                B_v=B_v_new;
            end
                        
            logutot=0;u_var=zeros(RANK,RANK);
            for i=1:m  
                [u(i,:),u_var_all(i,:,:),logu] = Fun_u(squeeze(A_u(i,:,:)),B_u(i,:));
                u_var=u_var+squeeze(u_var_all(i,:,:));
                logutot=logutot+logu;
            end
            u_var=u_var/m;
            
            logvtot=0;
            for  i=1:n  
                [v(i,:),v_var_all(i,:,:),logv] = Fun_v(squeeze(A_v(i,:,:)),B_v(i,:));
                v_var=v_var+squeeze(v_var_all(i,:,:));
                logvtot=logvtot+logv;
            end
            v_var=v_var/n;
            
            free_nrg=logutot+logvtot;%This is a wrong formula, 
                                     %The correct one still needs
                                     %needs to be written :-(
            
            if (t==0)  delta_free_nrg=old_free_nrg-free_nrg;old_free_nrg=free_nrg; break; end
            if (opt.damping>=0)  delta_free_nrg=old_free_nrg-free_nrg;old_free_nrg=free_nrg; break;end
            %Otherwise adapative damping
            if (free_nrg>old_free_nrg)
                delta_free_nrg=old_free_nrg-free_nrg;old_free_nrg=free_nrg;
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
          %  break;
        end
        t=t+1;
    end
end


