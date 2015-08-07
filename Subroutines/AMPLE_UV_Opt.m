% AMPLE_UV_Opt  : Function to set AMPLE_UV_Opt to their default values
%
%    Details of the option:
%   .nb_iter            max number of iterations [default : 1000]
%   .verbose_n          print results every n iteration (0 -> never) [1]
%   .conv_criterion     convergence criterion [10^(-8)]
%   .signal_u             a solution to compare to while running
%   .signal_v             a solution to compare to while running
%   .init_sol           0 (zeros) 1 (random) 2 (SVD) 3 (solution) [1]
%   .damping            damping coefficient of the learning [-1]
%                       damping=-1 means adaptive damping, otherwise fixed
%   .prior_u              prior on the data [gauss]
%   .prior_v              prior on the data [Community]
%                       One can use 'Gauss' of 'Community'

function opt = AMPLE_UV_Opt()
       opt.nb_iter=1000;
       opt.verbose_n=1;
       opt.init_sol=1;
       opt.conv_criterion =10^(-6);
       opt.signal_u=[];
       opt.signal_v=[];
       opt.damping=-1;
       opt.prior_u='Gauss';        
       opt.prior_v='Community';        
end