% LowRAMP_XX_Opt  : Function to set LowRAMP_XX_Opt to their default values
%
%    Details of the option:
%   .nb_iter            max number of iterations [default : 1000]
%   .verbose_n          print results every n iteration (0 -> never) [1]
%   .conv_criterion     convergence criterion [10^(-8)]
%   .signal             a solution to compare to while running
%   .init_sol           0 (zeros) 1 (random) 2 (SVD) 3 (solution) [1]
%   .damping            damping coefficient of the learning [-1]
%                       damping=-1 means adaptive damping, otherwise fixed
%   .prior              prior on the data [Community]
%                       One can use 'Gauss','Community' or 'Rank1Binary'
%   .prior_option       If using Rank1Binary, one can pass the fraction rho
%                       of values equal to 1


function opt = LowRAMP_XX_Opt()
       opt.nb_iter=1000;
       opt.verbose_n=1;
       opt.init_sol=1;
       opt.conv_criterion =10^(-6);
       opt.signal=[];
       opt.damping=-1;
       opt.prior='Community';       
       opt.prior_option=[];        
end