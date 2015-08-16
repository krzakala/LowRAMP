path(path,'./Subroutines');
path(path,'./Functions');
%
m=5000;n=5000;%size of the vector U and V
m_m=250;n_n=250;

Delta=0.001;%Variance of the gaussian noise
RANK=1;%rank

fprintf(1,'Creating a %dx%d signal with a %dx%d submtrix hidden \n',m,n,m_m,n_n);
Y=[zeros(m-m_m,n); ones(m_m,n_n) zeros(m_m,n-n_n)];

subplot(2,2,1)
imshow(Y)
title('The original matrix')

%Adding noise!
fprintf(1,'adding a noise with std %f \n',sqrt(n)*sqrt(Delta));
W=Y/sqrt(n)+sqrt(Delta)*randn(m,n);
subplot(2,2,2)
imshow(W*100)
title('The noisy matrix')

%Computing the score and the inverse Fischer information
S=W/Delta;Iinv=Delta;

%Now options
opt=LowRAMP_UV_Opt;
opt.nb_iter=100;
opt.init_sol=4;%Update random>0
opt.damping=0.5;%adapatative damping
opt.prior_u='Rank1Binary';  
opt.prior_u_option=m_m/m;            
opt.prior_v='Rank1Binary';         
opt.prior_v_option=n_n/n;            
opt.verbose_n=1;

%Calling the code
fprintf(1,'Running LowRAMP \n');
tic
[ u_ample,v_ample ]  = LowRAMP_UV(S,Iinv,RANK,opt)    ;
toc;
%rounding to nearest integer
u_hat=round(u_ample);
v_hat=round(v_ample);
subplot(2,2,3)
imshow(u_hat*v_hat')
title('Reconstructed matrix')
subplot(2,2,4)
plot(v_ample,'b')
hold on
plot(u_ample,'r');
hold off
title('Identification of sub-matrix')

