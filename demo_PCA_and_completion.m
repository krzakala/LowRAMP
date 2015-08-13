path(path,'./Subroutines');
path(path,'./Functions');
%
m=1000;n=1000;%size of the vector U and V
Delta=1e-4;%Variance of the gaussian noise
RANK=3;%rank

fprintf(1,'Creating a %dx%d signal of rank %d \n',m,n,RANK);
U = randn(m,RANK);
V = randn(n,RANK);

%Adding noise!
Y=U*V'/sqrt(n)+sqrt(Delta)*randn(m,n);

%Computing the score and the inverse Fischer information
S=Y/Delta;Iinv=Delta;


%Calling the code
fprintf(1,'Running LowRAMP \n');
opt=LowRAMP_UV_Opt;
opt.damping=-1;%adaptive damping
opt.prior_u='Gauss';     opt.signal_u=U;       
opt.prior_v='Gauss';     opt.signal_v=V;       
opt.verbose_n=1;
tic
[ u_ample,v_ample ]  = LowRAMP_UV(S,Iinv,RANK,opt)    ;
toc;
disp('Squared Reconstruction error on the matrix');
mean2((u_ample*v_ample'/sqrt(n)-Y).^2)


