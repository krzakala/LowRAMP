path(path,'./Subroutines');
m=500;n=100;%size of the vector U and V
Delta=1e-4;%Variance of the gaussian noise
RANK=3;%rank

fprintf(1,'Creating a signal of rank %d \n',RANK);
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
opt.damping=0.2;
opt.verbose_n=1;
tic
[ u_ample,v_ample ]  = LowRAMP_UV(S,Iinv,RANK,opt)    ;
toc;
disp('Squared Reconstruction error');
mean2((u_ample*v_ample'/sqrt(n)-Y).^2)

tic
[ u_amp,v_amp ] = LowRAMP_UV_completion(S,Iinv,ones(size(S)),RANK,opt)    ;
toc
disp('Squared Reconstruction error');
mean2((u_amp*v_amp'/sqrt(n)-Y).^2)
