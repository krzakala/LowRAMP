path(path,'./Subroutines');
m=4000;n=200;%size of the vector U and V
Delta=1e-4;%Variance of the gaussian noise
fraction=1;%fraction of observed entries : half!
RANK=3;%rank

fprintf(1,'Creating a signal of rank %d \n',RANK);
u = randn(m,RANK);
v = randn(n,RANK);

%Adding noise!
noise=randn(m,n);
Y_sup=rand(m,n)<fraction;

Y=u*v'/sqrt(n)+noise*sqrt(Delta);
S=(Y_sup.*Y)/(Delta*sqrt(n));
Iinv=Delta;

%I would like to have a regularization here...

fprintf(1,'Running LowRAMP \n');
opt=AMPLE_UV_Opt;
opt.damping=0.5;
opt.prior_u='Gauss';        
opt.prior_v='Gauss';        
tic
[ u_amp,v_amp ] = LowRAMP_UV_completion(S,Iinv,Y_sup,RANK,opt)    ;
toc
