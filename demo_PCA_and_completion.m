path(path,'./Subroutines');
path(path,'./Functions');
%

fprintf(1,'Let us first try a regular PCA factorization \n');

m=5000;n=5000;%size of the vector U and V
Delta=1e-4;%Variance of the gaussian noise
RANK=3;%rank
fprintf(1,'Creating a %dx%d signal of rank %d ...\n',m,n,RANK);

fprintf(1,'Creating a %dx%d signal of rank %d ...\n',m,n,RANK);

U = randn(m,RANK);
V = randn(n,RANK);

fprintf(1,'... and adding a Gaussian noise with sigma  %f \n',sqrt(Delta));

%Adding noise!
Y=U*V'/sqrt(n)+sqrt(Delta)*randn(m,n);

%Computing the score and the inverse Fischer information
S=Y/Delta;Iinv=Delta;


%Calling the code
fprintf(1,'Running LowRAMP \n');
opt=LowRAMP_UV_Opt;
opt.nb_iter=20;
opt.damping=-1;%adaptive damping
opt.prior_u='Gauss';     opt.signal_u=U;       
opt.prior_v='Gauss';     opt.signal_v=V;       
opt.verbose_n=1;
tic
[ u_ample,v_ample ]  = LowRAMP_UV(S,Iinv,RANK,opt)    ;
toc;
fprintf(1,'Done! The Squared Reconstruction error on the matrix reads %e \n',mean2((u_ample*v_ample'/sqrt(n)-Y).^2));

subplot(1,3,1)
imshow(10*abs(Y))
title('The original matrix')

subplot(1,3,2)
imshow(10*abs((u_ample*v_ample'/sqrt(n))))
title('Recovered after factorization')

pause


fprintf(1,'Let us now hide 90 percent of all entries \n');
fraction=0.1;
Support=rand(size(Y))<fraction;
%Calling the code
fprintf(1,'Running LowRAMP \n');
tic
[ u_ample,v_ample ]  = LowRAMP_UV(S.*Support,Iinv/fraction,RANK,opt)    ;
toc;
fprintf(1,'Done! The Squared Reconstruction error on the matrix reads %e \n',mean2((u_ample*v_ample'/sqrt(n)-Y).^2));

subplot(1,3,3)
imshow(10*abs((u_ample*v_ample'/sqrt(n))))
title('Recovered after completion from 10% of the entries')

