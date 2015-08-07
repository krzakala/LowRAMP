path(path,'./Subroutines');
n=2000;m=500;%size of the vector U and V
Delta=1e-2;%Variance of the gaussian noise
RANK=3;%rank

fprintf(1,'Creating a signal of rank %d \n',RANK);
V = zeros(m,RANK);
for i=1:m
    V(i,ceil(rand()*RANK))=1;
end
U = randn(n,RANK);

%Adding noise!
Y=U*V'/sqrt(n)+sqrt(Delta)*randn(n,m);

%Computing the score and the inverse Fischer information
S=Y/Delta;Iinv=Delta;


%Calling the code
fprintf(1,'Running AMP \n');
damp=-1;%adaptive damping
init_sol=0;%random init
tic
[ u_ample,v_ample ]  = AMPLE_UV(S,Iinv,RANK)    ;
toc;

