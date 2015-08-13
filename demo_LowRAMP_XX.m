path(path,'./Subroutines');
path(path,'./Functions');
%
n=2500;%size of the vector X
RANK=3;%rank
p=0.5;
Deltaeff=0.05;

Delta=sqrt(p*(1-p)/Deltaeff);
pout = p - Delta/(RANK*sqrt(n));
pin = p + (1-1/RANK)*Delta/sqrt(n);

fprintf(1,'Creating a %dx%d signal of rank %d \n',n,n,RANK);
 X = zeros(n,RANK);
for i=1:n
    X(i,ceil(rand()*RANK))=1;
end

%creating the adjacency matrix
random1=triu(rand(n,n)<pin,1);
random1=random1 +random1';
random2=triu(rand(n,n)<pout,1);
random2=random2 +random2';
    
A=X*X'.*random1+(1-X*X').*random2;
S=(Delta/pout)*A - (1-A)*Delta/(1-pout);
mu=(pin-pout)*sqrt(n);
Iinv=(mu*mu/(pout*(1-pout)))^-1;

%Calling the code
fprintf(1,'Running LowRAMP \n');
tic
[x_ample] = LowRAMP_XX(S,Iinv,RANK)    ;
toc;
