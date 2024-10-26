function a=UCL(A,alpha)
n=length(A);
sigma=std(A);
h=1.06*sigma*n^(-1/5);
a=ksdensity(A,alpha,'Bandwidth',h,'Function','icdf');
end