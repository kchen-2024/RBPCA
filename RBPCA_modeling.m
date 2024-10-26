
function [Q,Q_UCL,W,u,Z,lambda,V]=RBPCA_modeling(x_NOC,m,c,p)
n_NOC=size(x_NOC,2); 
[R,Z_Centering,W,u,Z]=BxBy_Norm(x_NOC',m,p,c);%random Bernoulli feature
[V,D]=eig(R);
[lambda,index] = sort(diag(D),'descend');
V = V(:,index);
a=max(find(lambda>mean(lambda)));  
V=V(:,1:a);
lambda=diag(lambda(1:a));
for k=1:n_NOC
Zk=Z_Centering(k,:)';
Q(k)=Zk'*(eye(m,m)-V*V')*Zk;
end
Q_UCL=UCL(Q,0.99);

function [r,z_Centering,W,u,z] = BxBy_Norm(x,m,p,c) %%random Bernoulli feature
    n=size(x,1); 
    d=size(x,2); 
    W= binornd(1,p,m,d);  
    u=2*pi*rand(m,1);
    B =-p/sqrt(c*p*(1-p)/2)*x*ones(d,m)+repmat(u',n,1);
    z=cos(x*W'./sqrt(c*p*(1-p)/2)+B);
    z_Centering=z-mean(z,1);  
    r=z_Centering'*z_Centering./(n-1);  
end
end