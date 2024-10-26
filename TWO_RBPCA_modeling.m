
function [Q_stat,Q_UCL,W,u,Z,lambda,V,y_mea]=TWO_RBPCA_modeling(x_NOC,m,c,p,l)
n_NOC=size(x_NOC,2); 
[W,u,Z]=BxBy_Norm(x_NOC',m,p,c);%random Bernoulli feature
y=cell(1,n_NOC-l);
for t=l+1:n_NOC
y{t}=Z(t-l:t,:);%time-lagged matrix
end
for i=1:l+1
    for j=1:m
        s=0;
        for t=l+1:n_NOC
        s=s+y{t}(i,j);
        end
        y_mea(i,j)=s/(n_NOC-l);
    end
end
s=zeros(m,m);
for t=l+1:n_NOC
    y_Centering{t}=y{t}-y_mea; 
    s=s+y_Centering{t}'*y_Centering{t};
end
G=s./(n_NOC-l);
[V,D]=eig(G);
[lambda,index] = sort(diag(D),'descend');
V = V(:,index);
a=max(find(lambda>mean(lambda)));  
V=V(:,1:a);
lambda=diag(lambda(1:a));
%calculate statistic
for t=l+1:n_NOC
    for k=1:l+1
        Zk=y_Centering{t}(k,:)';
        Q(k)=Zk'*(eye(m,m)-V*V')*Zk;
    end
    Q_stat(t)=sum(Q);
end
Q_stat(find(Q_stat==0))=[];
Q_UCL=UCL(Q_stat,0.991);

function [W,u,z] = BxBy_Norm(x,m,p,c) 
    n=size(x,1); 
    d=size(x,2); 
    W= binornd(1,p,m,d);  
    u=2*pi*rand(m,1);
    B =-p/sqrt(c*p*(1-p)/2)*x*ones(d,m)+repmat(u',n,1);
    z=cos(x*W'./sqrt(c*p*(1-p)/2)+B);
end
end