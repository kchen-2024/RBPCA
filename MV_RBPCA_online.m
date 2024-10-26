%i:number of updates
%delta：threshold of update
%XZ：data in the window
%R:covariance matrix computed from previous window data
function [result_Q,XZ,XZ_Centering,R,Q_UCL,i]=MV_RBPCA_online(i,delta,XZ,XZ_Centering,R,Q_UCL,x_new,W,u,Z,V,c,p,n_NOC)
m=size(V,1);
d=length(x_new);
z_new=cos(W*x_new./sqrt(c*p*(1-p)/2)-p/sqrt(c*p*(1-p)/2)*ones(m,d)*x_new+u);
z_new_Centering=z_new-mean(XZ')';
Q_new=z_new_Centering'*(eye(m,m)-V*V')*z_new_Centering;
if Q_new<Q_UCL
    result_Q=0;
    for j=1:n_NOC
        z_new_hat(j)=dot(z_new_Centering,XZ_Centering(:,j));
    end
    juli=abs(norm(z_new_hat)-norm(z_new_Centering));
    if juli>delta
        i=i+1;
        XZ=[XZ,z_new];
        XZ(:,1)=[];
        XZ_Centering=XZ-mean(XZ,2);
        R=XZ_Centering*XZ_Centering'./(n_NOC-1);
        [V,D]=eig(R);
        [lambda,index] = sort(diag(D),'descend');
        V = V(:,index);
        a=max(find(lambda>mean(lambda)));
        V=V(:,1:a);
        lambda=diag(lambda(1:a));
        for k=1:n_NOC
            Q(k)=XZ_Centering(:,k)'*(eye(m,m)-V*V')*XZ_Centering(:,k);
        end
        Q_UCL=UCL(Q,0.991);
    end
else
    result_Q=1;
end
end