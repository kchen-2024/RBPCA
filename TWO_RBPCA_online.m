
function [Q_stat]=TWO_RBPCA_online(x_new,B,U,Z,lambda,V,y_mea,c,p,l)
    m=size(V,1);
    d=size(x_new,1);
    n=size(x_new,2);
    for i=1:n
        x=x_new(:,i);
        z_new(:,i)=cos(B*x./sqrt(c*p*(1-p)/2)-p/sqrt(c*p*(1-p)/2)*ones(m,d)*x+U); %m*1
    end
    Z=[Z;z_new'];
    y=Z(end-l:end,:);%time-lagged matrix
    y_Centering=y-y_mea;%mean-centered
    for k=1:l+1
        Zk=y_Centering(k,:)';
        Q(k)=Zk'*(eye(m,m)-V*V')*Zk;
    end
    Q_stat=sum(Q);
end