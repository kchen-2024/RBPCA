    
function [Q]=RBPCA_online(x_new,B,U,Z,lambda,V,c,p)
    m=size(V,1);
    d=length(x_new);
    z_new=cos(B*x_new./sqrt(c*p*(1-p)/2)-p/sqrt(c*p*(1-p)/2)*ones(m,d)*x_new+U); 
    z_new_Centering=z_new-mean(Z)';
    Q=z_new_Centering'*(eye(m,m)-V*V')*z_new_Centering;
end  