clear all
clc

for jjj=1:1
    NOC=NOC_modeling(1200); 
    x_NOC=NOC(:,1:1000);
    x_fault=[NOC(:,1001:1200),Fault1(300)];%Fault samples (the first 200 are normal and the last 300 are faults)
    X=[x_NOC,x_fault];
    ori_label=[zeros(200,1);ones(300,1)];
    dim=size(x_NOC,1);%dimension
    n_NOC=size(x_NOC,2);%sample size under normal operating conditions
    n_fault=size(x_fault,2); %sample size of faults

    tic
    for i=1:n_NOC
        for j=i+1:n_NOC
            corre(i,j)=corr(x_NOC(:,i),x_NOC(:,j));
        end
    end
    corre(n_NOC,:)=zeros(1,n_NOC);

    w=500; %window width
    [B,IX] = sort(corre(:),'descend');
    [I,J] = ind2sub(size(corre), IX);
    shanchu=zeros(1,100000);
    for i=1:100000
        in=min(I(i),J(i));
        if  ismember(in,shanchu)==0
            shanchu(i)=in;
        end
        if nnz(shanchu)==(n_NOC-w)
            break
        end
    end
    shanchu(find(shanchu==0))=[];
    x_NOC(:,shanchu)=[];
    n_NOC=w;

    %normalize data
    MEAN=mean(x_NOC')';
    VAR=std(x_NOC')';
    for i=1:n_NOC
        x_NOC(:,i)=(x_NOC(:,i)-MEAN)./VAR;
    end
    for i=1:n_fault
        x_fault(:,i)=(x_fault(:,i)-MEAN)./VAR;
    end

    %first stage: modeling
    c=40*dim;
    m=150;
    p=0.05;
    alpha=0.8;
    [Q,Q_UCL,W,u,Z,Z_Centering,V,R,correlation,update_yuzhi]=MV_RBPCA_modeling(x_NOC,m,c,p,alpha);
    MT(jjj)=toc;
    i1=0;
    XZ1=Z';
    XZ1_Centering=Z_Centering';
    R1=R;
    Q_UCL1=Q_UCL;

    %second stage: online monitoring
    for k=1:n_fault
        x_new=x_fault(:,k);
        tic
        [result_Q1(k),XZ1,XZ1_Centering,R1,Q_UCL1,i1]=MV_RBPCA_online(i1,update_yuzhi,XZ1,XZ1_Centering,R1,Q_UCL1,x_new,W,u,Z,V,c,p,n_NOC);
        OT(jjj)=toc;
    end

    %calculate fault detection rate and false alarm rate
    num_data(jjj)=i1;
    pre_label=result_Q1;
    fault=find(ori_label==1);
    normal=find(ori_label==0);
    far=0;
    fdr=0;
    for j=1:n_fault
        if ori_label(j)==0 && pre_label(j)==1
            far=far+1;
        end
        if ori_label(j)==1 && pre_label(j)==1
            fdr=fdr+1;
        end
    end
    FAR_Q_data(jjj)=far/length(normal);
    FDR_Q_data(jjj)=fdr/length(fault);
end
mean(FAR_Q_data)
mean(FDR_Q_data)
mean(num_data)
mean(MT)
mean(OT)

%first dataset: the numerical example
function x_NOC=NOC_modeling(n)
for i=1:n
    t=0.01+1.99*rand(1);
    e1=sqrt(0.01)*randn(1);
    e2=sqrt(0.01)*randn(1);
    e3=sqrt(0.01)*randn(1);
    x1=t+e1;
    x2=t*t-3*t+e2;
    x3=-t*t*t+3*t*t+e3;
    x_NOC(:,i)=[x1;x2;x3];
end
end

%Fault 1
function x=Fault1(n)
for i=1:n
    t=0.01+1.99*rand(1);
    e1=sqrt(0.01)*randn(1);
    e2=sqrt(0.01)*randn(1);
    e3=sqrt(0.01)*randn(1);
    x1=t+e1-0.5;
    x2=t*t-3*t+e2;
    x3=-t*t*t+3*t*t+e3;
    x(:,i)=[x1;x2;x3];
end
end

%Fault 2
function x=Fault2(n)
for i=1:n
    t=0.01+1.99*rand(1);
    e1=sqrt(0.01)*randn(1);
    e2=sqrt(0.01)*randn(1);
    e3=sqrt(0.01)*randn(1);
    x1=t+e1;
    x2=t*t-3*t+e2+0.01*i;
    x3=-t*t*t+3*t*t+e3;
    x(:,i)=[x1;x2;x3];
end
end