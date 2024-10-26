clear all
clc

data_train=load('D:\ServerMachineDataset\train\machine-1-1.txt');
data_test=load('D:\ServerMachineDataset\test\machine-1-1.txt');
label_test=load('D:\ServerMachineDataset\test_label\machine-1-1.txt');
x_NOC=data_test(14351:15250,:)';
x_fault=data_test(15251:16150,:)';
queshi=unique([find(all(x_NOC==0,2));find(all(x_fault==0,2))]);
x_NOC(queshi,:)=[];
x_fault(queshi,:)=[];
X=[x_NOC,x_fault];
ori_label=label_test(15251:16150);
dim=size(x_NOC,1);%dimension
n_NOC=size(x_NOC,2);%sample size under normal operating conditions
n_fault=size(x_fault,2); %sample size of faults

% construct time-lagged vectors
tic
l=8;
for i=1:l+1
    for j=1:n_NOC+n_fault-l
        X_lagged(dim*(i-1)+1:dim*i,j)=X(:,j+i-1);
    end
end
x_lagged=X_lagged(:,1:n_NOC-l);
x_fault_lagged=X_lagged(:,n_NOC-l+1:n_NOC+n_fault-l);
MEAN_lagged=mean(x_lagged')';
VAR_lagged=std(x_lagged')';
for i=1:n_NOC-l
    x_lagged(:,i)=(x_lagged(:,i)-MEAN_lagged)./VAR_lagged;
end
for i=1:n_fault-1
    x_fault_lagged(:,i)=(x_fault_lagged(:,i)-MEAN_lagged)./VAR_lagged;
end

time_l=toc;

%normalize data
MEAN=mean(x_NOC')';
VAR=std(x_NOC')';
for i=1:n_NOC
    x_NOC(:,i)=(x_NOC(:,i)-MEAN)./VAR;
end
for i=1:n_fault
    x_fault(:,i)=(x_fault(:,i)-MEAN)./VAR;
end

for jjj=1:500
    %first stage: modeling
    c=40*dim;
    m=300;
    p=0.05;
    tic
    [Qtrain,Q_UCL_1,B,U,Z,lambda,V]=RBPCA_modeling(x_NOC,m,c,p);
    t_train_RBPCA(jjj)=toc;
    tic
    [Qtrain_lagged,Q_UCL_lagged,B_lagged,U_lagged,Z_lagged,lambda_lagged,V_lagged]=RBPCA_modeling(x_lagged,m,c,p);
    t_train_DRBPCA(jjj)=toc;
    tic
    [Qtrain_2,Q_UCL_2,B2,U2,Z2,lambda2,V2,y_mea]=TWO_RBPCA_modeling(x_NOC,m,c,p,8);
    t_train_2DRBPCA(jjj)=toc;
    %second stage: online monitoring
    for k=1:n_fault
        x_new=x_fault(:,k); 
        x_new_lagged=x_fault_lagged(:,k);
        tic
        [Q1(k)]=RBPCA_online(x_new,B,U,Z,lambda,V,c,p);
        t_test_RBPCA(jjj,k)=toc;
        tic
        [Q_lagged(k)]=RBPCA_online(x_new_lagged,B_lagged,U_lagged,Z_lagged,lambda_lagged,V_lagged,c,p);
        t_test_DRBPCA(jjj,k)=toc;
    end
    for k=1:n_fault
        x_new=x_fault(:,1:k); 
        tic
        [Q2(k)]=TWO_RBPCA_online(x_new,B2,U2,Z2,lambda2,V2,y_mea,c,p,8);
        t_test_2DRBPCA(jjj,k)=toc;
    end
    indicator=[Q1;Q_lagged;Q2]; 
    threshold=[Q_UCL_1,Q_UCL_lagged,Q_UCL_2];
    %calculate fault detection rate and false alarm rate
    for i=1:3
        result=find(indicator(i,:)>threshold(i));
        pre_label=zeros(n_fault,1);
        pre_label(result)=1;
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
        FAR(jjj,i)=far/length(normal);
        FDR(jjj,i)=fdr/length(fault);
    end
end

disp('RBPCA DRBPCA 2D-RBPCA')
mean(FDR,1)
mean(FAR,1)
mean(time_l)
time_train_RBPCA=mean(t_train_RBPCA)
time_train_DRBPCA=mean(t_train_DRBPCA)
time_train_2DRBPCA=mean(t_train_2DRBPCA)
time_test_RBPCA=mean(mean(t_test_RBPCA))
time_test_DRBPCA=mean(mean(t_test_DRBPCA))
time_test_2DRBPCA=mean(mean(t_test_2DRBPCA))