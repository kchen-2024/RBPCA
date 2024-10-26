clear all
clc
data_train=load('D:\ServerMachineDataset\train\machine-1-1.txt');
train0=find(all(data_train==0,1));
data_test=load('D:\ServerMachineDataset\test\machine-1-1.txt');
test0=find(all(data_test==0,1));
label_test=load('D:\ServerMachineDataset\test_label\machine-1-1.txt');
queshi=unique([train0,test0]);
data_train(:,queshi)=[];
data_test(:,queshi)=[];

for jjj=1:500
    x_NOC=data_test(14351:15250,:)';
    x_fault=data_test(15251:16150,:)';
    X=[x_NOC,x_fault];
    ori_label=label_test(15251:16150);
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
    m=300;
    p=0.05;
    alpha=0.9;
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