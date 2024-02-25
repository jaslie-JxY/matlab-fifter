%�������ϵͳЭͬ��λPF�˲��㷨
%״̬���̣�X(k+1)=FX(k)+G(U(k)+W(k))
%λ�ù۲ⷽ�̣�Z=������빫ʽ+V(k)
%��λ�۲ⷽ�̣��۲�Ƕ�

clc;
clear;
close all;

%��ʼ��    
T=1;                 %��������
N=100;                %��������

X1=zeros(3,N);
X2=zeros(3,N);      %3��N���󣬼�¼������״̬�仯
X11=zeros(3,N);
X22=zeros(3,N);     %��¼��������Ļ�����״̬�仯

theta1=36.87*pi/180;
theta2=53.13*pi/180;
v1=1;
v2=2;
palstance=0;           %������λ��״̬��ʼ��

X1(:,1)=[1,1,theta1];    
X2(:,1)=[1,2,theta2];    
X11(:,1)=[1,1,theta1];
X22(:,1)=[1,2,theta2];    %�����һ�г�ʼ��������λ��״̬,x,yλ�ã��˶�����

G1=[T*cos(theta1),0;T*sin(theta1),0;0,T];      
G2=[T*cos(theta2),0;T*sin(theta2),0;0,T];   %3��2����������������

U1=[v1;palstance];
U2=[v2;palstance];   %�������˶�״̬��ʼ��

Q=diag([1,1]);       %������������Խ��󣩶Խ���Ԫ�طֱ�Ϊ�ٶ����̺ͽ��ٶ�����
R1=1;                 %����۲ⷽ��
R2=1;                 %�Ƕȹ۲ⷽ��
P1=eye(3);            
P2=eye(3);              %Э�������ʼ��

Z00=zeros(1,N);
Z01=zeros(1,N);
Z02=zeros(1,N);            %�۲����ݴ洢����

%״̬����
for t=2:N
    X1(:,t)=X1(:,t-1)+G1*U1;
    X2(:,t)=X2(:,t-1)+G2*U2;     %�����������״̬���˶�����
end

for t=2:N  
    W1=sqrtm(Q)*randn(2,1);    
    W2=sqrtm(Q)*randn(2,1);      %����2��1�ķ��ϱ�׼��̬�ֲ��������,���ó˷��ı䷽��õ���������
    U1=U1+W1;
    U2=U2+W2;
    X11(:,t)=X1(:,t-1)+G1*U1;
    X22(:,t)=X2(:,t-1)+G2*U2;
    theta1=X11(3,t);
    theta2=X22(3,t);
    G1=[T*cos(theta1),0;T*sin(theta1),0;0,T];      
    G2=[T*cos(theta2),0;T*sin(theta2),0;0,T];
end

%�۲ⷽ�̣�����ֵ���������Լ��Ƕ�
for t=1:N
    Z00(:,t)=sqrt (       (          X2(2,t)-X1(2,t)           )^2   +    (X2(1,t)-X1(1,t)    )^2     )+sqrt(R1)*randn;  %����ֵ
    Z01(:,t)=atan(  (X2(2,t)-X1(2,t))/(X2(1,t)-X1(1,t)) -X1(3,t))+sqrt(R2)*randn;         
    Z02(:,t)=atan(  (X2(2,t)-X1(2,t))/(X2(1,t)-X1(1,t)) +pi-X2(3,t)  )+sqrt(R2)*randn;
end
Z1=[Z00;Z01];
Z2=[Z00;Z02];

%�������˶�״̬��ʼ��
theta1=36.87*pi/180;
theta2=53.13*pi/180;
G1=[T*cos(theta1),0;T*sin(theta1),0;0,T];      
G2=[T*cos(theta2),0;T*sin(theta2),0;0,T];
U1=[v1;palstance];
U2=[v2;palstance];   

%״̬����
for t=2:N
    X1(:,t)=X1(:,t-1)+G1*U1;
    X2(:,t)=X2(:,t-1)+G2*U2;     %�����������״̬���˶�����
end

for t=2:N  
    W1=sqrtm(Q)*randn(2,1);    
    W2=sqrtm(Q)*randn(2,1);    %����2��1�ķ��ϱ�׼��̬�ֲ��������,���ó˷��ı䷽��õ�������
    U1=U1+W1;
    U2=U2+W2;
    X11(:,t)=X1(:,t-1)+G1*U1;
    X22(:,t)=X2(:,t-1)+G2*U2;
    theta1=X11(3,t);
    theta2=X22(3,t);
    G1=[T*cos(theta1),0;T*sin(theta1),0;0,T];      
    G2=[T*cos(theta2),0;T*sin(theta2),0;0,T];
end

%λ�ù۲ⷽ�̣����ɹ۲�����
for t=1:N
    Z00(:,t)=sqrt (       (          X2(2,t)-X1(2,t)           )^2   +    (X2(1,t)-X1(1,t)    )^2     )+sqrt(R1)*randn;  %����ֵ
    Z01(:,t)=atan(  (X2(2,t)-X1(2,t))/(X2(1,t)-X1(1,t)) -X1(3,t))+sqrt(R2)*randn;         
    Z02(:,t)=atan(  (X2(2,t)-X1(2,t))/(X2(1,t)-X1(1,t)) +pi-X2(3,t)  )+sqrt(R2)*randn;
end
Z1=[Z00;Z01];
Z2=[Z00;Z02];

theta1=36.87*pi/180;
theta2=53.13*pi/180;
G1=[T*cos(theta1),0;T*sin(theta1),0;0,T];      
G2=[T*cos(theta2),0;T*sin(theta2),0;0,T];
U1=[v1;palstance];
U2=[v2;palstance];   %�������˶�״̬��ʼ��

%�������Ӽ���
n=10000;                    %���Ӽ�������
XOLD1=zeros(3,n);      
XNEW1=zeros(3,n);
XPF1=zeros(3,N);
weight1=zeros(1,n);  
ZZ=zeros(1,n);          %�۲�ֵԤ�����
XOLD2=zeros(3,n);      
XNEW2=zeros(3,n);
XPF2=zeros(3,N);
weight2=zeros(1,n);        %Ȩ��
XPF1(:,1)=X1(:,1);
XPF2(:,1)=X2(:,1);

%���ó�ֵ�������������ֵ�㹻����
for i=1:n
    XOLD1(:,i)=X1(:,1);
    XOLD2(:,i)=X2(:,1);
    weight1(i)=1/n;
    weight2(i)=1/n;
end

%PF�˲�
for i=2:N
    %Ԥ�ⲽ,���Ӽ��ϸ���Ԥ��
    for j=1:n
        w1=normrnd(0,1,2,1);    
        w2=normrnd(0,1,2,1);     %���ӱ�׼��̬�ֲ���3��1�������
        XOLD1(:,j)=XOLD1(:,j)+G1*(U1+w1);
        XOLD2(:,j)=XOLD2(:,j)+G2*(U2+w2);   %���Ӽ������˸���״̬����Ԥ����һʱ�����Ӽ���״̬
    end     
    %���²�������Ȩ�ظ���
    for j=1:n
        detay=XOLD2(2,j)-XOLD1(2,j);
        detax=XOLD2(1,j)-XOLD1(1,j);
        ZD0=sqrt(detay^2+detax^2);                      %����״̬Ԥ��ֵ�����۲�ֵ
        ZD01=atan(detay/detax)-XOLD1(3,j);
        ZD02=atan(detay/detax)+pi-XOLD2(3,j);
        dz1=[abs(Z00(:,i)-ZD0);abs(Z01(:,i)-ZD01)];         
        dz2=[abs(Z00(:,i)-ZD0);abs(Z02(:,i)-ZD02)];       %�������Ӽ���Ԥ��۲�ֵ��ʵ����ʵ�۲�ֵ��ֵ�ľ���ֵ����Ȩ��
        weight1(j)=sqrt(2*pi)*normpdf(dz1(1),0,1)*(sqrt(2*pi)*normpdf(dz1(2),0,1));
        weight2(j)=sqrt(2*pi)*normpdf(dz2(1),0,1)*(sqrt(2*pi)*normpdf(dz2(2),0,1));         %Ȩ�ظ���
    end
    %Ȩ�ع�һ��
    weight1=weight1/sum(weight1);
    weight2=weight2/sum(weight2);
    %�ز���
    c1=zeros(1,n);
    c2=zeros(1,n);
    c1(1)=weight1(1);
    c2(1)=weight2(1);        
    for j=2:n
        c1(j)=c1(j-1)+weight1(j);
        c2(j)=c2(j-1)+weight2(j);
    end                                  %����Ȩ�ػ�������
    for j=1:n
        a1=unifrnd(0,1);       %���ȷֲ������       
        for k=1:n
            if(a1<c1(k))
                XNEW1(:,j)=XOLD1(:,k);        %��������������ĸ��������Ӧ���ӽ��и���
                break;
            end
        end
    end
    for j=1:n
        a2=unifrnd(0,1);             
        for k=1:n
            if(a2<c2(k))
                XNEW2(:,j)=XOLD2(:,k);        
                break;
            end
        end
    end
    %�µ����Ӹ��Ƹ������Ӽ���
    XOLD1=XNEW1;
    XOLD2=XNEW2;
    for j=1:n
        weight1(j)=1/n;
        weight2(j)=1/n;               %Ȩ��������Ϊ1/n
    end
    XPF1(:,i)=sum(XNEW1,2)/n;
    XPF2(:,i)=sum(XNEW2,2)/n;         %�������Ӽ��ϵľ�ֵ��Ϊ�˲�ֵ
end

%������
PFpositionerror1=sqrt( (XPF1(1,:)-X1(1,:)).^2 + (XPF1(2,:)-X1(2,:)).^2   );
PFpositionerror2=sqrt( (XPF2(1,:)-X2(1,:)).^2 + (XPF2(2,:)-X2(2,:)).^2   );    %λ�����
% PFdirectionerror1=XPF1(3,:)-X1(3,:);
% PFdirectionerror2=XPF2(3,:)-X2(3,:);

%��ͼ
hold on;box on;
title('�����˶�λ�켣');
plot(X1(1,:),X1(2,:),'-k.');  
plot(X11(1,:),X11(2,:),'-b+');
plot(XPF1(1,:),XPF1(2,:),'-rs');
plot(XPF2(1,:),XPF2(2,:),'-gs');
plot(X2(1,:),X2(2,:),'-k.');  
plot(X22(1,:),X22(2,:),'-b+');
legend('�����˶��켣', '��������λ�켣', 'PF�˲��켣 (robot 1)', 'PF�˲��켣 (robot 2)',  'Location', 'best');

figure
hold on;box on;
title('PF�˲�λ�����');
plot(PFpositionerror1,'-rs');
plot(PFpositionerror2,'-gs');
legend('һ�Ż������˶�λ�����','���Ż������˶�λ�����',  'Location', 'best');

