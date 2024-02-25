%�������ϵͳЭͬ��λUKF�˲��㷨
%״̬���̣�X(k+1)=FX(k)+G(U(k)+W(k))
%λ�ù۲ⷽ�̣�����
%��λ�۲ⷽ�̣��Ƕ�

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

n=3;       %Xά��
weight=zeros(1,2*n+1);  %Ȩ��
lamda=3-n;     %һ��ȡ3-n
alpha=1;
kalpha=0;
belta=2;     %UT�任���ϵ��

Wm=zeros(1,2*n+1);
Wc=zeros(1,2*n+1);

for j=1:2*n+1
    Wm(j)=1/(2*(n+lamda));                    %��ֵ
    Wc(j)=1/(2*(n+lamda));                    %Э����
end
Wm(1)=lamda/(n+lamda);
Wc(1)=lamda/(n+lamda)+1-alpha^2+belta;        %Ȩֵ��ֵ

XUKF1=zeros(3,N);
XUKF1(:,1)=X1(:,1);
XUKF2=zeros(3,N);
XUKF2(:,1)=X2(:,1);           %UKF�˲�ֵ�洢

%UKF�˲�
for i=2:N
    xsigma1=zeros(n,2*n+1);      
    xsigma1(:,1)=XUKF1(:,i-1);
    xsigma2=zeros(n,2*n+1);       %sigma�㼯
    xsigma2(:,1)=XUKF2(:,i-1);
    L1=chol(P1*(n+lamda));
    L2=chol(P2*(n+lamda));         %����ֽ�
    for j=1:n
        xsigma1(:,j+1)=xsigma1(:,1)+L1(:,j);
        xsigma1(:,j+1+n)=xsigma1(:,1)-L1(:,j);
        xsigma2(:,j+1)=xsigma2(:,1)+L2(:,j);
        xsigma2(:,j+1+n)=xsigma2(:,1)-L2(:,j);
    end
    %�㼯һ��Ԥ��
    xsigmaminus1=zeros(n,2*n+1); 
    xsigmaminus2=zeros(n,2*n+1);
    for j=1:2*n+1
        xsigmaminus1(:,j)=xsigma1(:,j)+G1*U1;
        xsigmaminus2(:,j)=xsigma2(:,j)+G2*U2;
    end
    %���ֵ��Э�������
    x1hat=zeros(n,1);
    p1=zeros(n,n);
    x2hat=zeros(n,1);
    p2=zeros(n,n);
    for j=1:2*n+1
        x1hat=x1hat+Wm(j)*xsigmaminus1(:,j);
        x2hat=x2hat+Wm(j)*xsigmaminus2(:,j);     %��Ȩ��ֵ
    end
    for j=1:2*n+1
        p1=p1+Wc(j)*(xsigmaminus1(:,j)-x1hat)*(xsigmaminus1(:,j)-x1hat)';
        p2=p2+Wc(j)*(xsigmaminus2(:,j)-x2hat)*(xsigmaminus2(:,j)-x2hat)';     %Э�������
    end
    p1=p1+G1*Q*G1'; 
    p2=p2+G2*Q*G2';     %Ԥ�ⲽ����
    %���²�������һ��Ԥ��ֵ���ٴ�ʹ��UT�任�������µ�sigma�㼯
    xsigma1(:,1)=x1hat;   
    L1=chol(p1*(n+lamda)); 
    xsigma2(:,1)=x2hat;   
    L2=chol(p2*(n+lamda));%�����µľ�ֵ��Э������
    for j=1:n
        xsigma1(:,j+1)=xsigma1(:,1)+L1(:,j);
        xsigma1(:,j+1+n)=xsigma1(:,1)-L1(:,j);
        xsigma2(:,j+1)=xsigma2(:,1)+L2(:,j);
        xsigma2(:,j+1+n)=xsigma2(:,1)-L2(:,j);
    end
    %����Ԥ��۲���
    z1=zeros(2,2*n+1);
    z2=zeros(2,2*n+1);
    z1hat=zeros(2,1);
    z2hat=zeros(2,1);
    for j=1:2*n+1
        detay=xsigma2(2,j)-xsigma1(2,j);
        detax=xsigma2(1,j)-xsigma1(1,j);
        z1(1,j)=sqrt(detay^2+detax^2);            %����
        z1(2,j)=atan(detay/detax)-xsigma1(3,j);         %��λ  
        z2(1,j)=sqrt(detay^2+detax^2);
        z2(2,j)=atan(detay/detax)+pi-xsigma2(3,j);
        z1hat=z1hat+Wm(j)*z1(:,j);
        z2hat=z2hat+Wm(j)*z2(:,j);            %��Ȩ��ֵ
    end
    %Э����
    Pz1=zeros(2,2);
    Pz2=zeros(2,2);
    Pxz1=zeros(3,2);
    Pxz2=zeros(3,2);
    for j=1:2*n+1
        Pz1=Pz1+Wc(j)*(z1(:,j)-z1hat)*(z1(:,j)-z1hat)';
        Pz2=Pz2+Wc(j)*(z2(:,j)-z2hat)*(z2(:,j)-z2hat)';
        Pxz1=Pxz1+Wc(j)*(xsigma1(:,j)-x1hat)*(z1(:,j)-z1hat)';
        Pxz2=Pxz2+Wc(j)*(xsigma2(:,j)-x2hat)*(z2(:,j)-z2hat)';
    end
    Pz1=Pz1+[R1,0;0,R2];
    Pz2=Pz2+[R1,0;0,R2];
    %����������
    K1=Pxz1/Pz1;
    K2=Pxz2/Pz2;
    %���²�
    XUKF1(:,i)=x1hat+K1*(Z1(:,i)-z1hat);
    XUKF2(:,i)=x2hat+K2*(Z2(:,i)-z2hat);   %״ֵ̬����
    P1=p1-K1*Pz1*K1';
    P2=p2-K2*Pz2*K2';           %Э�������
end

%������
UKFpositionerror1=sqrt( (XUKF1(1,:)-X1(1,:)).^2 + (XUKF1(2,:)-X1(2,:)).^2   );
UKFpositionerror2=sqrt( (XUKF2(1,:)-X2(1,:)).^2 + (XUKF2(2,:)-X2(2,:)).^2   );

%��ͼ
figure
hold on;box on;
title('�����˶�λ�켣');
plot(X1(1,:),X1(2,:),'-k.');  
plot(X11(1,:),X11(2,:),'-b+');
plot(XUKF1(1,:),XUKF1(2,:),'-rs');
plot(XUKF2(1,:),XUKF2(2,:),'-gs');
plot(X2(1,:),X2(2,:),'-k.');  
plot(X22(1,:),X22(2,:),'-b+');
legend('�����˶��켣', '��������λ�켣', 'UKF�˲��켣 (robot 1)', 'UKF�˲��켣 (robot 2)', 'Location', 'best');

figure
hold on;box on;
title('UKF�˲�λ�����');
plot(UKFpositionerror1,'-r+');
plot(UKFpositionerror2,'-gs');
legend('һ�Ż������˶�λ�����','���Ż������˶�λ�����',  'Location', 'best');