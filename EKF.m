%�������ϵͳЭͬ��λEKF�˲��㷨
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

%EKF�˲�
XEKF1=zeros(3,N);     
XEKF2=zeros(3,N);     %EKF�˲����ݾ���
XEKF1(:,1)=X1(:,1);
XEKF2(:,1)=X2(:,1);   %�����ʼ��
P1=eye(3);           
P2=eye(3);            %Э�������ʼ��
XN1=zeros(3,N);
XN1(:,1)=X1(:,1);     
XN2=zeros(3,N);
XN2(:,1)=X2(:,1);     %״̬Ԥ��ֵ�۲����

%�����˲�����
for i=2:N
    XN1(:,i)=XEKF1(:,i-1)+G1*U1;      
    XN2(:,i)=XEKF2(:,i-1)+G2*U2;      %״̬һ��Ԥ��,����
    P11=P1+G1*Q*G1';                
    P22=P2+G2*Q*G2';                   %һ��Ԥ��Э������
    detay=XN2(2,i)-XN1(2,i);
    detax=XN2(1,i)-XN1(1,i);
    ZD0=sqrt(detay^2+detax^2);  %����״̬Ԥ��ֵ�����۲�ֵ
    ZD01=atan(detay/detax)-XN1(3,i);
    ZD02=atan(detay/detax)+pi-XN2(3,i);                                 %����״̬Ԥ��ֵ��Ƕȹ۲�ֵ
    ZD1=[ZD0;ZD01];
    ZD2=[ZD0;ZD02];
    H1=[-detax/ZD0,-detay/ZD0,0;detay/ZD0^2,-detax/ZD0^2,-1];                        
    H2=[detax/ZD0,detay/ZD0,0;-detay/ZD0^2,detax/ZD0^2,-1];          %���롢��λ�۲ⷽ��һ�����Ի������ſɱȾ���
    K1=P11*H1'/(H1*P11*H1'+R1);                    
    K2=P22*H2'/(H2*P22*H2'+R2);                    %���¿���������
    XEKF1(:,i)=XN1(:,i)+K1*(Z1(:,i)-ZD1);      
    XEKF2(:,i)=XN2(:,i)+K2*(Z2(:,i)-ZD2);           %����ϵͳ״̬%%%%%%%%%
    P1=P1*(eye(3)-K1*H1);                        
    P2=P2*(eye(3)-K2*H2);                         %����Э������
end

%������
EKFpositionerror1=sqrt( (XEKF1(1,:)-X1(1,:)).^2 + (XEKF1(2,:)-X1(2,:)).^2   );
EKFpositionerror2=sqrt( (XEKF2(1,:)-X2(1,:)).^2 + (XEKF2(2,:)-X2(2,:)).^2   );    %λ�����
% EKFdirectionerror1=XEKF1(3,:)-X2(3,:);
% EKFdirectionerror2=XEKF2(3,:)-X2(3,:);

%��ͼ
figure
hold on;box on;
title('�����˶�λ�켣');
plot(X1(1,:),X1(2,:),'-k.');  
plot(X11(1,:),X11(2,:),'-b+');
plot(XEKF1(1,:),XEKF1(2,:),'-rs');
plot(XEKF2(1,:),XEKF2(2,:),'-gs');
plot(X2(1,:),X2(2,:),'-k.');  
plot(X22(1,:),X22(2,:),'-b+');
legend('�����˶��켣', '��������λ�켣', 'EKF�˲��켣 (robot 1)', 'EKF�˲��켣 (robot 2)',  'Location', 'best');

figure
hold on;box on;
title('EKF�˲�λ�����');
plot(EKFpositionerror1,'-r+');
plot(EKFpositionerror2,'-gs');
legend('һ�Ż������˶�λ�����','���Ż������˶�λ�����','Location', 'best');

% figure
% hold on;box on;
% title('EKF�˲���λ���');
% plot(EKFdirectionerror1,'-r+');
% plot(EKFdirectionerror2,'-gs');
% legend('һ�Ż������˶���λ���','���Ż������˶���λ���');

% position_error = [EKFpositionerror1' EKFpositionerror2'];
% filename = 'position_error.txt';
% writematrix(position_error, filename, 'Delimiter', 'tab');
% position_error = [EKFpositionerror1' EKFpositionerror2'];
% filename = 'position_error.txt';
% writematrix(position_error, filename, 'Delimiter', 'tab');