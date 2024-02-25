%多机器人系统协同定位EKF滤波算法
%状态方程：X(k+1)=FX(k)+G(U(k)+W(k))
%位置观测方程：距离
%方位观测方程：角度

clc;
clear;
close all;

%初始化        
T=1;                 %采样周期
N=100;                %采样次数

X1=zeros(3,N);
X2=zeros(3,N);      %3×N矩阵，记录机器人状态变化
X11=zeros(3,N);
X22=zeros(3,N);     %记录加入误差后的机器人状态变化

theta1=36.87*pi/180;
theta2=53.13*pi/180;
v1=1;
v2=2;
palstance=0;           %机器人位姿状态初始化

X1(:,1)=[1,1,theta1];    
X2(:,1)=[1,2,theta2];    
X11(:,1)=[1,1,theta1];
X22(:,1)=[1,2,theta2];    %矩阵第一列初始化机器人位姿状态,x,y位置，运动方向

G1=[T*cos(theta1),0;T*sin(theta1),0;0,T];      
G2=[T*cos(theta2),0;T*sin(theta2),0;0,T];   %3×2过程噪声驱动矩阵

U1=[v1;palstance];
U2=[v2;palstance];   %机器人运动状态初始化

Q=diag([1,1]);       %过程噪声方差（对角阵）对角线元素分别为速度误差方程和角速度误差方差
R1=1;                 %距离观测方差
R2=1;                 %角度观测方差

Z00=zeros(1,N);
Z01=zeros(1,N);
Z02=zeros(1,N);            %观测数据存储矩阵

%状态方程
for t=2:N
    X1(:,t)=X1(:,t-1)+G1*U1;
    X2(:,t)=X2(:,t-1)+G2*U2;     %无误差下理想状态的运动迭代
end

for t=2:N  
    W1=sqrtm(Q)*randn(2,1);    
    W2=sqrtm(Q)*randn(2,1);      %产生2×1的符合标准正态分布的随机数,利用乘法改变方差，得到过程噪声
    U1=U1+W1;
    U2=U2+W2;
    X11(:,t)=X1(:,t-1)+G1*U1;
    X22(:,t)=X2(:,t-1)+G2*U2;
    theta1=X11(3,t);
    theta2=X22(3,t);
    G1=[T*cos(theta1),0;T*sin(theta1),0;0,T];      
    G2=[T*cos(theta2),0;T*sin(theta2),0;0,T];
end

%观测方程，量测值包括距离以及角度
for t=1:N
    Z00(:,t)=sqrt (       (          X2(2,t)-X1(2,t)           )^2   +    (X2(1,t)-X1(1,t)    )^2     )+sqrt(R1)*randn;  %量测值
    Z01(:,t)=atan(  (X2(2,t)-X1(2,t))/(X2(1,t)-X1(1,t)) -X1(3,t))+sqrt(R2)*randn;         
    Z02(:,t)=atan(  (X2(2,t)-X1(2,t))/(X2(1,t)-X1(1,t)) +pi-X2(3,t)  )+sqrt(R2)*randn;
end
Z1=[Z00;Z01];
Z2=[Z00;Z02];

%机器人运动状态初始化
theta1=36.87*pi/180;
theta2=53.13*pi/180;
G1=[T*cos(theta1),0;T*sin(theta1),0;0,T];      
G2=[T*cos(theta2),0;T*sin(theta2),0;0,T];
U1=[v1;palstance];
U2=[v2;palstance];   

%EKF滤波
XEKF1=zeros(3,N);     
XEKF2=zeros(3,N);     %EKF滤波数据矩阵
XEKF1(:,1)=X1(:,1);
XEKF2(:,1)=X2(:,1);   %矩阵初始化
P1=eye(3);           
P2=eye(3);            %协方差阵初始化
XN1=zeros(3,N);
XN1(:,1)=X1(:,1);     
XN2=zeros(3,N);
XN2(:,1)=X2(:,1);     %状态预测值观测矩阵

%核心滤波代码
for i=2:N
    XN1(:,i)=XEKF1(:,i-1)+G1*U1;      
    XN2(:,i)=XEKF2(:,i-1)+G2*U2;      %状态一步预测,递推
    P11=P1+G1*Q*G1';                
    P22=P2+G2*Q*G2';                   %一步预测协方差阵
    detay=XN2(2,i)-XN1(2,i);
    detax=XN2(1,i)-XN1(1,i);
    ZD0=sqrt(detay^2+detax^2);  %根据状态预测值求距离观测值
    ZD01=atan(detay/detax)-XN1(3,i);
    ZD02=atan(detay/detax)+pi-XN2(3,i);                                 %根据状态预测值求角度观测值
    ZD1=[ZD0;ZD01];
    ZD2=[ZD0;ZD02];
    H1=[-detax/ZD0,-detay/ZD0,0;detay/ZD0^2,-detax/ZD0^2,-1];                        
    H2=[detax/ZD0,detay/ZD0,0;-detay/ZD0^2,detax/ZD0^2,-1];          %距离、方位观测方程一阶线性化，求雅可比矩阵
    K1=P11*H1'/(H1*P11*H1'+R1);                    
    K2=P22*H2'/(H2*P22*H2'+R2);                    %更新卡尔曼增益
    XEKF1(:,i)=XN1(:,i)+K1*(Z1(:,i)-ZD1);      
    XEKF2(:,i)=XN2(:,i)+K2*(Z2(:,i)-ZD2);           %更新系统状态%%%%%%%%%
    P1=P1*(eye(3)-K1*H1);                        
    P2=P2*(eye(3)-K2*H2);                         %更新协方差阵
end

%误差分析
EKFpositionerror1=sqrt( (XEKF1(1,:)-X1(1,:)).^2 + (XEKF1(2,:)-X1(2,:)).^2   );
EKFpositionerror2=sqrt( (XEKF2(1,:)-X2(1,:)).^2 + (XEKF2(2,:)-X2(2,:)).^2   );    %位置误差
% EKFdirectionerror1=XEKF1(3,:)-X2(3,:);
% EKFdirectionerror2=XEKF2(3,:)-X2(3,:);

%画图
figure
hold on;box on;
title('机器人定位轨迹');
plot(X1(1,:),X1(2,:),'-k.');  
plot(X11(1,:),X11(2,:),'-b+');
plot(XEKF1(1,:),XEKF1(2,:),'-rs');
plot(XEKF2(1,:),XEKF2(2,:),'-gs');
plot(X2(1,:),X2(2,:),'-k.');  
plot(X22(1,:),X22(2,:),'-b+');
legend('理想运动轨迹', '传感器定位轨迹', 'EKF滤波轨迹 (robot 1)', 'EKF滤波轨迹 (robot 2)',  'Location', 'best');

figure
hold on;box on;
title('EKF滤波位置误差');
plot(EKFpositionerror1,'-r+');
plot(EKFpositionerror2,'-gs');
legend('一号机器人运动位置误差','二号机器人运动位置误差','Location', 'best');

% figure
% hold on;box on;
% title('EKF滤波方位误差');
% plot(EKFdirectionerror1,'-r+');
% plot(EKFdirectionerror2,'-gs');
% legend('一号机器人运动方位误差','二号机器人运动方位误差');

% position_error = [EKFpositionerror1' EKFpositionerror2'];
% filename = 'position_error.txt';
% writematrix(position_error, filename, 'Delimiter', 'tab');
% position_error = [EKFpositionerror1' EKFpositionerror2'];
% filename = 'position_error.txt';
% writematrix(position_error, filename, 'Delimiter', 'tab');