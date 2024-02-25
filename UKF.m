%多机器人系统协同定位UKF滤波算法
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
P1=eye(3);            
P2=eye(3);              %协方差阵初始化

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

n=3;       %X维数
weight=zeros(1,2*n+1);  %权重
lamda=3-n;     %一般取3-n
alpha=1;
kalpha=0;
belta=2;     %UT变换相关系数

Wm=zeros(1,2*n+1);
Wc=zeros(1,2*n+1);

for j=1:2*n+1
    Wm(j)=1/(2*(n+lamda));                    %均值
    Wc(j)=1/(2*(n+lamda));                    %协方差
end
Wm(1)=lamda/(n+lamda);
Wc(1)=lamda/(n+lamda)+1-alpha^2+belta;        %权值赋值

XUKF1=zeros(3,N);
XUKF1(:,1)=X1(:,1);
XUKF2=zeros(3,N);
XUKF2(:,1)=X2(:,1);           %UKF滤波值存储

%UKF滤波
for i=2:N
    xsigma1=zeros(n,2*n+1);      
    xsigma1(:,1)=XUKF1(:,i-1);
    xsigma2=zeros(n,2*n+1);       %sigma点集
    xsigma2(:,1)=XUKF2(:,i-1);
    L1=chol(P1*(n+lamda));
    L2=chol(P2*(n+lamda));         %矩阵分解
    for j=1:n
        xsigma1(:,j+1)=xsigma1(:,1)+L1(:,j);
        xsigma1(:,j+1+n)=xsigma1(:,1)-L1(:,j);
        xsigma2(:,j+1)=xsigma2(:,1)+L2(:,j);
        xsigma2(:,j+1+n)=xsigma2(:,1)-L2(:,j);
    end
    %点集一步预测
    xsigmaminus1=zeros(n,2*n+1); 
    xsigmaminus2=zeros(n,2*n+1);
    for j=1:2*n+1
        xsigmaminus1(:,j)=xsigma1(:,j)+G1*U1;
        xsigmaminus2(:,j)=xsigma2(:,j)+G2*U2;
    end
    %求均值、协方差矩阵
    x1hat=zeros(n,1);
    p1=zeros(n,n);
    x2hat=zeros(n,1);
    p2=zeros(n,n);
    for j=1:2*n+1
        x1hat=x1hat+Wm(j)*xsigmaminus1(:,j);
        x2hat=x2hat+Wm(j)*xsigmaminus2(:,j);     %加权均值
    end
    for j=1:2*n+1
        p1=p1+Wc(j)*(xsigmaminus1(:,j)-x1hat)*(xsigmaminus1(:,j)-x1hat)';
        p2=p2+Wc(j)*(xsigmaminus2(:,j)-x2hat)*(xsigmaminus2(:,j)-x2hat)';     %协方差矩阵
    end
    p1=p1+G1*Q*G1'; 
    p2=p2+G2*Q*G2';     %预测步结束
    %更新步：根据一步预测值，再次使用UT变换，产生新的sigma点集
    xsigma1(:,1)=x1hat;   
    L1=chol(p1*(n+lamda)); 
    xsigma2(:,1)=x2hat;   
    L2=chol(p2*(n+lamda));%利用新的均值和协方差阵
    for j=1:n
        xsigma1(:,j+1)=xsigma1(:,1)+L1(:,j);
        xsigma1(:,j+1+n)=xsigma1(:,1)-L1(:,j);
        xsigma2(:,j+1)=xsigma2(:,1)+L2(:,j);
        xsigma2(:,j+1+n)=xsigma2(:,1)-L2(:,j);
    end
    %生成预测观测量
    z1=zeros(2,2*n+1);
    z2=zeros(2,2*n+1);
    z1hat=zeros(2,1);
    z2hat=zeros(2,1);
    for j=1:2*n+1
        detay=xsigma2(2,j)-xsigma1(2,j);
        detax=xsigma2(1,j)-xsigma1(1,j);
        z1(1,j)=sqrt(detay^2+detax^2);            %距离
        z1(2,j)=atan(detay/detax)-xsigma1(3,j);         %方位  
        z2(1,j)=sqrt(detay^2+detax^2);
        z2(2,j)=atan(detay/detax)+pi-xsigma2(3,j);
        z1hat=z1hat+Wm(j)*z1(:,j);
        z2hat=z2hat+Wm(j)*z2(:,j);            %加权均值
    end
    %协方差
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
    %卡尔曼增益
    K1=Pxz1/Pz1;
    K2=Pxz2/Pz2;
    %更新步
    XUKF1(:,i)=x1hat+K1*(Z1(:,i)-z1hat);
    XUKF2(:,i)=x2hat+K2*(Z2(:,i)-z2hat);   %状态值更新
    P1=p1-K1*Pz1*K1';
    P2=p2-K2*Pz2*K2';           %协方差更新
end

%误差分析
UKFpositionerror1=sqrt( (XUKF1(1,:)-X1(1,:)).^2 + (XUKF1(2,:)-X1(2,:)).^2   );
UKFpositionerror2=sqrt( (XUKF2(1,:)-X2(1,:)).^2 + (XUKF2(2,:)-X2(2,:)).^2   );

%画图
figure
hold on;box on;
title('机器人定位轨迹');
plot(X1(1,:),X1(2,:),'-k.');  
plot(X11(1,:),X11(2,:),'-b+');
plot(XUKF1(1,:),XUKF1(2,:),'-rs');
plot(XUKF2(1,:),XUKF2(2,:),'-gs');
plot(X2(1,:),X2(2,:),'-k.');  
plot(X22(1,:),X22(2,:),'-b+');
legend('理想运动轨迹', '传感器定位轨迹', 'UKF滤波轨迹 (robot 1)', 'UKF滤波轨迹 (robot 2)', 'Location', 'best');

figure
hold on;box on;
title('UKF滤波位置误差');
plot(UKFpositionerror1,'-r+');
plot(UKFpositionerror2,'-gs');
legend('一号机器人运动位置误差','二号机器人运动位置误差',  'Location', 'best');