%多机器人系统协同定位PF滤波算法
%状态方程：X(k+1)=FX(k)+G(U(k)+W(k))
%位置观测方程：Z=两点距离公式+V(k)
%方位观测方程：观测角度

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

%状态方程
for t=2:N
    X1(:,t)=X1(:,t-1)+G1*U1;
    X2(:,t)=X2(:,t-1)+G2*U2;     %无误差下理想状态的运动迭代
end

for t=2:N  
    W1=sqrtm(Q)*randn(2,1);    
    W2=sqrtm(Q)*randn(2,1);    %产生2×1的符合标准正态分布的随机数,利用乘法改变方差，得到误差矩阵
    U1=U1+W1;
    U2=U2+W2;
    X11(:,t)=X1(:,t-1)+G1*U1;
    X22(:,t)=X2(:,t-1)+G2*U2;
    theta1=X11(3,t);
    theta2=X22(3,t);
    G1=[T*cos(theta1),0;T*sin(theta1),0;0,T];      
    G2=[T*cos(theta2),0;T*sin(theta2),0;0,T];
end

%位置观测方程，生成观测数据
for t=1:N
    Z00(:,t)=sqrt (       (          X2(2,t)-X1(2,t)           )^2   +    (X2(1,t)-X1(1,t)    )^2     )+sqrt(R1)*randn;  %量测值
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
U2=[v2;palstance];   %机器人运动状态初始化

%产生粒子集合
n=10000;                    %粒子集合总数
XOLD1=zeros(3,n);      
XNEW1=zeros(3,n);
XPF1=zeros(3,N);
weight1=zeros(1,n);  
ZZ=zeros(1,n);          %观测值预测矩阵
XOLD2=zeros(3,n);      
XNEW2=zeros(3,n);
XPF2=zeros(3,N);
weight2=zeros(1,n);        %权重
XPF1(:,1)=X1(:,1);
XPF2(:,1)=X2(:,1);

%设置初值，对自身给定初值足够自信
for i=1:n
    XOLD1(:,i)=X1(:,1);
    XOLD2(:,i)=X2(:,1);
    weight1(i)=1/n;
    weight2(i)=1/n;
end

%PF滤波
for i=2:N
    %预测步,粒子集合更新预测
    for j=1:n
        w1=normrnd(0,1,2,1);    
        w2=normrnd(0,1,2,1);     %服从标准正态分布的3×1随机变量
        XOLD1(:,j)=XOLD1(:,j)+G1*(U1+w1);
        XOLD2(:,j)=XOLD2(:,j)+G2*(U2+w2);   %粒子集合依此根据状态方程预测下一时刻粒子集合状态
    end     
    %更新步，粒子权重更新
    for j=1:n
        detay=XOLD2(2,j)-XOLD1(2,j);
        detax=XOLD2(1,j)-XOLD1(1,j);
        ZD0=sqrt(detay^2+detax^2);                      %根据状态预测值求距离观测值
        ZD01=atan(detay/detax)-XOLD1(3,j);
        ZD02=atan(detay/detax)+pi-XOLD2(3,j);
        dz1=[abs(Z00(:,i)-ZD0);abs(Z01(:,i)-ZD01)];         
        dz2=[abs(Z00(:,i)-ZD0);abs(Z02(:,i)-ZD02)];       %根据粒子集合预测观测值与实际真实观测值差值的绝对值分配权重
        weight1(j)=sqrt(2*pi)*normpdf(dz1(1),0,1)*(sqrt(2*pi)*normpdf(dz1(2),0,1));
        weight2(j)=sqrt(2*pi)*normpdf(dz2(1),0,1)*(sqrt(2*pi)*normpdf(dz2(2),0,1));         %权重更新
    end
    %权重归一化
    weight1=weight1/sum(weight1);
    weight2=weight2/sum(weight2);
    %重采样
    c1=zeros(1,n);
    c2=zeros(1,n);
    c1(1)=weight1(1);
    c2(1)=weight2(1);        
    for j=2:n
        c1(j)=c1(j-1)+weight1(j);
        c2(j)=c2(j-1)+weight2(j);
    end                                  %根据权重划分区间
    for j=1:n
        a1=unifrnd(0,1);       %均匀分布随机数       
        for k=1:n
            if(a1<c1(k))
                XNEW1(:,j)=XOLD1(:,k);        %根据随机数落在哪个区间对响应粒子进行复制
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
    %新的粒子复制给旧粒子集合
    XOLD1=XNEW1;
    XOLD2=XNEW2;
    for j=1:n
        weight1(j)=1/n;
        weight2(j)=1/n;               %权重重新设为1/n
    end
    XPF1(:,i)=sum(XNEW1,2)/n;
    XPF2(:,i)=sum(XNEW2,2)/n;         %利用粒子集合的均值作为滤波值
end

%误差分析
PFpositionerror1=sqrt( (XPF1(1,:)-X1(1,:)).^2 + (XPF1(2,:)-X1(2,:)).^2   );
PFpositionerror2=sqrt( (XPF2(1,:)-X2(1,:)).^2 + (XPF2(2,:)-X2(2,:)).^2   );    %位置误差
% PFdirectionerror1=XPF1(3,:)-X1(3,:);
% PFdirectionerror2=XPF2(3,:)-X2(3,:);

%画图
hold on;box on;
title('机器人定位轨迹');
plot(X1(1,:),X1(2,:),'-k.');  
plot(X11(1,:),X11(2,:),'-b+');
plot(XPF1(1,:),XPF1(2,:),'-rs');
plot(XPF2(1,:),XPF2(2,:),'-gs');
plot(X2(1,:),X2(2,:),'-k.');  
plot(X22(1,:),X22(2,:),'-b+');
legend('理想运动轨迹', '传感器定位轨迹', 'PF滤波轨迹 (robot 1)', 'PF滤波轨迹 (robot 2)',  'Location', 'best');

figure
hold on;box on;
title('PF滤波位置误差');
plot(PFpositionerror1,'-rs');
plot(PFpositionerror2,'-gs');
legend('一号机器人运动位置误差','二号机器人运动位置误差',  'Location', 'best');

