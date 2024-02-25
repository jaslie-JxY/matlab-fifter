%多机器人系统协同定位EKF、PF、UKF滤波算法
%状态方程：X(k+1)=FX(k)+G(U(k)+W(k))
%位置观测方程：距离
%方位观测方程：角度

tic % 开始计时

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
        ZD02=atan(detay/detax)+pi-XN2(3,i);                             %根据状态预测值求角度观测值
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

    %PF滤波
    %产生粒子集合
    n=100;                    %粒子集合总数
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

    %核心滤波代码
    for i=2:N
        %预测步,粒子集合更新预测
        for j=1:n
            w1=normrnd(0,1,2,1);    
            w2=normrnd(0,1,2,1);                %服从标准正态分布的2×1随机变量
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
        sum1=sum(weight1,2);
        sum2=sum(weight2,2);      %对每一行进行求和操作
        for j=1:n
            weight1(j)=weight1(j)./sum1;
            weight2(j)=weight2(j)./sum2;     
        end

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
            a1=unifrnd(0,1);       %0到1均匀分布随机数   
            for m=1:n
                if(a1<c1(m))
                    XNEW1(:,j)=XOLD1(:,m);        %根据随机数落在哪个区间对响应粒子进行复制
                    break;
                end
            end
        end
        for j=1:n
            a2=unifrnd(0,1);             
            for m=1:n
                if(a2<c2(m))
                    XNEW2(:,j)=XOLD2(:,m);        
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

    %UKF滤波
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

    P1=eye(3);            
    P2=eye(3);              %协方差阵初始化
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
    EKFpositionerror1=sqrt( (XEKF1(1,:)-X1(1,:)).^2 + (XEKF1(2,:)-X1(2,:)).^2   );
    UKFpositionerror1=sqrt( (XUKF1(1,:)-X1(1,:)).^2 + (XUKF1(2,:)-X1(2,:)).^2   );
    PFpositionerror1=sqrt( (XPF1(1,:)-X1(1,:)).^2 + (XPF1(2,:)-X1(2,:)).^2   );
    EKFpositionerror2=sqrt( (XEKF2(1,:)-X2(1,:)).^2 + (XEKF2(2,:)-X2(2,:)).^2   );  
    UKFpositionerror2=sqrt( (XUKF2(1,:)-X2(1,:)).^2 + (XUKF2(2,:)-X2(2,:)).^2   );
    PFpositionerror2=sqrt( (XPF2(1,:)-X2(1,:)).^2 + (XPF2(2,:)-X2(2,:)).^2   );    %位置误差
   



%画图
figure
hold on;box on;
title('机器人运动轨迹', 'FontSize', 17);
plot(X1(1,:),X1(2,:),'-k.');             %实线、黑、点
plot(X11(1,:),X11(2,:),'-b+');           %蓝
plot(XEKF1(1,:),XEKF1(2,:),'-rs');
plot(XUKF1(1,:),XUKF1(2,:),'-gD');
plot(XPF1(1,:),XPF1(2,:),'-mh');
plot(X2(1,:),X2(2,:),'-k.');             %实线、黑、点
plot(X22(1,:),X22(2,:),'-b+');           %蓝
plot(XEKF2(1,:),XEKF2(2,:),'-rs');
plot(XUKF2(1,:),XUKF2(2,:),'-gD');
plot(XPF2(1,:),XPF2(2,:),'-mh');
legend('理想运动轨迹','实际观测轨迹','EKF滤波轨迹','UKF滤波轨迹','PF滤波轨迹', 'FontSize', 10, 'Location', 'best');

figure
hold on;box on;
title('一号机器人运动位置误差', 'FontSize', 17);
plot(EKFpositionerror1,'-bs');
plot(UKFpositionerror1,'-rD');
plot(PFpositionerror1,'-mh');
legend('EKF滤波位置误差','UKF滤波位置误差','PF滤波位置误差', 'FontSize', 10, 'Location', 'best');

figure
hold on;box on;
title('二号机器人运动位置误差', 'FontSize', 17);
plot(EKFpositionerror2,'-bs');
plot(UKFpositionerror2,'-rD');
plot(PFpositionerror2,'-mh');
legend('EKF滤波位置误差','UKF滤波位置误差','PF滤波位置误差', 'FontSize', 10, 'Location', 'best');

elapsed_time = toc; % 计算运行时间
disp(['程序运行时间为：', num2str(elapsed_time), '秒']); % 显示运行时间
