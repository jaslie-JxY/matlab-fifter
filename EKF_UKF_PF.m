%�������ϵͳЭͬ��λEKF��PF��UKF�˲��㷨
%״̬���̣�X(k+1)=FX(k)+G(U(k)+W(k))
%λ�ù۲ⷽ�̣�����
%��λ�۲ⷽ�̣��Ƕ�

tic % ��ʼ��ʱ

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
        ZD02=atan(detay/detax)+pi-XN2(3,i);                             %����״̬Ԥ��ֵ��Ƕȹ۲�ֵ
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

    %PF�˲�
    %�������Ӽ���
    n=100;                    %���Ӽ�������
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

    %�����˲�����
    for i=2:N
        %Ԥ�ⲽ,���Ӽ��ϸ���Ԥ��
        for j=1:n
            w1=normrnd(0,1,2,1);    
            w2=normrnd(0,1,2,1);                %���ӱ�׼��̬�ֲ���2��1�������
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
        sum1=sum(weight1,2);
        sum2=sum(weight2,2);      %��ÿһ�н�����Ͳ���
        for j=1:n
            weight1(j)=weight1(j)./sum1;
            weight2(j)=weight2(j)./sum2;     
        end

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
            a1=unifrnd(0,1);       %0��1���ȷֲ������   
            for m=1:n
                if(a1<c1(m))
                    XNEW1(:,j)=XOLD1(:,m);        %��������������ĸ��������Ӧ���ӽ��и���
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

    %UKF�˲�
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

    P1=eye(3);            
    P2=eye(3);              %Э�������ʼ��
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
    EKFpositionerror1=sqrt( (XEKF1(1,:)-X1(1,:)).^2 + (XEKF1(2,:)-X1(2,:)).^2   );
    UKFpositionerror1=sqrt( (XUKF1(1,:)-X1(1,:)).^2 + (XUKF1(2,:)-X1(2,:)).^2   );
    PFpositionerror1=sqrt( (XPF1(1,:)-X1(1,:)).^2 + (XPF1(2,:)-X1(2,:)).^2   );
    EKFpositionerror2=sqrt( (XEKF2(1,:)-X2(1,:)).^2 + (XEKF2(2,:)-X2(2,:)).^2   );  
    UKFpositionerror2=sqrt( (XUKF2(1,:)-X2(1,:)).^2 + (XUKF2(2,:)-X2(2,:)).^2   );
    PFpositionerror2=sqrt( (XPF2(1,:)-X2(1,:)).^2 + (XPF2(2,:)-X2(2,:)).^2   );    %λ�����
   



%��ͼ
figure
hold on;box on;
title('�������˶��켣', 'FontSize', 17);
plot(X1(1,:),X1(2,:),'-k.');             %ʵ�ߡ��ڡ���
plot(X11(1,:),X11(2,:),'-b+');           %��
plot(XEKF1(1,:),XEKF1(2,:),'-rs');
plot(XUKF1(1,:),XUKF1(2,:),'-gD');
plot(XPF1(1,:),XPF1(2,:),'-mh');
plot(X2(1,:),X2(2,:),'-k.');             %ʵ�ߡ��ڡ���
plot(X22(1,:),X22(2,:),'-b+');           %��
plot(XEKF2(1,:),XEKF2(2,:),'-rs');
plot(XUKF2(1,:),XUKF2(2,:),'-gD');
plot(XPF2(1,:),XPF2(2,:),'-mh');
legend('�����˶��켣','ʵ�ʹ۲�켣','EKF�˲��켣','UKF�˲��켣','PF�˲��켣', 'FontSize', 10, 'Location', 'best');

figure
hold on;box on;
title('һ�Ż������˶�λ�����', 'FontSize', 17);
plot(EKFpositionerror1,'-bs');
plot(UKFpositionerror1,'-rD');
plot(PFpositionerror1,'-mh');
legend('EKF�˲�λ�����','UKF�˲�λ�����','PF�˲�λ�����', 'FontSize', 10, 'Location', 'best');

figure
hold on;box on;
title('���Ż������˶�λ�����', 'FontSize', 17);
plot(EKFpositionerror2,'-bs');
plot(UKFpositionerror2,'-rD');
plot(PFpositionerror2,'-mh');
legend('EKF�˲�λ�����','UKF�˲�λ�����','PF�˲�λ�����', 'FontSize', 10, 'Location', 'best');

elapsed_time = toc; % ��������ʱ��
disp(['��������ʱ��Ϊ��', num2str(elapsed_time), '��']); % ��ʾ����ʱ��
