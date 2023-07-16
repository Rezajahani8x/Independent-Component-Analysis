                                        %% HW8
file = load('hw8.mat');
A = file.A;
Noise = file.Noise;
S = file.S;
[M,N] = size(A);
[~,T] = size(S);

                %% Forming the matrix and plotting the signals
X = A*S + Noise;
X_denoised = X - Noise;

figure(1);
subplot(3,1,1);
plot(1:T,S(1,:));
xlim([1,T]);
xlabel('t');
title('s_1(t)');
grid on;
subplot(3,1,2);
plot(1:T,S(2,:));
xlim([1,T]);
xlabel('t');
title('s_2(t)');
grid on;
subplot(3,1,3);
plot(1:T,S(3,:));
xlim([1,T]);
xlabel('t');
title('s_3(t)');
grid on;
figure(2);
subplot(3,1,1);
plot(1:T,X_denoised(1,:));
xlim([1,T]);
xlabel('t');
title('x_1(t) No noise');
grid on;
subplot(3,1,2);
plot(1:T,X_denoised(2,:));
xlim([1,T]);
xlabel('t');
title('x_2(t) No noise');
grid on;
subplot(3,1,3);
plot(1:T,X_denoised(3,:));
xlim([1,T]);
xlabel('t');
title('x_3(t) No noise');
grid on;
figure(3);
subplot(3,1,1);
plot(1:T,X(1,:));
xlim([1,T]);
xlabel('t');
title('x_1(t)');
grid on;
subplot(3,1,2);
plot(1:T,X(2,:));
xlim([1,T]);
xlabel('t');
title('x_2(t)');
grid on;
subplot(3,1,3);
plot(1:T,X(3,:));
xlim([1,T]);
xlabel('t');
title('x_3(t)');
grid on;

                %% Part 1
num_iter = 100;
miu = 1e-1;
[B,Y,f_B] = ICA_1(X,num_iter,miu,M,N,T);
C = A*B;
C_temp = zeros(N,N);
for i=1:N
   c_max = max(abs(C(i,:)));
   C_temp(i,i) = c_max;
end
C = C_temp;
disp("AB=");
disp(C);

                %% Part 2
Es1 = 1/T*sum(S(1,:).^2);
Es2 = 1/T*sum(S(2,:).^2);
Es3 = 1/T*sum(S(3,:).^2);
B_alt = inv(A) + 0.3*randn(N,M);
Y_alt = B_alt*X;
S_hat = Y_alt;
E = norm(S_hat-S,'fro')^2 / norm(S,'fro')^2;
figure(4);
subplot(3,1,1);
plot(1:T,S(1,:),'blue'); hold on;
plot(1:T,S_hat(1,:),'red');
xlim([1,T]);
xlabel('t');
title('s_1(t) and y_1(t)');
grid on;
legend('s_1(t)','y_1(t)');
subplot(3,1,2);
plot(1:T,S(2,:),'blue'); hold on;
plot(1:T,S_hat(2,:),'red');
xlim([1,T]);
xlabel('t');
title('s_2(t) and y_2(t)');
grid on;
legend('s_2(t)','y_2(t)');
subplot(3,1,3);
plot(1:T,S(3,:),'blue'); hold on;
plot(1:T,S_hat(3,:),'red');
xlim([1,T]);
xlabel('t');
title('s_3(t) and y_3(t)');
grid on;
legend('s_3(t)','y_3(t)');
disp("Error=");
disp(E);

                %% Part 3
figure(5);
plot(1:num_iter,f_B);
xlabel('Iteration Number');
ylabel('f(B)');
title('Cost Function in each Iteration');
grid on;

                
            %% Local Necessary Functions
function [B,Y,f_B] = ICA_1(X,num_iter,miu,M,N,T)
    B = rand(N,M);
    Y = B * X;
    f_B = zeros(num_iter,1);
    for i = 1:num_iter
       psi = zeros(M,T);
       for m=1:M
          ym = Y(m,:);
          psi(m,:) = psi_calc(T,ym,ym);
       end
       df = 1/T*psi*transpose(X) - inv(transpose(B));
       B = B - miu*df;
       B = normr(B);
       Y = B*X;
       f_B(i) = cost_func(Y,df);
    end
end

function psi = psi_calc(T,ym,y)
    K = [ones(1,T);y;y.^2;y.^3;y.^4;y.^5];
    K_y = [ones(1,T);ym;ym.^2;ym.^3;ym.^4;ym.^5];
%     E_K_y_diff = [0;1*T;sum(2*ym);sum(3*ym.^2);sum(4*ym.^3);sum(5*ym.^4)];
    E_K_y_diff = [0;1;mean(2*ym);mean(3*ym.^2);mean(4*ym.^3);mean(5*ym.^4)];
    theta = inv(1/T*K_y * transpose(K_y)) * E_K_y_diff;
    psi = transpose(theta) * K;
end

function f_B = cost_func(Y,df)
    [M,~] = size(Y);
    H = entropy(Y);
    Hs = 0;
    for i=1:M
       Hs = Hs + entropy(Y(i,:)); 
    end
%     f_B = -H + Hs;
    f_B = norm(df);
end