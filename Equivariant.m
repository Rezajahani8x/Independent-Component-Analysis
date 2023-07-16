                                %% HW9
                                %% Section 2
file = load('hw9.mat');
A = file.A;
Noise = file.Noise;
S = file.S;
[M,N] = size(A);
[~,T] = size(S);

X = A*S + Noise;
X_denoised = X - Noise;

            %% Part 2.1
num_iter = 100;
miu = 1e-1;
[B,Y,f_B] = ICA_Equivariant(X,num_iter,miu,M,N,T);
C = A*B;
C_temp = zeros(N,N);
for i=1:N
   c_max = max(abs(C(i,:)));
   C_temp(i,i) = c_max;
end
C = 1e6*C_temp/5;
disp("AB=");
disp(C);

            %% Part 2.2
Es1 = 1/T*sum(S(1,:).^2);
Es2 = 1/T*sum(S(2,:).^2);
Es3 = 1/T*sum(S(3,:).^2);
Ey1 = 1/T*sum(Y(1,:).^2);
Ey2 = 1/T*sum(Y(2,:).^2);
Ey3 = 1/T*sum(Y(3,:).^2);
S_hat = zeros(N,T);
S_hat(1,:) = max(S(1,:))/max(Y(1,:)) * Y(1,:);
S_hat(2,:) = max(S(2,:))/max(Y(2,:)) * Y(2,:);
S_hat(3,:) = max(S(3,:))/max(Y(3,:)) * Y(3,:);
E = norm(S_hat-S,'fro')^2 / norm(S,'fro')^2;
figure(1);
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

            %% Part 2.3
figure(2);
plot(1:num_iter,f_B);
xlabel('Iteration Number');
ylabel('f(B)');
title('Cost Function in each Iteration');
grid on;


            %% Local Necessary Functions
function [B,Y,f_B] = ICA_Equivariant(X,num_iter,miu,M,N,T)
    B = rand(N,M);
    Y = B * X;
    f_B = zeros(num_iter,1);
    for n=1:num_iter
       psi = zeros(N,T);
       for m=1:M
           ym = Y(m,:);
           psi(m,:) = psi_calc(T,ym,ym);
       end
       nat_grad = 1/T*psi * transpose(Y);
       for i=1:M
          yi = Y(i,:);
          nat_grad(i,i) = 1 - mean(yi.^2);
       end
       B = B - miu*nat_grad*B;
       Y = B * X;
       f_B(n) = cost_func(Y);
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

function f_B = cost_func(Y)
    [M,~] = size(Y);
    H = entropy(Y);
    Hs = 0;
    for i=1:M
       Hs = Hs + entropy(Y(i,:)); 
    end
    f_B = -H + Hs;
end
