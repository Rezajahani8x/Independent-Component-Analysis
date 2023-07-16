                                %% HW10
                                %% Section 2
file = load('hw10.mat');
A = file.A;
Noise = file.Noise;
S = file.S;
[M,N] = size(A);
[~,T] = size(S);

                        %% Forming the matrix
X = A*S + Noise;
X_denoised = X - Noise;

                        %% Part 2.1
num_iter = 1000;
miu = 1e-2;
[B,Y,f_B] = ICA_Kurt_Max_FP(X,num_iter,M);
C = A*B;
C_temp = zeros(N,N);
for i=1:N
   c_max = max(abs(C(i,:)));
   C_temp(i,i) = c_max;
end
C = C_temp;
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
S_hat(1,:) = Es1/Ey1 * Y(1,:);
S_hat(2,:) = -Es2/Ey2 * Y(2,:);
S_hat(3,:) = -Es3/Ey3 * Y(3,:);
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

                        %% Part 2.3
figure(5);
plot(1:num_iter,f_B);
xlabel('Iteration Number');
ylabel('f(B)');
title('Cost Function in each Iteration');
grid on;

        %% Local Necessary Functions
        
function KURT = kurt(y)
    KURT = mean(y.^4) - 3*(mean(y.^2))^2;
end
function bm = constraint_projection(m,M,bm,B)
    if m==1
       bm = bm/norm(bm); 
    end
    RM = zeros(M,m-1);
    for i=1:m-1
       bi = B(i,:);
       RM(:,i) = transpose(bi);
    end
    if m~=1
       bm = (eye(M) - RM*transpose(RM))*bm;
       bm = bm/norm(bm);
    end
end

function [B_sep,Y,f_B] = ICA_Kurt_Max_FP(X,num_iter,M)
    Rx = X * transpose(X);
    [U,D] = eig(Rx);
    d = diag(D);
    [d, ind] = sort(d,'descend');
    U = U(:, ind);
    D = diag(d);
    W = D^(-1/2) * transpose(U);
    Z = W * X;

    B = rand(M,M);
    Y = B * Z;
    f_B = zeros(num_iter,1);
    for i=1:num_iter
        f_B_temp = zeros(M,1);
        for m=1:M
           bm = B(m,:);
           bm = transpose(bm);
           ym = Y(m,:);
           f_B_temp(m) = abs(kurt(ym));
           b_new_total = Z.*((transpose(bm) * Z).^3) - 3*bm;
           b_new = mean(b_new_total,2);
           bm = b_new;
           bm = constraint_projection(m,M,bm,B);
           B(m,:) = transpose(bm);
        end
        f_B(i) = mean(f_B_temp);
        Y = B * Z;
    end
    B_sep = W * B;
end