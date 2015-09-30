function [w xi xi_star] = solve_tsvm(X_train, Y_train, X_test, Y_hat, C, C_negative_star, C_positive_star)

[n, d] = size(X_train);
n_test = size(X_test, 1);

H = eye(d + n + n_test);
H(d+1:end, d+1:end) = 0;

f = zeros(d + n + n_test, 1);
f(d+1 : d+n) = C;
C_hat = ones(n_test, 1) * C_negative_star;
C_hat(Y_hat==1) = C_positive_star;
f(d+n+1:end) = C_hat;

lb = zeros(d+n+n_test, 1) - inf;
lb(d+1:end) = 0;
ub = zeros(d+n+n_test, 1) + inf;

A1 = diag([Y_train', Y_hat']);
A2 = [[X_train; X_test], A1];
A = -1 * A1 * A2;

b = ones(n+n_test, 1) * -1;

result = quadprog(H, f, A, b, [], [], [], []);

w = result(1:d);
xi = result(d+1:d+n);
xi_star = result(d+n+1:end);

