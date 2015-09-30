function [w xi xi_star] = solve_cccp_tsvm(X_train, Y_train, X_new, Y_new, C, C_star, beta)

[L, d] = size(X_train);
U = size(X_new, 1) / 2;

H = eye(d + L + 2*U);
H(d+1:end, d+1:end) = 0;

f = zeros(d + L + 2*U, 1);
f(d+1 : d+L) = C;
f(d+L+1:end) = C_star;
f(1:d) = sum(X_new' * (beta.*Y_new), 2);

lb = zeros(d+L+2*U, 1) - inf;
lb(d+1:end) = 0;

ub = zeros(d+L+2*U, 1) + inf;

A1 = diag([Y_train', Y_new']);
A2 = [[X_train; X_new], A1];
A = -1 * A1 * A2;

b = ones(L+2*U, 1) * -1;

Aeq = sum(X_new(1:U,:), 1) / U;
Aeq = [Aeq, zeros(1,L+2*U)];

beq = sum(Y_train) / L;

result = quadprog(H, f, A, b, Aeq, beq, lb, []);

w = result(1:d);
xi = result(d+1:d+L);
xi_star = result(d+L+1:end);

