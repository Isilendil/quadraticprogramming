function w = svm_qp_train_bias(X, Y, C)

X = [ones(size(X,1),1), X];
[n, d] = size(X);

H = eye(d+n);
H(d+1:end, d+1:end) = 0;

f = zeros(d+n, 1);
f(d+1:end) = C;

lb = zeros(d+n, 1);
lb(1:d) = -inf;

ub = zeros(d+n, 1) + inf;

A1 = diag(Y);
A2 = [X, diag(Y)];
A = -1 * A1 * A2;

b = ones(n, 1) * -1;

w = quadprog(H, f, A, b, [], [], lb, []);

