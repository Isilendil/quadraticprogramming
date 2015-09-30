function [w, Y_hat] = tsvm_qp_train(X_train, Y_train, X_test, C, C_star, num_positive)

%X = [ones(size(X,1),1), X];
[n, d] = size(X_train);

H = eye(d+n);
H(d+1:end, d+1:end) = 0;

f = zeros(d+n, 1);
f(d+1:end) = C;

lb = zeros(d+n, 1);
lb(1:d) = -inf;

ub = zeros(d+n, 1) + inf;

A1 = diag(Y_train);
A2 = [X_train, diag(Y_train)];
A = -1 * A1 * A2;

b = ones(n, 1) * -1;

result = quadprog(H, f, A, b, [], [], lb, []);

xi = result(d+1:end);
w = result(1:d);

[n_test, d_test] = size(X_test);
margin_hat = X_test * w;
[value, index] = sort(margin_hat, 'descend');
Y_hat = -1 * ones(n_test, 1);
Y_hat(index(1:num_positive)) = 1;

C_negative_star = 10^(-5);
C_positive_star = C_negative_star * (num_positive / (n_test - num_positive));

while (C_negative_star < C_star) | (C_positive_star < C_star)

  [w xi xi_star] = solve_tsvm(X_train, Y_train, X_test, Y_hat, C, C_negative_star, C_positive_star);

  margin_hat = X_test * w;
  [value, index] = sort(margin_hat, 'descend');
  Y_hat = -1 * ones(n_test, 1);
  Y_hat(index(1:num_positive)) = 1;

  while true

    Y_hat_product = tril(Y_hat * Y_hat');
    xi_star_sum = repmat(xi_star, 1, n_test) + repmat(xi_star', n_test, 1);

		[r c] = find( (Y_hat_product < 0) & (xi_star_sum > 2) );

    if length(r) == 0
		  break;	
		end

		for i = 1 : length(r)
			m = r(i);
			l = c(i);
			if (xi_star(m) > 0) & (xi_star(l) > 0)
				Y_hat(m) = -Y_hat(m);
				Y_hat(l) = -Y_hat(l);
				solve_tsvm(X_train, Y_train, X_test, Y_hat, C, C_negative_star, C_positive_star);
			  break; % end for
			end
		end
	end

	C_negative_star = min(C_negative_star*2, C_star);
	C_positive_star = min(C_positive_star*2, C_star);

end





