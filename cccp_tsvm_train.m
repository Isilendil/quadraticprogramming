function w = cccp_tsvm_train(X_train, Y_train, X_test, C, C_star, s)

[L, d] = size(X_train);
U = size(X_test, 1);

X_new = [X_test; X_test];
Y_new = [ones(U, 1); ones(U, 1) * -1];

% svm on training data
[w, xi] = svm_qp_train(X_train, Y_train, C);

margin_hat = X_new * w;
beta_update = C_star * ((Y_new.*margin_hat)<s);
%beta_update = [zeros(L,1); beta_update];

beta = -inf;

while (norm(beta_update-beta)) >= eps

  beta = beta_update;

  [w, xi, xi_star] = solve_cccp_tsvm(X_train, Y_train, X_new, Y_new, C, C_star, beta);

  margin_hat = X_new * w;
  beta_update = C_star * ((Y_new.*margin_hat)<s);
  %beta_update = [zeros(n,1); beta_update];

end


