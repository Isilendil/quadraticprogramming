clear all;


result_file = fopen('result', 'w');

fprintf(result_file, 'svmtrain_LS \t svmtrain_SMO \t svmtrain_QP \t svm_qp \t tsvm \n');

for iter = 1:45

%data_file = iter;
load(sprintf('data/%d', iter));


X = image_fea;
Y = image_gnd;
X = text_fea;
Y = text_gnd;

MaxX=max(X,[],2);
MinX=min(X,[],2);
DifX=MaxX-MinX;
idx_DifNonZero=(DifX~=0);
DifX_2=ones(size(DifX));
DifX_2(idx_DifNonZero,:)=DifX(idx_DifNonZero,:);
X = bsxfun(@minus, X, MinX);
X = bsxfun(@rdivide, X , DifX_2);


[n,d] = size(X);
n_train = round(n / 2);

X_train = X(1:n_train, :);
Y_train = Y(1:n_train);

X_test = X(n_train+1:end, :);
Y_test = Y(n_train+1:end);
n_test = size(X_test, 1);

w = svm_qp_train(X_train, Y_train, 1);
Y_hat = sign(X_test * w);
rate_svm_qp = sum(Y_hat~=Y_test) / length(Y_test);

%C = 1;
%[w, Y_hat] = tsvm_qp_train(X_train, Y_train, X_test, C, C/2, round(n_test/2));
%rate_tsvm_qp = sum(Y_hat~=Y_test) / length(Y_test);

L = n_train;
U = n_test;
C = 1;
C_star = L * C / U;
s = 0;
w = cccp_tsvm_train(X_train, Y_train, X_test, C, C_star, s);
Y_hat = sign(X_test * w);
rate_tsvm = sum(Y_hat~=Y_test) / length(Y_test);

option = statset('MaxIter', 50000);
svmStruct = svmtrain(X_train, Y_train, 'method', 'QP', 'options', option);
Y_hat = svmclassify(svmStruct, X_test);
rate_svmtrain_QP = sum(Y_hat~=Y_test) / length(Y_test);

svmStruct = svmtrain(X_train, Y_train, 'method', 'LS');
Y_hat = svmclassify(svmStruct, X_test);
rate_svmtrain_LS = sum(Y_hat~=Y_test) / length(Y_test);

svmStruct = svmtrain(X_train, Y_train);
Y_hat = svmclassify(svmStruct, X_test);
rate_svmtrain_SMO = sum(Y_hat~=Y_test) / length(Y_test);

fprintf(result_file, '%.4f \t %.4f \t %.4f \t %.4f \t %.4f \n', rate_svmtrain_LS, rate_svmtrain_SMO, rate_svmtrain_QP, rate_svm_qp, rate_tsvm);
end


fclose(result_file);

