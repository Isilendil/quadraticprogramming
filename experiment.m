clear all;

%load fisheriris 
%xdata = meas(51:end, 3:4);
%group = species(51:end);
%svmStruct = svmtrain(xdata, group, 'ShowPlot', true);

result_file = fopen('result', 'w');

fprintf(result_file, 'svmtrain_LS \t svmtrain_SMO \t svmtrain_QP \t svm_qp \n');

for iter = 1:45

%data_file = iter;
load(sprintf('data/%d', iter));


X = image_fea;
Y = image_gnd;

X = text_fea;
MaxX=max(X,[],2);
MinX=min(X,[],2);
DifX=MaxX-MinX;
idx_DifNonZero=(DifX~=0);
DifX_2=ones(size(DifX));
DifX_2(idx_DifNonZero,:)=DifX(idx_DifNonZero,:);
X = bsxfun(@minus, X, MinX);
X = bsxfun(@rdivide, X , DifX_2);


[n,d] = size(X);
num_train = round(3 * n / 4);

X_train = X(1:num_train, :);
Y_train = Y(1:num_train);

X_test = X(num_train+1:end, :);
Y_test = Y(num_train+1:end);

option = statset('MaxIter', 50000);
svmStruct = svmtrain(X_train, Y_train, 'method', 'QP', 'options', option);
Y_hat = svmclassify(svmStruct, X_test);
rate_svmtrain_QP = sum(Y_hat~=Y_test) / length(Y_test);

svmStruct = svmtrain(X_train, Y_train, 'method', 'LS');
Y_hat = svmclassify(svmStruct, X_test);
rate_svmtrain_LS = sum(Y_hat~=Y_test) / length(Y_test);

svmStruct = svmtrain(X_train, Y_train);
Y_hat = svmclassify(svmStruct, X_test);
rate_svmtrain_SMO = sum(Y_hat~=Y_test) / length(Y_test):

w = svm_qp_train(X_train, Y_train, 1);
Y_hat = sign(X_test * w(1:d));
rate_svm_qp = sum(Y_hat~=Y_test) / length(Y_test);

fprintf(result_file, '%.4f \t %.4f \t %.4f \t %.4f \n', rate_svmtrain_LS, rate_svmtrain_SMO, rate_svmtrain_QP, rate_svm_qp);
end


for iter = 1:45

%data_file = iter;
load(sprintf('data/%d', iter));


X = text_fea;
Y = text_gnd;

X = text_fea;
MaxX=max(X,[],2);
MinX=min(X,[],2);
DifX=MaxX-MinX;
idx_DifNonZero=(DifX~=0);
DifX_2=ones(size(DifX));
DifX_2(idx_DifNonZero,:)=DifX(idx_DifNonZero,:);
X = bsxfun(@minus, X, MinX);
X = bsxfun(@rdivide, X , DifX_2);


[n,d] = size(X);
num_train = round(3 * n / 4);

X_train = X(1:num_train, :);
Y_train = Y(1:num_train);

X_test = X(num_train+1:end, :);
Y_test = Y(num_train+1:end);

option = statset('MaxIter', 50000);
svmStruct = svmtrain(X_train, Y_train, 'method', 'QP', 'options', option);
Y_hat = svmclassify(svmStruct, X_test);
rate_svmtrain_QP = sum(Y_hat~=Y_test) / length(Y_test);

svmStruct = svmtrain(X_train, Y_train, 'method', 'LS');
Y_hat = svmclassify(svmStruct, X_test);
rate_svmtrain_LS = sum(Y_hat~=Y_test) / length(Y_test);

svmStruct = svmtrain(X_train, Y_train);
Y_hat = svmclassify(svmStruct, X_test);
rate_svmtrain_SMO = sum(Y_hat~=Y_test) / length(Y_test):

w = svm_qp_train(X_train, Y_train, 1);
Y_hat = sign(X_test * w(1:d));
rate_svm_qp = sum(Y_hat~=Y_test) / length(Y_test);

fprintf(result_file, '%.4f \t %.4f \t %.4f \t %.4f \n', rate_svmtrain_LS, rate_svmtrain_SMO, rate_svmtrain_QP, rate_svm_qp);
end


fclose(result_file);

