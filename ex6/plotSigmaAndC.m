function sigma = plotSigmaAndC(X,y)
disp(size(X))
disp(size(y))

train_end = floor(size(X,1)*0.6);
val_end = floor(size(X,1)*0.2) + train_end;
test_end = size(X,1);

X_train = X(1:train_end,      1:size(X,2));
X_val   = X(train_end:val_end,1:size(X,2));
X_test  = X(val_end:test_end, 1:size(X,2));

y_train = y(1:train_end,      1:size(y,2));
y_val   = y(train_end:val_end,1:size(y,2));
y_test  = y(val_end:test_end, 1:size(y,2));

C = 3;

C_values = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigmas = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

error_train = zeros(length(sigmas), 1);
error_val = zeros(length(sigmas), 1);

for i=1:length(C_values)
  for j = 1:length(sigmas)
    model = svmTrain(X_train, y_train, C_values(i), @(x1, x2) gaussianKernel(x1, x2, sigmas(j)));
    insights = svmPredict(model, X_train);
    predictions = svmPredict(model, X_val);
    error_train(i,j) = mean(double(insights ~= y_train));
    error_val(i,j) = mean(double(predictions ~= y_val));
  end
end

[min_arr, min_i] = min(error_val)
[min_val, min_j] = min(min_arr)
%disp(error_val(min_i(min_j),min_j))
disp(C_values(min_i(min_j)))
disp(sigmas(min_j))

disp(error_train)
disp("----------")
disp(error_val)
newplot()
mesh(sigmas, C_values, error_train);
mesh(sigmas, C_values, error_val);
title('Learning curve for svm')
legend('Train', 'Cross Validation')
xlabel('Value of sigma')
ylabel('Error')
end