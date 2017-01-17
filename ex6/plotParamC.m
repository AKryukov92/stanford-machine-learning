function sigma = plotParamC(X,y)
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
  
  C_values = [0.01; 0.03; 0.1; 0.2; 0.3; 0.4; 0.5; 1; 3; 4; 5; 6];
  sigma = 3;
  
  error_train = zeros(length(C_values), 1);
  error_val = zeros(length(C_values), 1);
  
  for i=1:length(C_values)
    model = svmTrain(X_train, y_train, C_values(i), @(x1, x2) gaussianKernel(x1, x2, sigma));
    insights = svmPredict(model, X_train);
    predictions = svmPredict(model, X_val);
    error_train(i) = mean(double(insights ~= y_train));
    error_val(i) = mean(double(predictions ~= y_val));
  end
  
  disp(error_train)
  disp("----------")
  disp(error_val)
  newplot()
  plot(C_values, error_train, C_values, error_val);
  title('Learning curve for svm')
  legend('Train', 'Cross Validation')
  xlabel('Value of sigma')
  ylabel('Error')
end