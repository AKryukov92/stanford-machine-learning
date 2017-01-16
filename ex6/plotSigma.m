function sigma = plotSigma(X,y)
  disp(size(X))
  disp(size(y))
  
  train_end = floor(size(X,1)*0.6);
  val_end = floor(size(X,1)*0.2) + train_end;
  test_end = size(X,1);
  X_train = X(1:train_end,1:size(X,2));
  X_val = X(train_end:val_end,1:size(X,2));
  X_test = X(val_end:test_end,1:size(X,2));
  
  
end