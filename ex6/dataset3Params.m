function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.

C_values = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigmas = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

error_train = zeros(length(C_values), 1);
error_val = zeros(length(C_values), 1);

for i=1:length(C_values)
  for j = 1:length(sigmas)
    model = svmTrain(X, y, C_values(i), @(x1, x2) gaussianKernel(x1, x2, sigmas(j)));
    insights = svmPredict(model, X);
    predictions = svmPredict(model, Xval);
    error_train(i,j) = mean(double(insights ~= y));
    error_val(i,j) = mean(double(predictions ~= yval));
  end
end

[min_arr, min_i] = min(error_val);
[min_val, min_j] = min(min_arr);

C = C_values(min_i(min_j));
sigma = sigmas(min_j);
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
