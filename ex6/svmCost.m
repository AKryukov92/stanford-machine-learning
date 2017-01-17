function [J, grad] = svmCost(X, y, theta, C)

m = length(y);

J = 1/m*C*sum() + 1/2*sum(theta.^2)

end