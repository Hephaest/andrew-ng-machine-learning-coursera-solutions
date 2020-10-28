function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

hx = X * theta;
theta_cut = theta(2 : end);
cost_sum = (1 / (2 * m)) * sum((hx - y) .^2);
J  =  cost_sum  + (lambda / (2 * m)) * sum(theta_cut .^ 2);

x_0 = X(:, 1);
x_rest = X(:, 2 : end);
grad(1) = (1 / m) * (x_0' * (hx - y));
grad(2 : end) = (1 / m) * (x_rest' * (hx - y)) + (lambda / m) * theta_cut;



% =========================================================================

grad = grad(:);

end
