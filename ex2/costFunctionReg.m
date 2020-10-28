function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
g_z = X * theta;
h_x = sigmoid(g_z);
% Escape x_0
reg_cost = (lambda / m) * theta(2:end);
reg_cost_sum = (lambda / (2 * m)) * sum(theta(2:end).^2);
deta_sum = sum((-y .* log(h_x)) - ((1 - y) .* log(1 - h_x)));
% Return the cost and gradient
J = (1 / m) * deta_sum + reg_cost_sum;
% Return the gradient, we only have 2 thetas
% theta_0 is not for penalty wherea MATLAB starts at 1 not 0
grad(1) = (1/m) * (X(:, 1))' * (h_x - y);
grad(2: end) = (1/m) * (X(:, 2: end))' * (h_x - y) + reg_cost;



% =============================================================

end
