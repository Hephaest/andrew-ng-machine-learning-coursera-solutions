function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
theta_1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

theta_2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X = [ones(m, 1), X];
% You need to return the following variables correctly 
J = 0;
theta_1_grad = zeros(size(theta_1));
theta_2_grad = zeros(size(theta_2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

a_1 = X; z_2 = a_1 * theta_1'; gz_2 = sigmoid(z_2);
% Add 1 column ahead of the a_2
a_2 = [ones(size(gz_2, 1), 1), gz_2];

% Calculate hx
z_3 = a_2 * theta_2'; hx = sigmoid(z_3);

% Conversion without looping
vec_y = (1 : num_labels) == y;

% ===================================
% Part-1
% ===================================

% Cost function
cost_sum = sum((-vec_y .* log(hx)) - ((1 - vec_y) .* log(1 - hx)));
J = (1 / m) * sum(cost_sum);

% ===================================
% Part-2
% ===================================
delta_3 = hx - vec_y;
expand_z2 = ones(size(z_2, 1), 1);
delta_temp = (delta_3 * theta_2) .* [expand_z2 sigmoidGradient(z_2)];
delta_2 = delta_temp(:, 2: end);

theta_1_grad = (1 / m) * (delta_2' * a_1);
theta_2_grad = (1 / m) * (delta_3' * a_2);

% ===================================
% Part-3
% ===================================
theta_1_cut = theta_1(:, 2 : end);
theta_1_square_sum = sum(sum(theta_1_cut .^ 2));
theta_2_cut = theta_2(:, 2 : end);
theta_2_square_sum = sum(sum(theta_2_cut .^ 2));

reg_sum = (lambda / (2 * m)) * (theta_1_square_sum + theta_2_square_sum);
J = J + reg_sum;

theta_1_grad_temp = (lambda / m) * [zeros(size(theta_1, 1), 1) theta_1_cut];
theta_2_grad_temp = (lambda / m) * [zeros(size(theta_2, 1), 1) theta_2_cut];

theta_1_grad = theta_1_grad_temp + theta_1_grad;
theta_2_grad = theta_2_grad_temp + theta_2_grad;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [theta_1_grad(:) ; theta_2_grad(:)];


end