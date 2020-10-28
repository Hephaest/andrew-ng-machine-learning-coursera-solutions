function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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
% Using C examples in ex5
C_vec     = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
sigma_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
C_len     = length(C_vec);
sigma_len = length(sigma_vec);
% Initial errors
pred_error = zeros(C_len, sigma_len);

% Using double loop
for i = 1 : C_len
    for j = 1 : sigma_len
        C_test = C_vec(i); 
        sigma_test = sigma_vec(j);
        SMV_train = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
        pred = svmPredict(SMV_train, Xval);
        pred_error(i, j) = mean(double(pred ~= yval));
    end
end

% Find the minimum
[minVals, row] = min(pred_error);
[~, col]       = min(minVals);
index          = row(col);

C     = C_vec(index);
sigma = sigma_vec(col);
end