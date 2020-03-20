function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

#Simple Sigmoid function. Remember to use . operator to facilitate elementwise
# Operations so that, this function works with matrix, vector or scalar
g=1./(1.+exp(-z));

% =============================================================

end
