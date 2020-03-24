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
p=X*theta;% Prediction Vector
J=sum((p-y).^2)+ lambda*sum(theta(2:end).^2); %Regularized sum of errors
J=1/(2*m)*J; %Regularized Cost


Theta= theta;%Copy of Theta vector
Theta(1)=0; % Making theta0, i.e, Theta(1)=0, because it is not penalized
grad=(1/m).*(sum(X'*(p-y),2)+ lambda*Theta);#Vectorized calculation of gradients


% =========================================================================

grad = grad(:);

end
