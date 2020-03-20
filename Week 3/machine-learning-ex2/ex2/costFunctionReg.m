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
p=sigmoid(X*theta);#Calculate the Prediction Vector
x=-sum((y.*log(p)).+((1.-y).*log(1.-p)));#Non Regularised Cost
J=(1/m)*(x+(0.5*lambda*sum(theta(2:end).^2)));#Add the Regularisation term

grad=(1/m)*X'*(p-y);#Calculate the Gradient Vector without Regularisation
temp=theta;
temp(1)=0;#We do not regularise theta0 so set theta(1) to 0
grad = grad + (lambda*(1/m)).*temp#Add the regularisation part to the gradients

% =============================================================

end
