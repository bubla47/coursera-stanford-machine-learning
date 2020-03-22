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
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
#Feed Forward
X=[ones(m,1) X];%5000x401
z2= Theta1*X';%(25x401)x(401x5000)=25x5000
a2= (sigmoid(z2))';%5000x25
a2= [ones(size(a2,1),1) a2];%5000x26
z3= a2*Theta2';%(5000x26)x(26x10)=5000x10
a3= sigmoid(z3);%5000x10

#Cost Calculation
for k=1:num_labels
  y_curr= (y==k); #One vs All algorithm
  J=J+sum((y_curr.*log(a3(:,k)))+((1.-y_curr).*log(1.-a3(:,k)))); 
endfor
J=-(1/m)*J;

#Adding Regularisation
theta1=Theta1(:,2:end); #all hidden units, except bias
theta2=Theta2(:,2:end); #all hidden units, except bias
reg1=sum(sum(theta1.^2)); #regularisation portion of Theta1
reg2=sum(sum(theta2.^2)); #regularisation portion of Theta2
reg=reg1+reg2; #Total regularisation
J=J+((1/(2*m))*lambda*reg); #Final cost

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
#VERY IMPORTANT, REFER TO THE GIVEN FORMULA IN THE LECTURES.
#HELPS to implement the algorithm by keeping track of the shape of 
# matrices and vectors.
for i=1:m
  a1=X(i,:);%1x401
  z2=Theta1*a1';%(25x401)x(401x1)=25x1
  a2=(sigmoid(z2))';%1x25
  a2=[ones(size(a2,1),1) a2];%1x26
  z3=Theta2*a2';%(10x26)x(26x1)=10x1
  a3=sigmoid(z3);%10x1
  yi=([1:size(a3,1)]'==y(i));%10x1
  del3=a3-yi;%10x1
  del2=Theta2'*del3.*sigmoidGradient([1; z2]);%(26x10)x(10x1)=26x1
  Theta1_grad=Theta1_grad+(del2(2:end)*a1);%(25x401)+(25x1)*(1x401)=25x401
  Theta2_grad=Theta2_grad+(del3*a2);%(10x26)+(10x1)*(1*26)
endfor
Theta1_grad=(1/m)*Theta1_grad;#Calculate Average
Theta2_grad=(1/m)*Theta2_grad;#Calculate Average
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
theta1=[zeros(size(theta1,1),1) theta1]; #Set the bias unit to 0
theta2=[zeros(size(theta2,1),1) theta2]; #Set the bias unit to 0
Theta1_grad=Theta1_grad+(lambda/m)*theta1; #Regularised Gradients of Theta1
Theta2_grad=Theta2_grad+(lambda/m)*theta2; #Regularised Gradients of Theta2

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
