function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
X=[ones(m,1) X]; #Add ones to the X matrix to denote x0=1 for each eg
z2= Theta1*X'; #Second layer calculated values (25x401)x(401x5000) = 25x5000
a2= (sigmoid(z2))'; #Probabilities 5000 x 25
a2= [ones(size(a2,1),1) a2]; #Add bias node, 5000 x 26
z3= a2*Theta2'; #Third layer values (5000x26) x (26x10) = 5000 x 10
a3= sigmoid(z3);#Final Output Probabilites, 5000 x 10
[a3, p]=max(a3, [], 2); #Find the max probability class along each row
# p is a column vector

% =========================================================================


end
