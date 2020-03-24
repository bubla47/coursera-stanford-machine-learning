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
c1 = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]; #array of values for C
sig = [0.03, 0.1, 0.3, 1, 3, 10, 30]; #array of values for sigma
#Train the SVM with C = 0.01, sigma = 0.01
model= svmTrain(X, y, 0.01, @(x1, x2) gaussianKernel(x1, x2, 0.01)); 
pred = svmPredict(model, Xval); #Predict the y values with C=0.01, sigma=0.01
error = mean(double(pred ~= yval));#Calculate error with C=0.01, sigma=0.01
C = 0.01 #Update C
sigma = 0.01 #Update sigma

#Now we generate all 63 remaining combinations of C and Sigma 
#And Compare their errors. If the error is less than a previously
#Tested Combination, we update C, sigma and error

# NOTE: It is important to calculate the error of the first combination outside
# The Loops as it is used as the base value for comparing the other combinations
# If this is skipped we may make the mistake of comparing the errors with 0
# Which of course is the least possible error and hence will result in an error

for i=1:length(c1)
  for j=1:length(sig)
    #Train SVM with C = c1(i) and sigma = sig(j)
    model= svmTrain(X, y, c1(i), @(x1, x2) gaussianKernel(x1, x2, sig(j)));
    #Predict the y values with C=c1(i), sigma=sig(j) 
    pred = svmPredict(model, Xval); 
    #Calculate error with C=c1(i), sigma=sig(j)
    error1 = mean(double(pred ~= yval));
    if error1 < error
      #update Error, C, sigma
      error = error1;
      C = c1(i);
      sigma = sig(j);
    endif
  endfor
endfor

#The Best possible values of C and sigma are chosen by this point

% =========================================================================

end
