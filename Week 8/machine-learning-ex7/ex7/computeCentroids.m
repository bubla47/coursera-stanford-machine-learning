function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%
for i=1:K
  #Compute a binary vector returning 1s for examples assigned to centroid i 
  # 0 otherwise
  cen = (idx == i);
  # X(cen,:) Returns only the values which are assigned to centroid i
  #Finally, calculate the mean and update the centroid
  # X(cen,:) returns the sum of coordinates, on dividing with sum(cen) 
  # we get mean of the coordinates, i.e, new centroid
  
  #To avoid division by zero which happens when a centroid is assigned no point
  if sum(cen) ~= 0
    centroids(i,:) = sum(X(cen,:))/sum(cen); #update the centroid
  endif
  ##NOTE: I earlier thought that we could also directly use mean(X(idx==i,:))
  # But that was causing a problem of non-conformant arguments in case of the
  # image pixels application which I could not solve and so I changed it to this 
endfor
% =============================================================


end

