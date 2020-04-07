function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
for i=1:length(X)
  idx(i) = 1; #Initialise idx(i) with the first centroid
   #Initialise the min distance with the distance between centroid 1 and X(i)
  min = sum((X(i,:) - centroids(1,:)).^2);
  for j=2:K #Loop through each remaining centroid
    #Calculate distance of X(i) from centroid j
    dist = sum((X(i,:) - centroids(j,:)).^2);
    #Check if it is less than already found min distance
    if dist < min
      min = dist; #Update min to store currently found minimum distance
      idx(i) = j; #Update idx to current centroid
    endif
  endfor
endfor

% =============================================================

end

