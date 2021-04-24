% CSE 847 Homework 5
% Clustering
% Camille Lewis

% Create random dataset
rng(0, 'twister'); % random seed for repeatability
num_points = 100;
num_features = 2;
% data is a d-by-N matrix whose values fall
% in the range (low_bound,high_bound)
low_bound = 0;
high_bound = 50;
data = (high_bound - low_bound).*rand(num_features, num_points) + low_bound;

% Plot raw data
figure
scatter(data(1,:), data(2,:), 'filled')
xlabel('x1 feature value')
ylabel('x2 feature value')
title('Randomly Generated Data Points')

% Perform k-means and spectral relaxed k-means on data
% across different k values (k = 2, 3, 4, 5, 6)
k_candidates = [2, 3, 4, 5, 6];
for i_k = 1:5
    % *** k-means **************
    % Perform k-means clustering using this value of k
    [centers, members] = k_means(data, k_candidates(i_k), 1000);
    
    % Plot data colored by cluster
    figure
    xlabel('x1 feature value')
    ylabel('x2 feature value')
    title(append('K-Means: Data Separated into ', int2str(k_candidates(i_k)), ' Clusters'))
    colors = ['r', 'g', 'b', 'm', 'c', 'y'];
    
    % Iterate over memberships
    hold on
    for i_point = 1:num_points
        % Determine which cluster this point is in
        for i_cluster = 1:k_candidates(i_k)
            if members(i_point, i_cluster) == 1
                scatter(data(1,i_point), data(2, i_point), colors(i_cluster), 'filled');
                continue
            end
        end
    end
    hold off
    
    % *** spectral relaxation **************
    % Perform spectral relaxation of k-means using this value of k
    members = spectral_relaxation(data, k_candidates(i_k));
    
    % Plot data colored by cluster
    figure
    xlabel('x1 feature value')
    ylabel('x2 feature value')
    title(append('Spectral Relaxation: Data Separated into ', int2str(k_candidates(i_k)), ' Clusters'))
    colors = ['r', 'g', 'b', 'm', 'c', 'y'];
    
    % Iterate over memberships
    hold on
    for i_point = 1:num_points
        % Determine which cluster this point is in
        for i_cluster = 1:k_candidates(i_k)
            if members(i_point, i_cluster) == 1
                scatter(data(1,i_point), data(2, i_point), colors(i_cluster), 'filled');
                continue
            end
        end
    end
    hold off
    
end


function [C, M] = k_means(X, k, maxiter)
% Function to perform k-means clustering
% INPUTS
%    X - a data matrix
%    k - an integer representing the number of clusters
%    maxiter - an integer - the maximum number of iterations of the
%              algorithm
% OUTPUT
%    C - a matrix whose columns are the centers of the clusters
%    M - a matrix whose entries M_ij are 1 if data point i belongs to
%        cluster j and 0 otherwise
dimensions = size(X);
d = dimensions(1);
N = dimensions(2);
M = zeros(N, k);
prev_C = 1;
% Initialize C- randomly choose k points from X
% by choosing random number in [1, N] k times
C = zeros(d, k);
for i= 1:k
    num = randi([1, N]);
    C(:, i) = X(:, num);
end

% while centroids are changing
for iter = 1:maxiter
    prev_C = C;
    % Fix C and compute M
    for i = 1:N
        min_j = 1;
        min_dist = norm(X(:,i) - C(:,1))^2;
        for j = 2:k
            this_dist = norm(X(:,i) - C(:,j))^2;
            if this_dist < min_dist
                min_dist = this_dist;
                min_j = j;
            end
        end
        M(i,:) = zeros(1, k);
        M(i, min_j) = 1;
    end
            
    % Fix M and compute C
    for j = 1:k
        numerator = 0.0;
        denominator = 0.0;
        for i =1:N
            numerator = numerator + M(i, j) .* X(:,i);
            denominator = denominator + M(i, j);
        end
        % if denominator is zero, set c_j to be a randomly chosen data
        % point from X
        if denominator == 0
            num = randi([1, N]);
            C(:, j) = X(:, num);
        else
            C(:, j) = (1.0 / denominator) .* numerator;
        end
    end
    if sum(C - prev_C, 'all') == 0
        break
    end
end
end

function M = spectral_relaxation(X, k)
% Function to perform k-means clustering
% INPUTS
%    X - a data matrix
%    k - an integer representing the number of clusters
% OUTPUT
%    M - a matrix whose entries M_ij are 1 if data point i belongs to
%        cluster j and 0 otherwise
dimensions = size(X);
N = dimensions(2);

% Compute eigendecomposition
[V, D] = eig(transpose(X) * X);

% Sort vals and vectors in descending order
[d, indices] = sort(diag(D), 'descend');
D = D(indices, indices);
V = V(:,indices);

% Keep eigenvectors corresponding to k largest eigenvalues
Y = V(:, 1:k);

% Perform QR decomposition to find permutation matrix P s.t.
% Y'P = QR
[Q, R, P] = qr(transpose(Y));
R11 = R(:, 1:k);
R12 = R(:, k+1:N);
R_hat = horzcat(eye(k), R11 \ R12) * transpose(P);

% Cluster membership of a data vector is the row index of the largest
% element(in absolute value) of the corresponding column of R_hat
M = zeros(N, k);
% Iterate over columns of R_hat
for i = 1:N
    max_j = 1;
    max_val = abs(R_hat(1,i));
    for j = 2:k
        this_val = abs(R_hat(j,i));
        if this_val > max_val
            max_j = j;
            max_val = this_val;
        end
    end
    M(i, max_j) = 1;
end
end
