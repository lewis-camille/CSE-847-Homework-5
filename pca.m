% CSE 847 Homework 5
% Principal Component Analysis
% Camille Lewis

load USPS.mat

% Show first two original images
A1 = reshape(A(1,:), 16, 16);
imwrite(A1', 'original_image_1.png')
A2 = reshape(A(2,:), 16, 16);
imwrite(A2', 'original_image_2.png')

p_candidates = [10, 50, 100, 200];
errors = zeros(1,4);

for i_p = 1:4
    % Compute PCs using PCA
    pcs = PCA(A, p_candidates(i_p));
    
    % Reconstruct data
    A_hat = (A * pcs) * transpose(pcs);
    
    % Compute total reconstruction error
    errors(i_p) = norm(A - A_hat, 'fro');
    
    % Show first two reconstructed images
    A1 = reshape(A_hat(1,:), 16, 16);
    imwrite(A1', append('reconstructed_image_1_p_', int2str(p_candidates(i_p)), '.png'))
    A2 = reshape(A_hat(2,:), 16, 16);
    imwrite(A2', append('reconstructed_image_2_p_', int2str(p_candidates(i_p)), '.png'))
end

% Output reconstruction errors
errors


function pcs = PCA(X, p)
% Function to perform PCA
% INPUTS
%    X - a data matrix
%    p - an integer representing the number of PCs to keep
% OUTPUT
%    pcs - a matrix whose columns are the resulting PCs
    dimensions = size(X);
    N = dimensions(1);
    % Center the data by subtracting the mean of the rows
    mean_of_rows = (1.0 / N) .* sum(X);
    for i = 1:N
        X(i,:) = X(i,:) - mean_of_rows;
    end
    
    % Compute SVD of centered matrix, X = USV'
    [U, S, V] = svd(X);
    
    % Return first p columns of V
    pcs = V(:,1:p);
    
end