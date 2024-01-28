import torch
import numpy as np   
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import itertools


# 1. Data preprocessing: Start by preprocessing your data to obtain a similarity matrix or distance matrix. If you have a distance metric, you can convert it into a similarity matrix by taking the inverse of the distances.
# 2. Construct the graph: Create an undirected weighted graph G = (V, E) based on the similarity matrix. Each data point corresponds to a vertex in the graph, and the edges between vertices are weighted by the similarities or dissimilarities between the corresponding points.
# NOTE: weights in the case of crypto transfers are represent the normalized total transfer amount between two addresses 

df = pd.read_csv("./network_table.csv")
network_cols = df[['source_id', 'dest_id', 'wt']]
source = df.source_id
destination = df.dest_id
data = df.wt
vertices = np.unique(np.concatenate([df['source_id'].values, df['dest_id'].values]))

rows = source.max() + 1
cols = destination.max() + 1
dim = max(rows, cols) # gonna make a square matrix

# 3. Compute the Laplacian matrix: There are several types of Laplacians that can be used for spectral clustering, such as the symmetric normalized Laplacian (Lsym) and the unnormalized Laplacian (Lun). For Lsym, compute the degree matrix D by summing the weights of each vertex's connections (wij), and then form Lsym = D - W, where W is the weighted adjacency matrix.

ind = torch.tensor(np.array([source,destination]))

# A_sparse is "W" in the above equation; the weighted adjacency matrix
A_sparse = torch.sparse_coo_tensor(indices = ind, values = torch.tensor(data), size=[dim,dim])

A_torch = A_sparse.to_dense()

# HACK: this is the fix for a singular matrix error: where nodes receive but do not send, the matrix will become singular. So, to fix this, we add a small constant to any all-zero rows, excluding the diagonal, which makes the matrix non-singular, and without influencing the results. 
zero_rows = torch.all(A_torch == 0, dim=1)
# Define the small constant
small_constant = 1e-9

# Add the small constant to all zero rows, excluding the diagonal
for i, is_zero_row in enumerate(zero_rows):
    if is_zero_row and i != A_torch.size(1) - 1:  # Exclude the diagonal
        A_torch[i] += small_constant

# Convert the sparse adjacency matrix to a dense matrix and move it to GPU
A_dense = A_torch.double().cuda()  # Move the tensor to GPU memory

# Create the degree matrix D
D = torch.diag(torch.sum(A_dense, dim=1)).cuda()  # Move the tensors to GPU memory before computation

L = D - A_dense
# print(f"Laplacian: {L}")

# 4. Compute the eigendecomposition: Calculate the eigenvalues and eigenvectors of the Laplacian matrix. The smallest non-zero eigenvalue (λ2) and its corresponding eigenvector (u2) will be used in the next step.

eigenvalues, eigenvectors = torch.linalg.eig(L)
eigenvalues = eigenvalues.real  # get the real part of the eigenvalues ONLY
# print(f"eigenvalues: {eigenvalues}")
# print(f"eigenvectors: {eigenvectors}")


# Get the index of the second smallest eigenvalue (smallest non-zero)
second_smallest_index = torch.argsort(eigenvalues)[1]
print(f"second_smallest_index: {second_smallest_index}")

# Get the second smallest eigenvalue and its corresponding eigenvector
lambda2 = eigenvalues[second_smallest_index]
print(f"lambda2: {lambda2}")
u2 = eigenvectors[:, second_smallest_index]
print(f"u2: {u2}")

old_cluster_assignments = np.zeros(u2.shape[0])
max_iterations = 300  # maximum number of iterations (to avoid infinite loops; it's not really used, though)

# 7. Iterate and Check Convergence (Repeat steps 5-6): The process might require multiple iterations (known as the Girvan-Newman algorithm) to refine the cluster assignment and optimize the quality of the clusters. In each iteration, identify edges with the smallest eigenvalues, remove them from the graph, and repeat steps 3-6 on the updated graph until convergence or a satisfactory clustering result is achieved.
def check_convergence(old_cluster_assignments, new_cluster_assignments):
    return np.array_equal(old_cluster_assignments, new_cluster_assignments)


# 5. Form the affinity matrix: Construct an affinity matrix A using the second smallest eigenvalue and eigenvector as follows:
# a_ij = exp(-λ2 * ||u_i - u_j||^2 / 2 * sum(sum(||u_i - u_j||^2)))
# where a_ij is the affinity between data points i and j, and ||.|| denotes the Euclidean distance between two points in the feature space.
n = u2.shape[0]  # number of data points
A = torch.zeros((n, n))  # initialize the affinity matrix
for i in range(n):
    for j in range(n):
        # calculate the Euclidean distance between u_i and u_j
        distance = torch.norm(u2[i] - u2[j])
        # calculate the affinity
        A[i, j] = torch.exp(-lambda2 * distance**2 / (2 * torch.sum(distance**2 + 1e-15))) # 1e-15 is a small constant to prevent division by zero
        # ISSUE: the diagonal of the affinity matrix should not be nan. The diagonal of the affinity matrix represents the affinity of each data point with itself, which should be the maximum possible value, not nan.
        # The nan values are likely due to a division by zero in the calculation of the affinity. This can happen when i equals j, because the Euclidean distance between u_i and u_j is zero, leading to a division by zero in the denominator of the affinity calculation.
        # SOLUTION: To avoid this, you can add a small constant to the denominator to prevent division by zero

    # 6. Cluster assignment: Assign each data point to its nearest cluster center based on the affinity matrix. You can use various clustering algorithms like k-means or hierarchical clustering for this step. Alternatively, you can set a threshold for the affinity values and assign data points to the same cluster if their affinity is above the threshold.
    # Number of clusters

### GRID SEARCH ###
# Here, we will use k-means clustering to assign data points to clusters. The number of clusters (k) is a hyperparameter that you can tune to get the best clustering results. You can use grid search to find the best value of k. To do this, we iterate over a range of values for k and also the "n_init" parameter, and compute the silhouette score for each value of k. The silhouette score is a metric that measures the quality of the clusters. It is a value between -1 and 1, where a value closer to 1 indicates better clustering results. We use the silhouette score to find the best value of k, the return its details.

# Pick the number of clusters you want to iterate over (to try different values of k)
k_array = np.linspace(5, 40, 36).round().astype(int)
print(f"k_array: {k_array}")
# k = 20  # Change this to your desired number of clusters

# Pick the number of initializations you want to iterate over (to try different values of n_init)
inits = np.linspace(10, 50, 5).round().astype(int)
print(f"inits: {inits}")
combinations = list(itertools.product(k_array, inits))
# print(f"combinations: {combinations}")
N = len(combinations)

# The actual grid search and k-means clustering is done in the following function
def get_kmeans_score_and_labels(combinations, A, init_cluster_assignments):
    scores_list = []
    labels_list = []
    old_cluster_assignments = init_cluster_assignments

    for i, (k, init) in enumerate(combinations):
        # Perform k-means clustering on the affinity matrix

        for iteration in range(max_iterations):
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=init, random_state=0)
            kmeans.fit(A)

            # Get the cluster assignments
            labels = kmeans.labels_
            labels_set = set(labels)
            num_clusters = len(labels_set)

            # Check for convergence (e.g., change in cluster assignments)
            if check_convergence(old_cluster_assignments, labels):
                # if -1 in labels_set:
                #     num_clusters -= 1

                if num_clusters < 2:
                    scores_list.append(-10)
                    labels_list.append('none')
                    c = (k, init)
                    print(f"combination: {c} on iteration {i+1} of {N} resulted in {num_clusters} clusters. Moving on.")
                    continue

                # Compute the silhouette score. Here, we use the weighted adjacency matrix to determine the quality of the clusters.
                scores_list.append(silhouette_score(A_torch.cpu(), labels))
                labels_list.append(labels)
                print(f"Converged at {iteration + 1} iterations. index: {i}, score: {scores_list[-1]}, clusters: {num_clusters}, init: {init}, labels: {labels_list[-1]}")
                break
                # continue

            # Update old_cluster_assignments to check for convergence in the next iteration
            old_cluster_assignments = labels.copy()

    # Collect and return the best result
    best_index = np.argmax(scores_list)
    best_params = combinations[best_index]
    best_labels = labels_list[best_index]
    best_score = scores_list[best_index]

    return {'best k': best_params[0],
            'best n_init': best_params[1], 
            'best_score': best_score, 
            'best_labels': best_labels}

# run the algorithm, and get the best result
best_results = get_kmeans_score_and_labels(combinations, A, old_cluster_assignments)
print(f"best_results: {best_results}")

# save the cluster labels to a file. This can then be joined to your network data to display the cluster assignments for each node.
np.savetxt('cluster_assignments.csv', best_results['best_labels'], delimiter=',', fmt='%d') 






