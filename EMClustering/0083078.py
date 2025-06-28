import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.spatial.distance as dt
import scipy.stats as stats

group_means = np.array([[-5.0, +0.0],
                        [+0.0, +5.0],
                        [+5.0, +0.0],
                        [+0.0, -5.0],
                        [+0.0, +0.0]])
group_covariances = np.array([[[+0.4, +0.0],
                               [+0.0, +6.0]],
                              [[+6.0, +0.0],
                               [+0.0, +0.4]],
                              [[+0.4, +0.0],
                               [+0.0, +6.0]],
                              [[+6.0, +0.0],
                               [+0.0, +0.4]],
                              [[+6.0, +0.0],
                               [+0.0, +0.4]]])

# read data into memory
data_set = np.genfromtxt("hw05_data_set.csv", delimiter = ",")

# get X values
X = data_set[:, [0, 1]]

# set number of clusters
K = 5

# STEP 2
# should return initial parameter estimates
# as described in the homework description
def initialize_parameters(X, K):
    # your implementation starts below
    
    means = np.genfromtxt("hw05_initial_centroids.csv", delimiter=",")

    N = X.shape[0]
    distances = np.zeros((N, K))
    
    for k in range(K):
        distances[:, k] = np.sum((X - means[k])**2, axis=1)
    
    assignments = np.argmin(distances, axis=1)
    
    covariances = np.zeros((K, 2, 2))
    priors = np.zeros(K)
    
    for k in range(K):
        # Find points assigned to cluster k
        cluster_points = X[assignments == k]
        dist = cluster_points - means[k]
        covariances[k] = np.dot(dist.T, dist) / len(cluster_points)
        priors[k] = len(cluster_points) / N

    # your implementation ends above
    return(means, covariances, priors)

means, covariances, priors = initialize_parameters(X, K)

# STEP 3
# should return final parameter estimates of
# EM clustering algorithm
def em_clustering_algorithm(X, K, means, covariances, priors):
    # your implementation starts below

    N = X.shape[0]
    for _ in range(100):
        H = np.zeros((N, K))
        for k in range(K):
            #gaussian density calc from soft count hik 
            cov_inv = np.linalg.inv(covariances[k])
            cov_det = np.linalg.det(covariances[k])            
            dist = X - means[k]
            log_likelihood = -0.5 * (np.sum((dist @ cov_inv) * dist, axis=1)) - 0.5 * np.log(2 * np.pi * cov_det)
            H[:, k] = priors[k] * np.exp(log_likelihood)
        row_sums = np.sum(H, axis=1, keepdims=True)
        H = H / row_sums #normalize
        
        # M step
        N_k = np.sum(H, axis=0) 
        priors = N_k / N #prior update
        #mean update
        for k in range(K):
            if N_k[k] > 0:
                means[k] = np.sum(H[:, k:k+1] * X, axis=0) / N_k[k]
        
        #covariance update
        for k in range(K):
            if N_k[k] > 0:
                diff = X - means[k]
                weighted_sum = np.zeros((2, 2))
                for i in range(N):
                    weighted_sum += H[i, k] * np.outer(diff[i], diff[i])
                covariances[k] = weighted_sum / N_k[k]
    
    assignments = np.argmax(H, axis=1)
    
    # your implementation ends above
    return(means, covariances, priors, assignments)

means, covariances, priors, assignments = em_clustering_algorithm(X, K, means, covariances, priors)
print(means)
print(priors)

# STEP 4
# should draw EM clustering results as described
# in the homework description
def draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments):
    # your implementation starts below
    
    plt.figure(figsize=(10, 8))
    
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    
    for k in range(K):
        cluster_points = X[assignments == k]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[k], alpha=0.6, s=20)
    
    #had to find a way to render ellipses (from stackoverflow)
    def draw_ellipse(mean, cov, ax, style='--', color='black', alpha=0.8):

        #gaussian density = 0.01 level
        chi2_val = stats.chi2.ppf(0.99, df=2)  # 99% confidence interval for 2D  (this is equivalent to pdf = 0.01 level)
        
        # Eigenvalue decomposition
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
        # Calculate ellipse parameters
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        width = 2 * np.sqrt(chi2_val * eigenvals[0])
        height = 2 * np.sqrt(chi2_val * eigenvals[1])
        
        # Create ellipse
        ellipse = plt.matplotlib.patches.Ellipse(mean, width, height, angle=angle,
                                               fill=False, color=color, 
                                               linestyle=style, alpha=alpha, linewidth=2)
        ax.add_patch(ellipse)
    
    #actual  gaussian densities (dashed lines)
    ax = plt.gca()
    for k in range(K):
        draw_ellipse(group_means[k], group_covariances[k], ax, 
                    style='--', color='black', alpha=0.6)
    
    #estimated Gaussians (solid)
    for k in range(K):
        draw_ellipse(means[k], covariances[k], ax, 
                    style='-', color=colors[k], alpha=0.8)
    

    plt.xlim(-8, 8)
    plt.ylim(-8, 8)
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
    
    # your implementation ends above
    
draw_clustering_results(X, K, group_means, group_covariances, means, covariances, assignments)