import matplotlib.pyplot as plt
import csv
import math
import numpy as np

# function to calculate euclidean distance between two points
def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

# implementation of k-means clustering algorithm
def kmeans(data, k, max_iterations=100):
    # step 1: select first k data points as initial centroids
    centroids = data[:k].copy()
    
    prev_centroids = None
    iteration = 0
    
    while iteration < max_iterations:
        # check if centroids have converged
        if prev_centroids is not None and np.array_equal(centroids, prev_centroids):
            break
            
        # copy current centroids for convergence check
        prev_centroids = centroids.copy()
        
        # step 2: assign each data point to closest centroid
        # initialize clusters
        clusters = [[] for _ in range(k)]
        
        for point in data:
            # calculate distances to each centroid
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            # find closest centroid
            closest_centroid = distances.index(min(distances))
            # assign point to cluster
            clusters[closest_centroid].append(point)
        
        # step 4: compute new centroids by averaging points in each cluster
        for i in range(k):
            if len(clusters[i]) > 0:
                # calculate average of all points in the cluster
                centroids[i] = np.mean(clusters[i], axis=0)
        
        iteration += 1
    
    # return final clusters and centroids
    return clusters, centroids, iteration

# read data from CSV file
def read_csv(filename):
    data = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        # skip header
        next(csvreader)
        for row in csvreader:
            data.append([float(row[0]), float(row[1])])
    return np.array(data)

# main function
def main():
    # read data
    data = read_csv('CustomerProfiles_Q2.csv')
    
    # 1) plot customer data points before clustering
    plt.figure(figsize=(10, 6))
    plt.scatter(data[:, 0], data[:, 1], color='blue', marker='o', label='Customers')
    plt.title('Customer Profiles Before Clustering')
    plt.xlabel('Average Monthly Spending (x1)')
    plt.ylabel('Total Number of Purchases (x2)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('customers_before_clustering.png')
    plt.show()
    
    # 2) apply k-means clustering with k=2
    k = 2
    clusters, centroids, iterations = kmeans(data, k)
    
    # 3) plot final clusters
    plt.figure(figsize=(10, 6))
    colors = ['red', 'green']
    
    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        plt.scatter(cluster[:, 0], cluster[:, 1], color=colors[i], marker='o', 
                   label=f'Cluster {i+1} ({len(cluster)} customers)')
    
    # plot centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='X', s=100, 
               label='Centroids')
    
    plt.title(f'Customer Clusters after K-means (k={k}, iterations={iterations})')
    plt.xlabel('Average Monthly Spending (x1)')
    plt.ylabel('Total Number of Purchases (x2)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('customers_after_clustering.png')
    plt.show()
    
    # 4) report number of data points in each cluster
    print("\nNumber of data points in each cluster:")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i+1}: {len(cluster)} customers")
    
    # 5) report final centroid coordinates
    print("\nFinal centroid coordinates:")
    for i, centroid in enumerate(centroids):
        print(f"Centroid {i+1}: ({centroid[0]:.2f}, {centroid[1]:.2f})")
n
if __name__ == "__main__":
    main() 