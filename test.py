# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans

# # Flatten and filter depth values (remove zeros)
# roi_depth_flattened = roi_depth.flatten()
# roi_depth_filtered = roi_depth_flattened[roi_depth_flattened > 0]  # Remove zero/invalid depths

# # Apply K-Means clustering
# k = 3  # Choose number of clusters based on histogram analysis
# kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
# roi_depth_filtered = roi_depth_filtered.reshape(-1, 1)  # Reshape for clustering
# kmeans.fit(roi_depth_filtered)

# # Find the dominant cluster (largest in population)
# labels, counts = np.unique(kmeans.labels_, return_counts=True)
# dominant_cluster_idx = labels[np.argmax(counts)]  # Most frequent cluster

# # Extract depth values belonging to the dominant cluster
# dominant_depth_values = roi_depth_filtered[kmeans.labels_ == dominant_cluster_idx]

# # Compute the estimated depth (mean or median of the dominant cluster)
# final_depth = np.mean(dominant_depth_values)
# print(f"Estimated Dominant Depth: {final_depth:.2f}")

# # Plot histogram with clustering result
# plt.figure(figsize=(8, 6))
# plt.hist(roi_depth_filtered, bins=50, color='blue', alpha=0.6, label="Depth Data")
# for center in kmeans.cluster_centers_:
#     plt.axvline(center, color='red', linestyle='dashed', linewidth=2, label=f"Cluster {center[0]:.2f}")
# plt.axvline(final_depth, color='green', linestyle='solid', linewidth=3, label="Dominant Depth Estimate")
# plt.xlabel("Depth Value")
# plt.ylabel("Frequency")
# plt.title("K-Means Clustering for Dominant Depth Extraction")
# plt.legend()
# plt.grid(True)
# plt.show()

import numpy as np

robot_pose = "./EIH_calibration/camera_pose.npy"
data = np.load(robot_pose)

print(data)
