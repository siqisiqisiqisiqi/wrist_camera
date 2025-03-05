import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

import numpy as np
from numpy.linalg import inv
from sklearn.mixture import GaussianMixture
import cv2

extrinsic_param = os.path.join(PARENT_DIR, 'params/intrinsic_param.npz')

with np.load(extrinsic_param) as X:
    mtx, dist = [X[i] for i in ('mtx', 'dist')]

fx = mtx[0, 0]
fy = mtx[1, 1]
cx = mtx[0, 2]
cy = mtx[1, 2]
camera_matrix = mtx
dist_coeffs = dist


def pixel_to_camera(uv_distorted, depths):
    """
    Converts a single pixel (u, v) at depth Z into a 3D point (X_c, Y_c, Z_c)
    in the camera coordinate system, given camera intrinsics and distortion.

    Parameters
    ----------
    u, v : float
        Pixel coordinates in the distorted image.
    Z : float
        Depth from the camera (the distance along the camera's Z axis).
    camera_matrix : np.ndarray of shape (3,3)
        The intrinsic matrix: 
            [[fx, 0,  cx],
             [0,  fy, cy],
             [0,   0,  1 ]]
    dist_coeffs : np.ndarray of shape (5,) or similar
        Distortion coefficients [k1, k2, p1, p2, k3].

    Returns
    -------
    X_c, Y_c, Z_c : float
        The 3D coordinates of that pixel in the camera frame.
    """
    # Pack (u, v) into the shape (N,1,2). Here N=1 since we have one point.
    # uv_distorted = np.array([[[u, v]]], dtype=np.float32)  # shape (1,1,2)
    P_c = np.zeros((depths.shape[0], 3))
    uv_distorted = np.expand_dims(uv_distorted, axis=1)

    # undistortPoints transforms pixel -> normalized coords (x_n, y_n)
    # in an ideal pinhole camera (f=1, principal point=(0,0)).
    uv_undist = cv2.undistortPoints(
        uv_distorted,
        camera_matrix,
        dist_coeffs
    )  # shape (1,1,2)

    for i, Z in enumerate(depths):
        # Extract normalized coordinates
        x_n = uv_undist[i, 0, 0]  # x normalized
        y_n = uv_undist[i, 0, 1]  # y normalized

        # Now, these normalized coords assume fx=1, fy=1, cx=cy=0.
        # But in a real camera, the actual 3D point in the camera frame is:
        #   X_c = x_n * Z
        #   Y_c = y_n * Z
        #   Z_c = Z
        #
        # Because we measure depth = distance along Z_c.
        # The direction is given by the normalized ray (x_n, y_n, 1),
        # and we scale it by Z.

        X_c = float(x_n * Z)
        Y_c = float(y_n * Z)
        Z_c = float(Z)
        P_c[i] = np.array([X_c, Y_c, Z_c])

    return P_c


def yolo2(yolo_pose_results):
    """Extract the yolo pose estimation information

    Parameters
    ----------
    yolo_pose_results : ultralytics return
        yolo pose result

    Returns
    -------
    points: ndarray
        the second and third keypoints in homogeneous format
    visual: ndarray
        the second and third keypoints visualbility
    """
    confidence_thresdhold = 0.5
    confidence = yolo_pose_results[0].keypoints.conf.detach().cpu().numpy()
    conf_mean = np.mean(confidence, axis=-1)
    indices = np.where(conf_mean > confidence_thresdhold)
    keypoints = yolo_pose_results[0].keypoints.data.detach().cpu().numpy()
    num_object = len(indices[0])
    points = keypoints[indices[0], 1, :2]

    bboxes = yolo_pose_results[0].boxes.xyxy.detach().cpu().numpy()
    bboxes = bboxes[indices[0], :]
    return points, bboxes, num_object


def yolo2bbox(yolo_pose_results):
    confidence_thresdhold = 0.5
    confidence = yolo_pose_results[0].keypoints.conf.detach().cpu().numpy()
    conf_mean = np.mean(confidence, axis=-1)
    indices = np.where(conf_mean > confidence_thresdhold)
    bboxes = yolo_pose_results[0].boxes.xyxy.detach().cpu().numpy()
    num_object = len(indices[0])
    points = bboxes[indices[0], :]
    return points, num_object


def depth_acquisit(points, depth_map, bboxes):
    depths = np.zeros(bboxes.shape[0])
    for i, bbox in enumerate(bboxes):
        bbox = bbox.astype(np.int32)
        x_min, y_min, x_max, y_max = bbox
        roi_depth = depth_map[y_min:y_max, x_min:x_max]

        # import matplotlib.pyplot as plt
        # roi_depth = np.where(roi_depth == 0, np.NaN, roi_depth)
        # fig, axs = plt.subplots()
        # axs.imshow(roi_depth, cmap='viridis')
        # # axs.imshow(depth, cmap='jet')
        # axs.set_title(f'depth map')
        # plt.show()

        # import matplotlib.pyplot as plt
        # # Flatten the ROI depth values for histogram
        # roi_depth_flattened = roi_depth.flatten()
        # # Plot histogram
        # plt.figure(figsize=(8, 6))
        # plt.hist(roi_depth_flattened, bins=50, color='blue', alpha=0.7)
        # plt.xlabel("Depth Value")
        # plt.ylabel("Frequency")
        # plt.title("Depth Distribution in Bounding Box")
        # plt.grid(True)
        # plt.show()

        roi_depth = np.where(roi_depth == np.NaN, 0, roi_depth)
        roi_depth_flattened = roi_depth.flatten()
        roi_depth_filtered = roi_depth_flattened[roi_depth_flattened > 0]
        roi_depth_filtered = roi_depth_filtered.reshape((-1, 1))
        gmm = GaussianMixture(n_components=3, random_state=42)
        gmm.fit(roi_depth_filtered)

        labels = gmm.predict(roi_depth_filtered)
        unique_labels, counts = np.unique(labels, return_counts=True)
        # Most frequent Gaussian component
        dominant_cluster_idx = unique_labels[np.argmax(counts)]

        # Extract depth values from dominant cluster
        dominant_depth_values = roi_depth_filtered[labels ==
                                                   dominant_cluster_idx]

        # Estimate the most accurate depth
        final_depth_gmm = np.mean(dominant_depth_values)
        depths[i] = final_depth_gmm
        # print(f"Estimated Dominant Depth (GMM): {final_depth_gmm:.2f}")

    return depths
