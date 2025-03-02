import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

import numpy as np

chess_size = 1  # mm
extrinsic_param = os.path.join(PARENT_DIR, 'params/E.npz')

with np.load(extrinsic_param) as X:
    mtx, dist, Mat, tvecs = [X[i] for i in ('mtx', 'dist', 'Matrix', 'tvec')]

tvec = tvecs * chess_size
fx = mtx[0, 0]
fy = mtx[1, 1]
cx = mtx[0, 2]
cy = mtx[1, 2]


def projection(points, mtx, Mat, tvecs, depth=0):
    """camera back projection

    Parameters
    ----------
    points : ndarray
        keypoints for camera back projection
    mtx : ndarray
        camera intrinsic parameter
    Mat : ndarray
        camera extrinsic parameter: rotation matrix
    tvecs : ndarray
        camera extrinsic parameter: translation vector
    depth : int, optional
        depth of the point in the world coordinate frame, by default 0

    Returns
    -------
    result: ndarray
        point in the world coordinate frame
    """
    num_object = points.shape[0]
    results = np.zeros_like(points)
    for i in range(num_object):
        point = points[i, :].reshape(3, 1)
        point2 = inv(mtx) @ point
        predefined_z = depth[i // 2]
        vec_z = Mat[:, [2]] * predefined_z
        Mat2 = np.copy(Mat)
        Mat2[:, [2]] = -1 * point2
        vec_o = -1 * (vec_z + tvecs)
        result = inv(Mat2) @ vec_o
        results[i] = result.squeeze()
    return results


def yolo2point(yolo_pose_results):
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
    points = keypoints[indices[0], :, :2]

    # one_vector = np.ones(points.shape[:2])
    # one_vector = np.expand_dims(one_vector, axis=2)
    # points = np.concatenate((points, one_vector), axis=2)
    return points, num_object


def yolo2bbox(yolo_pose_results):
    confidence_thresdhold = 0.5
    confidence = yolo_pose_results[0].keypoints.conf.detach().cpu().numpy()
    conf_mean = np.mean(confidence, axis=-1)
    indices = np.where(conf_mean > confidence_thresdhold)
    bboxes = yolo_pose_results[0].boxes.xyxy.detach().cpu().numpy()
    num_object = len(indices[0])
    points = bboxes[indices[0], :, :2]
    return points, num_object


def depth_acquisit(points, depth_img, depth_acquisit_scale=[1, 2, 3]):
    # dim = points.shape[-1]
    # pts = points.reshape((-1, dim))
    # depths = np.zeros(pts.shape[0])

    # # depth_img = depth_img
    # depth_trans = depth_transform(depth_img)
    # rows, cols = depth_trans.shape

    # for i, point in enumerate(pts):
    #     y, x = point[:2].astype(int)
    #     for scale in depth_acquisit_scale:

    #         row_start = max(0, x - scale)
    #         row_end = min(rows, x + scale + 1)
    #         col_start = max(0, y - scale)
    #         col_end = min(cols, y + scale + 1)

    #         neighborhood = depth_trans[row_start:row_end, col_start:col_end]

    #         non_zero_values = neighborhood[neighborhood != 0]
    #         if non_zero_values.size > 0:
    #             depth = np.mean(non_zero_values)  # Use mean of non-zero values
    #             break
    #         else:
    #             depth = 0
    #             continue
    #     depths[i] = depth
    # return depths, depth_trans
    pass


def img2robot(point, depth):
    point = point.reshape((-1, 3))
    depth = np.round(depth, 2)
    result = projection(point, mtx, Mat, tvec, depth)
    result = np.round(result, 2)
    result[:, -1] = depth
    result = result.reshape((-1, 3, 3))
    grasp_point = result[:, 1, :]
    return grasp_point, orientation, result