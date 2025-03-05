#!/home/grail/.virtualenvs/yolo_keypoint/bin/python
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(PARENT_DIR)

import cv2
import rospy
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from std_srvs.srv import Trigger

from src.calib_util import coordinates_draw


class Calibration:
    def __init__(self):
        rospy.init_node("calibration", anonymous=True)

        # Init variable
        self.rate = rospy.Rate(10)
        self.bridge = CvBridge()
        self.rvecs = None
        self.tvecs = None
        self.mtx = None
        self.dist = None
        self.index = 0
        # self.chess_size = 22.8  # unit: mm
        self.chess_size = 12.93  # unit: mm
        self.camera_poses = []

        # Init subscribers
        rospy.Subscriber("realsense/camera/color/image_raw",
                         Image, self.get_image)

        rospy.wait_for_service('/save_pose')
        self.save_pose = rospy.ServiceProxy('/save_pose', Trigger)

        rospy.wait_for_service('/append_pose')
        self.append_pose = rospy.ServiceProxy('/append_pose', Trigger)

    def get_image(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.image_t = 1e-9 * data.header.stamp.nsecs + \
                data.header.stamp.secs
        except CvBridgeError as e:
            print(e)

    def get_camera_info(self):
        camera_info = rospy.wait_for_message(
            "realsense/camera/color/camera_info", CameraInfo)
        self.mtx = np.array(camera_info.K).reshape(3, 3)
        self.dist = np.array(camera_info.D)[:5]
        # np.savez(f'{PARENT_DIR}/params/intrinsic_param.npz', mtx=self.mtx, dist=self.dist)

    def save_img(self, img, index):
        # save calibration image
        cv2.imwrite(f'{PARENT_DIR}/EIH_calibration/calib{index}.jpg', img)

        # save camera parameter
        Mat, _ = cv2.Rodrigues(self.rvecs)
        tvec = self.tvecs * self.chess_size
        T_cam = np.eye(4)
        T_cam[:3, :3] = Mat
        T_cam[:3, 3] = tvec.flatten()
        self.camera_poses.append(T_cam)
        # rospy.loginfo("Successfully save the calibration parameters.")

    def extrinsic_calibration(self):
        img = self.cv_image
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((6 * 9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
        axis = np.float32([[3, 0, 0], [0, 6, 0], [0, 0, -1]]).reshape(-1, 3)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
        if ret == True:
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria)

            # Find the rotation and translation vectors.
            ret, self.rvecs, self.tvecs = cv2.solvePnP(
                objp, corners2, self.mtx, self.dist)

            # # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(
                axis, self.rvecs, self.tvecs, self.mtx, self.dist)

            img = coordinates_draw(img, corners2, imgpts)
            self.save_img(img, self.index)
            self.index += 1
            rospy.loginfo("save the camera pose relative to the checkerboard ")
        else:
            rospy.loginfo("Can't detect the checkboard!!!")

    def run(self):
        self.get_camera_info()
        rospy.sleep(2)

        while not rospy.is_shutdown():
            img = np.copy(self.cv_image)

            # visualization
            cv2.imshow("image", img)
            k = cv2.waitKey(5)

            # extrinsic calibration
            if k == ord('s'):
                self.extrinsic_calibration()
                response = self.append_pose()
            if k == ord('q'):
                camera_pose = np.array(self.camera_poses)
                np.save(
                    f"{PARENT_DIR}/EIH_calibration/camera_pose.npy", camera_pose)
                response = self.save_pose()
                break

            self.rate.sleep()


if __name__ == "__main__":
    try:
        node = Calibration()
        node.run()
    except rospy.ROSInterruptException:
        pass

    cv2.destroyAllWindows()
    rospy.loginfo("Exiting camera Extrinsic calibration node!")
