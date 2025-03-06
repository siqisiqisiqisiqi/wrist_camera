#!/home/grail/.virtualenvs/yolo_keypoint/bin/python
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(PARENT_DIR)

import cv2
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO

from src.point_util import yolo2, depth_acquisit, pixel_to_camera
from wrist_camera.msg import Kpt, Kpts
from wrist_camera.srv import ObjectDetection, ObjectDetectionResponse


class Camera:
    def __init__(self):
        rospy.init_node("wirst_camera", anonymous=True)

        # Init variable
        self.rate = rospy.Rate(10)
        self.bridge = CvBridge()
        self.cv_image = None
        self.cv_depth = None
        self.image_t = None
        self.depth_t = None

        # Init subscribers
        rospy.Subscriber("realsense/camera/color/image_raw",
                         Image, self.get_image)
        rospy.Subscriber(
            "realsense/camera/depth/image_rect_raw", Image, self.get_depth)

        # YOLO init
        self.model = YOLO(f"{PARENT_DIR}/weights/bunched_rgb_kpts.pt")

        # Warm-up the model
        results = self.model(f"{PARENT_DIR}/test/test0.jpg", conf=0.2)

        # Create the service
        self.service = rospy.Service(
            "detect_object", ObjectDetection, self.detect_objects)

    def get_image(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.image_t = 1e-9 * data.header.stamp.nsecs + \
                data.header.stamp.secs
            # rospy.loginfo(f"image size is {self.cv_image.shape}")
        except CvBridgeError as e:
            print(e)

    def get_depth(self, data):
        try:
            self.cv_depth = self.bridge.imgmsg_to_cv2(data, "16UC1")
            self.cv_depth = np.where(self.cv_depth > 500, 0, self.cv_depth)
            self.depth_t = 1e-9 * data.header.stamp.nsecs + \
                data.header.stamp.secs
            # rospy.loginfo(f"depth image size is {self.cv_depth.shape}")
        except CvBridgeError as e:
            print(e)

    def pub_keypoints(self, pro_points):
        # publish the keypoints
        pub_kpts_data = Kpts()
        pub_kpts_data.num = pro_points.shape[0]
        for pro_point in pro_points:
            kpt_data = Kpt()
            kpt_data.data = pro_point.tolist()
            pub_kpts_data.kpt.append(kpt_data)
        self.kpt_pub.publish(pub_kpts_data)

    def inference(self, img):
        results = self.model(img, conf=0.2)
        if results[0].keypoints.conf is None:
            rospy.loginfo(f"detect confidence is not enough")
            return None, None
        points, bbox, num_object = yolo2(results)
        if num_object == 0:
            rospy.loginfo(f"key point confidence is not enough")
            return None, None
        return points, bbox

    def detect_objects(self, req):
        rospy.loginfo("Received object detection request...")

        img = np.copy(self.cv_image)
        depth_img = np.array(self.cv_depth, dtype=np.float32)
        points, bbox = self.inference(img)
        try:
            # # depth acquisition
            depths = depth_acquisit(points, depth_img, bbox)

            # # transformation
            P_c = pixel_to_camera(points, depths)
            x_coords = list(P_c[:, 0].astype(np.float32))
            y_coords = list(P_c[:, 1].astype(np.float32))
            z_coords = list(P_c[:, 2].astype(np.float32))
            rospy.loginfo("Completed the service!")
            return ObjectDetectionResponse(x_coords, y_coords, z_coords)
        except:
            rospy.loginfo("can't detect the green onion")
            return ObjectDetectionResponse([], [], [])

    def run(self):
        save_index = 0

        while not rospy.is_shutdown():
            if self.cv_image is None or self.cv_depth is None:
                rospy.loginfo("Waiting for camera initialization...")
                rospy.sleep(1)
                continue

            img = np.copy(self.cv_image)
            depth_img = np.array(self.cv_depth, dtype=np.float32)

            cv2.imshow("wrist_mage", img)
            k = cv2.waitKey(5)
            if k == ord('q'):
                break
            if k == ord('s'):
                cv2.imwrite(
                    f'{PARENT_DIR}/test/test{save_index}.jpg', self.cv_image)
                np.save(
                    f'{PARENT_DIR}/test/test{save_index}.npy', depth_img)
                save_index = save_index + 1

            self.rate.sleep()


if __name__ == "__main__":
    try:
        node = Camera()
        node.run()
    except rospy.ROSInterruptException:
        pass

    cv2.destroyAllWindows()
    rospy.loginfo("Finish the code!")
