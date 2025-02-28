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



class Camera:
    def __init__(self):
        rospy.init_node("camera", anonymous=True)

        # Init variable
        self.rate = rospy.Rate(10)
        self.bridge = CvBridge()
        self.cv_image = None
        self.cv_depth = None
        self.image_t = None
        self.depth_t = None

        # Init subscribers
        rospy.Subscriber("realsense/camera/color/image_raw", Image, self.get_image)
        rospy.Subscriber("realsense/camera/depth/image_rect_raw", Image, self.get_depth)

    def get_image(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "rgb8")
            self.image_t = 1e-9 * data.header.stamp.nsecs + \
                data.header.stamp.secs
            rospy.loginfo(f"image size is {self.cv_image.shape}")
        except CvBridgeError as e:
            print(e)

    def get_depth(self, data):
        try:
            self.cv_depth = self.bridge.imgmsg_to_cv2(data, "16UC1")
            self.depth_t = 1e-9 * data.header.stamp.nsecs + \
                data.header.stamp.secs
            rospy.loginfo(f"depth image size is {self.cv_depth.shape}")
        except CvBridgeError as e:
            print(e)

    def run(self):

        while not rospy.is_shutdown():
            if self.cv_image is None or self.cv_depth is None:
                rospy.loginfo("Waiting for camera initialization...")
                rospy.sleep(1)
                continue
            img = np.copy(self.cv_image)
            depth_img = np.array(self.cv_depth, dtype=np.float32)

            # cv2.imshow("image", img)
            # cv2.waitKey(5)
            
            depth_normalized = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
            depth_normalized = np.uint8(depth_normalized)

            # Apply a colormap for better visualization
            depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

            # Show the depth image
            cv2.imshow("Depth Image", depth_colormap)
            cv2.waitKey(1)
            self.rate.sleep()


if __name__ == "__main__":
    try:
        node = Camera()
        node.run()
    except rospy.ROSInterruptException:
        pass

    cv2.destroyAllWindows()
    rospy.loginfo("Finish the code!")
