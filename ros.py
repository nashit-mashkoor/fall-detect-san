#!/usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('tests_cv')
import sys
import rospy
import cv2
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from detect_fall_per_frame import FallDetectorOnSingleFrame

GSTREAMER_PIPELINE = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=21/1 ! nvvidconv flip-method=0 ! video/x-raw, width=960, height=616, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'

class RosFallDetection:

  def __init__(self):
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.callback)
    self.f = FallDetectorOnSingleFrame()
    self.ip_sets = [[] for _ in range(self.f.argss[0].num_cams)]
    self.lstm_sets = [[] for _ in range(self.f.argss[0].num_cams)]
    self.num_matched = 0

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    print("CV_image captured")

    print("------------------------")
    start = time.time()
    dict_vis = self.f.extract_keypoints_parallel(cv_image, self.f.argss[0])
    self.num_matched, new_num, indxs_unmatched, result = self.f.alg2_sequential(dict_vis, self.f.argss[0], self.f.consecutive_frames,
                                                                      self.ip_sets, self.lstm_sets, self.num_matched)
    print("Pred time on frame " + str(self.f.frame) + ": " + str(time.time() - start))
    print("------------------------")

    cv2.imshow("Image window", cv_image)
    cv2.waitKey(3)

def main(args):
  rfd = RosFallDetection()
  rospy.init_node('RosFallDetection', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)