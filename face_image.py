#!/usr/bin/env python
from __future__ import print_function

import roslib
roslib.load_manifest('tests_cv')
import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

HAAR_CASCADE_XML_FILE_FACE = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"

GSTREAMER_PIPELINE = 'nvarguscamerasrc ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=21/1 ! nvvidconv flip-method=0 ! video/x-raw, width=960, height=616, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink'


class image_converter:

  def __init__(self):
    self.image_pub = rospy.Publisher("image_topic_2",Image,queue_size=10)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/camera/color/image_raw",Image,self.callback)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    # Obtain face detection Haar cascade XML files from OpenCV
    face_cascade = cv2.CascadeClassifier(HAAR_CASCADE_XML_FILE_FACE)

    (rows,cols,channels) = cv_image.shape
    grayscale_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    detected_faces = face_cascade.detectMultiScale(grayscale_image, 1.3, 5)
    # Create rectangle around the face in the image canvas
    for (x_pos, y_pos, width, height) in detected_faces:
        cv2.rectangle(cv_image, (x_pos, y_pos), (x_pos + width, y_pos + height), (0, 0, 0), 2)


    cv2.imshow("Image window", cv_image)
    cv2.waitKey(3)

    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
      print(e)

def main(args):
  ic = image_converter()
  rospy.init_node('image_converter', anonymous=True)
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
