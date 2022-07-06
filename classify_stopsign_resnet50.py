#!/usr/bin/env python3

# -*- coding: utf-8 -*-
import sys
import cv2

import rospy
import roslib
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from object_recognition.msg import Predictor
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

from geometry_msgs.msg import Twist
from ackermann_msgs.msg import AckermannDriveStamped

GPU_OPTIONS = tf.compat.v1.GPUOptions(allow_growth=True)
CONFIG = tf.compat.v1.ConfigProto(gpu_options=GPU_OPTIONS)
CONFIG.gpu_options.per_process_gpu_memory_fraction = 0.5

sess = tf.compat.v1.Session(config=CONFIG)
tf.compat.v1.keras.backend.set_session(sess)
model = ResNet50(weights='imagenet')

graph = tf.compat.v1.get_default_graph()
target_size = (224, 224)

bridge = CvBridge()

stop_sign_detected = False
stop_sign_debounce_counter = 0

def callback(image_msg):
    global stop_sign_detected
    global stop_sign_debounce_counter
    
    #First convert the image to OpenCV image 
    cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="passthrough")
    height, width, channels = cv_image.shape
    #print(cv_image.shape)
    cv_image_target = cv2.resize(cv_image, target_size) 
    np_image = np.asarray(cv_image)               # read as np array
    np_image = np.expand_dims(np_image, axis=0)   # Add another dimension for tensorflow
    np_image = np_image.astype(float)  # preprocess needs float64 and img is uint8
    np_image = preprocess_input(np_image)         # Regularize the data
    
    preds = model.predict(np_image)            # Classify the image
    pred_string = decode_predictions(preds, top=1)   # Decode top 1 predictions
    print(pred_string)
    
    # Get the detected object and it's confidence
    # Example output for detecting stop sign. In ResNet50 stop sign is labelled as street_sign.
    # [[('n06794110', 'street_sign', 0.48952413)]] [[('n06794110', 'street_sign', 0.36558995)]] [[('n06794110', 'street_sign', 0.47690836)]] [[('n06794110', 'street_sign', 0.41784364)]] [[('n06794110', 'street_sign', 0.50704765)]]
    object_detected = pred_string[0][0][1]
    object_detect_conf = pred_string[0][0][2]
    
    # Checking if stop sign is detected with scores greater than a threshold i.e the confidence level.
    if (object_detected == "street_sign" and object_detect_conf > 0.0) or stop_sign_detected:  # Using stop_sign_detected as another condition to the counter since image scan will detect multiple objects and stop sign might not be in the detected list continuously.
      stop_sign_detected = True
      stop_sign_debounce_counter += 1
      print(stop_sign_debounce_counter)
      if stop_sign_debounce_counter > 50 and stop_sign_debounce_counter < 400:  # Checking if the stop sign is detected for more than some time then set the velocity to zero until some time before resuming the speed.
        velocity = 0
      if stop_sign_debounce_counter >= 400:  # Reset the stop_sign_detected after coming to a stop.
        stop_sign_detected = False  # Reset the detection flag
        stop_sign_debounce_counter = 0  # Reset the counter once stop sign is crossed.
      
    # Publish the message
    # ---
    msg = AckermannDriveStamped()
    msg.drive.speed = velocity
    pubDrive.publish(msg)
    # ---

rospy.init_node('classify', anonymous=True)

rospy.Subscriber("usb_cam/image_raw", Image, callback, queue_size = 1, buff_size = 16777216)

pub = rospy.Publisher('object_detector', Predictor, queue_size = 10)
pubDrive = rospy.Publisher('/vesc/ackermann_cmd_mux/input/teleop', AckermannDriveStamped, queue_size=1)

while not rospy.is_shutdown():
  rospy.spin()

