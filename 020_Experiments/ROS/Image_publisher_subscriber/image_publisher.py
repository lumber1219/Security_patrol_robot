#!/usr/bin/env python
#coding:utf-8
#reference:https://blog.csdn.net/LutherK/article/details/80374109
import sys
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
import rospy
import cv2
from std_msgs.msg import String
import cv_bridge
from sensor_msgs.msg import Image

# def talker():
#     pub = rospy.Publisher('chatter', String, queue_size=10)
#     rospy.init_node('talker', anonymous=True)
#     rate = rospy.Rate(10)  # 10hz
#     while not rospy.is_shutdown():
#         hello_str = "hello world %s" % rospy.get_time()
#         rospy.loginfo(hello_str)
#         pub.publish(hello_str)
#         rate.sleep()
#
#
# if __name__ == '__main__':
#     try:
#         talker()
#     except rospy.ROSInterruptException:
#         pass



def webcamImagePub():
    # 初始化ROS节点
    rospy.init_node('webcam_puber', anonymous=True)

    #registering as a publisher of a ROS topic. queue_size应该小一点，这样可以达到'real-time'否则publisher会发布过去的帧
    img_pub = rospy.Publisher('webcam/image_raw', Image, queue_size= 10)

    rate = rospy.Rate(20) # 20HZ对应刷新频率

    #
    cap = cv2.VideoCapture(0)
    scaling_factor = 1

    bridge = cv_bridge.CvBridge()

    if not cap.isOpened():
        sys.stdout.write("Webcam is not avaliable")
        return -1

    count = 0

    # loop until press 'esc' or 'q'
    while not rospy.is_shutdown():

        ret, frame = cap.read()

        if ret:
            count = count +1
        else:
            rospy.loginfo("Capturing image failed.")
        if count == 2:
            count = 0
            frame = cv2.resize(frame, None, fx = scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
            msg = bridge.cv2_to_imgmsg(frame, encoding = "bgr8")
            img_pub.publish(msg)
            print '** publishing webcam_frame **'
        rate.sleep()


if __name__ == "__main__":
    try:
        webcamImagePub()
    except rospy.ROSInterruptException:
        pass




