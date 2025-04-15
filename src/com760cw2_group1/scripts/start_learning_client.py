#!/usr/bin/env python

import rospy
import sys
from com760cw2_group1.srv import LearnNavigation

def start_learning_client():
    rospy.init_node('start_learning_client', anonymous=True)
    rospy.loginfo("Waiting for service...")
    
    # Wait for the service to become available
    rospy.wait_for_service('/Group1Bot/learn_navigation')
    
    try:
        # Create a service proxy
        learn_service = rospy.ServiceProxy('/Group1Bot/learn_navigation', LearnNavigation)
        
        # Give the robot time to spawn and stabilize
        rospy.sleep(5.0)
        
        # Call the service to start learning - increased episodes and steps for larger maze
        response = learn_service(True, 150, 300)  # 150 episodes, 300 max steps each
        rospy.loginfo("Service response: {}".format(response.message))
        
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: {}".format(e))

if __name__ == "__main__":
    try:
        start_learning_client()
    except rospy.ROSInterruptException:
        pass