#! /usr/bin/env python3

# import ros stuff
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from tf import transformations
from std_srvs.srv import *
from com760cw2_group1.msg import group1homingsignal

import math

class WallFollower:
    def __init__(self):
        self.active = False
        self.turning_right = False
        self.regions = {
            'right': 0,
            'fright': 0,
            'front': 0,
            'fleft': 0,
            'left': 0,
        }
        self.state = 0
        self.state_dict = {
            0: 'find the wall',
            1: 'turn left',
            2: 'follow the wall',
            3: 'turn right',  # New state for turning right
            4: 'find wall left'
        }
       
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        rospy.Subscriber('/group1Bot/laser/scan', LaserScan, self.clbk_laser)
        rospy.Subscriber('/group1_homing_signal', group1homingsignal, self.clbk_homing_signal)
        rospy.Service('wall_follower_switch', SetBool, self.wall_follower_switch)

    def wall_follower_switch(self, req):
        self.active = req.data
        res = SetBoolResponse()
        res.success = True
        res.message = 'Done!'
        return res

    def clbk_laser(self, msg):
        self.regions = {
            'right':  min(min(msg.ranges[0:143]), 10),
            'fright': min(min(msg.ranges[144:287]), 10),
            'front':  min(min(msg.ranges[288:431]), 10),
            'fleft':  min(min(msg.ranges[432:575]), 10),
            'left':   min(min(msg.ranges[576:719]), 10),
        }
        
        self.take_action()

    def clbk_homing_signal(self, data):
        
        self.instructionID = data.instructionID
        if self.instructionID == 1:
            rospy.loginfo('turning right')
            self.turning_right = True  

    def change_state(self, state):
        if state != self.state:
            print('Wall follower - [%s] - %s' % (state, self.state_dict[state]))
            self.state = state

    def take_action(self):
        d = 1.5
        regions = self.regions
        if self.turning_right == True:
        
            if regions['front'] > d and regions['fleft'] > d and regions['fright'] > d:
                self.change_state(4)
            elif regions['front'] < d and regions['fleft'] > d and regions['fright'] > d:
                self.change_state(3)
            elif regions['front'] > d and regions['fleft'] > d and regions['fright'] < d:
                self.change_state(2)
            elif regions['front'] < d and regions['fleft'] < d and regions['fright'] < d:
                self.change_state(3)
        
        else:

            if regions['front'] > d and regions['fleft'] > d and regions['fright'] > d:
                self.change_state(0)
            elif regions['front'] < d and regions['fleft'] > d and regions['fright'] > d:
                self.change_state(1)
            elif regions['front'] > d and regions['fleft'] > d and regions['fright'] < d:
                self.change_state(2)
            elif regions['front'] < d and regions['fleft'] < d and regions['fright'] < d:
                self.change_state(1)
            else:
                rospy.loginfo(regions)

    def find_wall(self):
        msg = Twist()
        msg.linear.x = 0.2
        msg.angular.z = -0.3
        return msg

    def find_wall_left(self):
        msg = Twist()
        msg.linear.x = 0.2
        msg.angular.z = 0.2
        return msg

    def turn_left(self):
        msg = Twist()
        msg.angular.z = 0.3
        return msg

    def turn_right(self):  # New method for turning right
        msg = Twist()
        msg.angular.z = -0.3
        print('turn right')
        return msg

    def follow_the_wall(self):
        msg = Twist()
        msg.linear.x = 0.5
        return msg

    def run(self):
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if not self.active:
                rate.sleep()
                continue
            
            msg = Twist()
            if self.state == 0:
                msg = self.find_wall()
            elif self.state == 1:
                msg = self.turn_left()
            elif self.state == 2:
                msg = self.follow_the_wall()
            elif self.state == 3:
                msg = self.turn_right()  # Use the new turn_right method
            elif self.state == 4:
                msg = self.find_wall_left()
            else:
                rospy.logerr('Unknown state!')
            
            self.pub.publish(msg)
            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('reading_laser')
    wall_follower = WallFollower()
    wall_follower.run()
