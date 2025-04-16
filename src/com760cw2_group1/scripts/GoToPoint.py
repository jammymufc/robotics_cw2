#! /usr/bin/env python3


# Import ROS libraries and messages
import rospy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from tf import transformations
from std_srvs.srv import SetBool, SetBoolResponse


import math


class GoToPoint:
    def __init__(self):
        # Initialize node, publishers, subscribers, and services
        rospy.init_node('go_to_point')
        
        # Robot's active state
        self.active = False
        # Robot's current position and orientation
        self.position = Point()
        self.yaw = 0
        
        # Goal position
        self.desired_position = Point()
        self.desired_position.x = rospy.get_param('des_pos_x')
        self.desired_position.y = rospy.get_param('des_pos_y')
        self.desired_position.z = 0
        
        # State of the machine
        self.state = 0
        
        # Precision parameters
        self.yaw_precision = math.pi / 90  # +/- 2 degrees allowed
        self.dist_precision = 0.3
        
        # Publisher setup
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # Subscriber setup
        rospy.Subscriber('/odom', Odometry, self.clbk_odom)
        
        # Service setup
        rospy.Service('go_to_point_switch', SetBool, self.go_to_point_switch)
        
        # Rate setup
        self.rate = rospy.Rate(20)


    def go_to_point_switch(self, req):
        # Service callback to activate/deactivate the robot's movement
        self.active = req.data
        res = SetBoolResponse()
        res.success = True
        res.message = 'Done!'
        return res


    def clbk_odom(self, msg):
        # Callback for updating the robot's current position and orientation
        self.position = msg.pose.pose.position
        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w)
        euler = transformations.euler_from_quaternion(quaternion)
        self.yaw = euler[2]


    def change_state(self, state):
        # Change the state of the robot's behavior
        self.state = state
        rospy.loginfo('State changed to [{}]'.format(self.state))


    def normalize_angle(self, angle):
        # Normalize the angle to the range [-pi, pi]
        if math.fabs(angle) > math.pi:
            angle = angle - (2 * math.pi * angle) / math.fabs(angle)
        return angle


    def fix_yaw(self):
        # Adjust the robot's orientation to face the desired position
        desired_yaw = math.atan2(self.desired_position.y - self.position.y, self.desired_position.x - self.position.x)
        err_yaw = self.normalize_angle(desired_yaw - self.yaw)


        rospy.loginfo(err_yaw)


        twist_msg = Twist()
        if math.fabs(err_yaw) > self.yaw_precision:
            twist_msg.angular.z = 0.7 if err_yaw > 0 else -0.7
        
        self.pub.publish(twist_msg)
        
        if math.fabs(err_yaw) <= self.yaw_precision:
            rospy.loginfo('Yaw error: [{}]'.format(err_yaw))
            self.change_state(1)


    def go_straight_ahead(self):
        # Move the robot straight towards the desired position
        desired_yaw = math.atan2(self.desired_position.y - self.position.y, self.desired_position.x - self.position.x)
        err_yaw = desired_yaw - self.yaw
        err_pos = math.sqrt(pow(self.desired_position.y - self.position.y, 2) + pow(self.desired_position.x - self.position.x, 2))
        
        if err_pos > self.dist_precision:
            twist_msg = Twist()
            twist_msg.linear.x = 0.6
            twist_msg.angular.z = 0.2 if err_yaw > 0 else -0.2
            self.pub.publish(twist_msg)
        else:
            rospy.loginfo('Position error: [{}]'.format(err_pos))
            self.change_state(2)
        
        if math.fabs(err_yaw) > self.yaw_precision:
            rospy.loginfo('Yaw error: [{}]'.format(err_yaw))
            self.change_state(0)


    def done(self):
        # Stop the robot once the goal is reached
        twist_msg = Twist()
        twist_msg.linear.x = 0
        twist_msg.angular.z = 0
        self.pub.publish(twist_msg)


    def run(self):
        # Main loop to check the robot's state and perform actions accordingly
        while not rospy.is_shutdown():
            if not self.active:
                self.rate.sleep()
                continue


            if self.state == 0:
                self.fix_yaw()
            elif self.state == 1:
                self.go_straight_ahead()
            elif self.state == 2:
                self.done()
            else:
                rospy.logerr('Unknown state!')


            self.rate.sleep()


if __name__ == '__main__':
    navigator = GoToPoint()
    navigator.run()
