#! /usr/bin/env python3

# import ros stuff
import rospy
# import ros message
from geometry_msgs.msg import Point
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf import transformations
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState
# import ros service
from std_srvs.srv import *
from com760cw2_group1.msg import group1homingsignal

import math
target_is_unreachable = False
srv_client_go_to_point_ = None
srv_client_wall_follower_ = None
yaw_ = 0
yaw_error_allowed_ = 5 * (math.pi / 180) # 5 degrees
position_ = Point()
initial_position_ = Point()
initial_position_.x = rospy.get_param('initial_x')
initial_position_.y = rospy.get_param('initial_y')
initial_position_.z = 0
desired_position_ = Point()
desired_position_.x = rospy.get_param('des_pos_x')
desired_position_.y = rospy.get_param('des_pos_y')
desired_position_.z = 0
regions_ = None
state_desc_ = ['Go to point', 'wall following']
state_ = 0
count_state_time_ = 0 # seconds the robot is in a state
count_loop_ = 0
# 0 - go to point
# 1 - wall following

# Alg1-specific variables
hit_points_ = []
leave_points_ = []
path_length_ = 0
prev_position_ = Point()

# callbacks
def clbk_odom(msg):
    global position_, yaw_, path_length_, prev_position_
    
    # position
    position_ = msg.pose.pose.position
    
    # yaw
    quaternion = (
        msg.pose.pose.orientation.x,
        msg.pose.pose.orientation.y,
        msg.pose.pose.orientation.z,
        msg.pose.pose.orientation.w)
    euler = transformations.euler_from_quaternion(quaternion)
    yaw_ = euler[2]
    
    # Update path length
    path_length_ += math.sqrt((position_.x - prev_position_.x)**2 + (position_.y - prev_position_.y)**2)
    prev_position_ = position_

def clbk_laser(msg):
    global regions_
    regions_ = {
        'right':  min(min(msg.ranges[0:143]), 10),
        'fright': min(min(msg.ranges[144:287]), 10),
        'front':  min(min(msg.ranges[288:431]), 10),
        'fleft':  min(min(msg.ranges[432:575]), 10),
        'left':   min(min(msg.ranges[576:719]), 10),
    }

def change_state(state):
    global state_, state_desc_
    global srv_client_wall_follower_, srv_client_go_to_point_
    global count_state_time_
    count_state_time_ = 0
    state_ = state
    log = "state changed: %s" % state_desc_[state]
    rospy.loginfo(log)
    if state_ == 0:
        resp = srv_client_go_to_point_(True)
        resp = srv_client_wall_follower_(False)
    if state_ == 1:
        resp = srv_client_go_to_point_(False)
        resp = srv_client_wall_follower_(True)
        
        # Check if the current position is a previously visited hit or leave point
        for point, path_len in hit_points_ + leave_points_:
            if math.sqrt((position_.x - point.x)**2 + (position_.y - point.y)**2) < 0.1:
                # Find the shortest path to the last hit point
                last_hit_point, _ = hit_points_[-1]
                shortest_path = find_shortest_path(point, last_hit_point)
                # Navigate to the last hit point using the shortest path
                #navigate_to_point(shortest_path)
                break

def distance_to_line(p0):
    # p0 is the current position
    # p1 and p2 points define the line
    global initial_position_, desired_position_
    p1 = initial_position_
    p2 = desired_position_
    # here goes the equation
    up_eq = math.fabs((p2.y - p1.y) * p0.x - (p2.x - p1.x) * p0.y + (p2.x * p1.y) - (p2.y * p1.x))
    lo_eq = math.sqrt(pow(p2.y - p1.y, 2) + pow(p2.x - p1.x, 2))
    distance = up_eq / lo_eq
    
    return distance

def normalize_angle(angle):
    if(math.fabs(angle) > math.pi):
        angle = angle - (2 * math.pi * angle) / (math.fabs(angle))
    return angle

def store_point(point_type, point):
    if point_type == 'hit':
        hit_points_.append((point, path_length_))
    elif point_type == 'leave':
        leave_points_.append((point, path_length_))
        

def find_shortest_path(start_point, end_point):
    global hit_points_, leave_points_
    
    # Combine hit and leave points into a single list
    points = hit_points_ + leave_points_
    
    # Find the indices of the start and end points in the points list
    start_index = None
    end_index = None
    for i, (point, _) in enumerate(points):
        if math.sqrt((point.x - start_point.x)**2 + (point.y - start_point.y)**2) < 0.1:
            start_index = i
        if math.sqrt((point.x - end_point.x)**2 + (point.y - end_point.y)**2) < 0.1:
            end_index = i
    
    # If either start or end point is not found, return an empty path
    if start_index is None or end_index is None:
        return []
    
    # If start and end points are the same, return a single-point path
    if start_index == end_index:
        return [start_point]
    
    # Determine the direction of the path (forward or backward)
    if start_index < end_index:
        path_indices = range(start_index, end_index + 1)
    else:
        path_indices = range(start_index, end_index - 1, -1)
    
    # Extract the points from the path indices
    path = [points[i][0] for i in path_indices]
    
    return path



def target_is_unreachable():
    global hit_points_, leave_points_
    
    # Check if the robot has crossed the m-line at the same point twice
    for i in range(len(leave_points_)):
        for j in range(i+1, len(leave_points_)):
            if math.sqrt((leave_points_[i][0].x - leave_points_[j][0].x)**2 + 
                         (leave_points_[i][0].y - leave_points_[j][0].y)**2) < 0.2:
                rospy.loginfo("Target is unreachable. The robot crossed the m-line twice at the same point.")
                turn_right = group1homingsignal()
                turn_right.instructionID = 1
                turn_right_pub.publish(homing_signal)
 
            return True
    
    return False

def main():
    global regions_, position_, desired_position_, state_, yaw_, yaw_error_allowed_
    global srv_client_go_to_point_, srv_client_wall_follower_
    global count_state_time_, count_loop_
    
    rospy.init_node('bug2')
    turn_right_pub = rospy.Publisher('group1homingsignal', group1homingsignal, queue_size=10)
    sub_laser = rospy.Subscriber('/group1Bot/laser/scan', LaserScan, clbk_laser)
    sub_odom = rospy.Subscriber('/odom', Odometry, clbk_odom)
    
    rospy.wait_for_service('/go_to_point_switch')
    rospy.wait_for_service('/wall_follower_switch')
    rospy.wait_for_service('/gazebo/set_model_state')
    
    srv_client_go_to_point_ = rospy.ServiceProxy('/go_to_point_switch', SetBool)
    srv_client_wall_follower_ = rospy.ServiceProxy('/wall_follower_switch', SetBool)
    srv_client_set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    
    # set robot position
    model_state = ModelState()
    model_state.model_name = 'group1Bot'
    model_state.pose.position.x = initial_position_.x
    model_state.pose.position.y = initial_position_.y
    resp = srv_client_set_model_state(model_state)
    
    # initialize going to the point
    change_state(0)
    
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        if regions_ == None:
            continue
        
        distance_position_to_line = distance_to_line(position_)
        
        if state_ == 0:
            if regions_['front'] > 0.15 and regions_['front'] < 1:
                store_point('hit', position_)
                change_state(1)
        
        elif state_ == 1:
            if count_state_time_ > 5 and \
               distance_position_to_line < 0.1:
                store_point('leave', position_)
                change_state(0)
                
            if target_is_unreachable():
                rospy.loginfo("Target is unreachable. The robot crossed the m-line twice at the same point.")
                turn_right = group1homingsignal()
                turn_right.instructionID = 1
                turn_right_pub.publish(group1homingsignal)
                rospy.loginfo("Target is unreachable. Exiting.")
               # break
                
        count_loop_ = count_loop_ + 1
        if count_loop_ == 20:
            count_state_time_ = count_state_time_ + 1
            count_loop_ = 0
            
        #rospy.loginfo("distance to line: [%.2f], position: [%.2f, %.2f]", distance_to_line(position_), position_.x, position_.y)
        rate.sleep()

if __name__ == "__main__":
    main()
