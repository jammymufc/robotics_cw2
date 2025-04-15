cmake_minimum_required(VERSION 3.0.2)
project(com760cw2_group1)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  nav_msgs
  message_generation
)

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  State.msg
  Action.msg
  Reward.msg
)

## Generate services in the 'srv' folder
add_service_files(
  FILES
  LearnNavigation.srv
)

## Generate added messages and services with any dependencies
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp rospy std_msgs message_runtime geometry_msgs sensor_msgs nav_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Mark scripts as executable
catkin_install_python(PROGRAMS
  scripts/q_learning_nav.py
  scripts/start_learning_client.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
