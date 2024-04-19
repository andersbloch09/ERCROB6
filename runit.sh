#! /usr/bin/bash

# The file will run the opening commands

source devel/setup.bash

# Get the name of the current directory
current_dir=$(basename "$PWD")

robot_ip=192.168.1.102

config_path=$HOME/${current_dir}/my_robot_calibration.yaml

roslaunch ur_robot_driver ur3e_bringup.launch \
 robot_ip:=${robot_ip} robot_description_file:=$HOME/${current_dir}/src/ur3e_moveit_config/launch/load_ur3e.launch \
 kinematics_config:=${config_path} &

sleep 2

roslaunch ur3e_moveit_config moveit_planning_execution.launch &

sleep 2

rosrun gripper service.py &
rosrun scannode arucoscan.py &
