#! /usr/bin/bash

# The file will run the opening commands

cd .. 

source devel/setup.bash

# Function to check for user input
check_input() {
    read -t 0.1 -n 1 key
    if [[ $key == "q" ]]; then
        # Kill background processes
        pkill -P $$
        exit
    fi
}
# Get the name of the current directory
current_dir=$(basename "$PWD")

robot_ip=192.168.1.102

config_path=$HOME/${current_dir}/src/my_robot_calibration.yaml

roslaunch ur_robot_driver ur3e_bringup.launch \
 robot_ip:=${robot_ip} robot_description_file:=$HOME/${current_dir}/src/ur3e_moveit_config/launch/load_ur3e.launch \
 kinematics_config:=${config_path} &
pid1=$!

sleep 5

roslaunch ur3e_moveit_config moveit_planning_execution.launch &
pid2=$!
sleep 3

rosrun gripper service.py &
pid3=$!

rosrun scannode arucoscan.py &
pid4=$!

# Main loop to check for user input
while true; do
    check_input
done