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

roslaunch ur3e_moveit_config demo.launch &
pid1=$!
sleep 5

rosrun gripper service.py &
pid2=$!
rosrun scannode arucoscan.py &
pid3=$!

# Main loop to check for user input
while true; do
    check_input
done