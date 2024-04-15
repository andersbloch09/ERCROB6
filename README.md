# Maintenance Task
## How to launch the scripts
To launch UR_driver run the following but remember to change the name of your workspace. 
```bash
roslaunch ur_robot_driver ur3e_bringup.launch robot_ip:=192.168.1.102 robot_description_file:=$HOME/ws_ur/src/ur3e_moveit_config/launch/load_ur3e.launch
```
```bash
roslaunch ur3e_moveit_config moveit_planning_execution.launch 
```
```bash 
rosrun newur urmove.py 
```
If for testing without robot in pure Rviz use the following commands: 
```bash 
roslaunch ur3e_moveit_config demo.launch
```
```bash 
rosrun newur urmove.py 
```
The demo and the execution.launch will be using the same settings for Rviz. 