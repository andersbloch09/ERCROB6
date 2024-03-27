# Maintenance Task
## How to launch the scripts
To launch UR_driver do the following
```bash
roslaunch ur_robot_driver ur3e_bringup.launch robot_ip:=192.168.1.102
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