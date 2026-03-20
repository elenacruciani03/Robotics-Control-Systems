# TurtleBot3 Trajectory Tracking
This project implements a control law in Python to make a TurtleBot3 Burger follow a figure-eight trajectory.

### Implementation Details
- Environment: Tested in ROS1 Noetic and Gazebo.
- Logic: The script calculates the error between the robot's current pose and the desired trajectory, generating velocity commands.
- Challenges: Tuning the PD/Control gains to minimize the tracking error while handling the physical constraints of the robot.
