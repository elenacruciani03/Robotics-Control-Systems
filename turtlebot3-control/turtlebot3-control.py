#!/usr/bin/env python3

import rospy
import numpy as np
import math
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point, PointStamped
from tf import transformations


last_rho = 0.0

def control(x_ref_b, y_ref_b):
    global current_pose, b, last_rho, ts

    
    x = current_pose[0] + b * np.cos(current_pose[2])
    y = current_pose[1] + b * np.sin(current_pose[2])
    theta = current_pose[2]

    # Position error 
    dx = x_ref_b - x
    dy = y_ref_b - y

    # Polar coordinates
    rho = np.sqrt(dx**2 + dy**2)
    alpha = np.arctan2(dy, dx) - theta
    alpha = np.arctan2(np.sin(alpha), np.cos(alpha))

    # Gains 
    k_rho = 3.5
    k_alpha = 6.5
    k_rho_d = 2.0

    # Derivative of rho 
    if ts > 0:
        drho = (rho - last_rho) / ts
    else:
        drho = 0.0

    last_rho = rho

    # PD Control
    v_raw = k_rho * rho + k_rho_d * drho
    omega_raw = k_alpha * alpha

    v = np.clip(v_raw, -MAX_LIN_VEL, MAX_LIN_VEL)
    omega = np.clip(omega_raw, -MAX_ANG_VEL, MAX_ANG_VEL)

    return v, omega

def stop_node():
    global cmd_vel_publisher, rate
    stop_msg = Twist()
    for step in range(10):
        cmd_vel_publisher.publish(stop_msg)
        rate.sleep()


def pose_updater(msg):
    global current_pose
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    orientation = np.array([msg.pose.pose.orientation.x,
                            msg.pose.pose.orientation.y,
                            msg.pose.pose.orientation.z,
                            msg.pose.pose.orientation.w])
    _, _, theta = transformations.euler_from_quaternion(orientation)
    current_pose = np.array([x, y, theta])


def compute_reference_trajectory():
    global hz, trajectory_type, rotation_matrix, init_x, init_y, init_theta, set_point_list, set_point_dot_list
    if trajectory_type == "run":
        v = 0.15
        T_ciclo = 30
        f_ciclo = 1 / T_ciclo
        w_ciclo = 2 * np.pi * f_ciclo
        r = v / w_ciclo
        T = T_ciclo
        w_x = 2 * w_ciclo
        w_y = w_ciclo

        start_time = 0.0
        stop_time = T
        step = 1 / hz
        num_samples = math.ceil((stop_time - start_time) / step)
        time_steps = np.linspace(start=0., stop=T, num=num_samples)

        circular_x = r * np.sin(w_x * time_steps)
        circular_y = r - r * np.cos(w_y * time_steps)
        circular_x_dot = w_x * r * np.cos(w_x * time_steps)
        circular_y_dot = w_y * r * np.sin(w_y * time_steps)

        for i in range(len(circular_x)):
            point = np.array([circular_x[i], circular_y[i]]).T
            point_dot = np.array([circular_x_dot[i], circular_y_dot[i]]).T
            point_w = np.dot(rotation_matrix, point).reshape((2,)) + np.array([init_x, init_y])
            set_point_list.append(point_w)
            set_point_dot_list.append(point_dot)
    else:
        print("Task completed")
        exit(-1)

    return set_point_list, set_point_dot_list


def rmse_calculation(ref_trj, xb_trj, start_index, end_index):
    ref_section = ref_trj[start_index:end_index]
    xb_section = xb_trj[start_index:end_index]
    error = np.array([(ref.x - xb.x)**2 + (ref.y - xb.y)**2
                      for ref, xb in zip(ref_section, xb_section)])
    rmse = np.sqrt(np.mean(error))
    return rmse


def distance_calculation(xb_trj):
    start = xb_trj[0]
    end = xb_trj[-1]
    error = np.array([(start.x - end.x)**2 + ((start.y - end.y)**2)])
    distance = np.sqrt(np.mean(error))
    return distance


if __name__ == "__main__":
    rospy.init_node("simple_controller", anonymous=False)
    hz = 10
    ts = 1/hz
    rate = rospy.Rate(hz)

    # Robot parameters
    b = 0.1
    MAX_LIN_VEL = 0.22  
    MAX_ANG_VEL = 2.84

    current_pose = np.zeros((3,))
    pose_subscriber = rospy.Subscriber("/odom", Odometry, pose_updater, queue_size=10)
    cmd_vel_publisher = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

    ref_signal_publisher = rospy.Publisher("/ref_signal", PointStamped, queue_size=10)
    xb_signal_publisher = rospy.Publisher("/xb_signal", PointStamped, queue_size=10)

    while not rospy.is_shutdown():
        trajectory_type = input("Experiment execution or end [run, fine]: ")

        init_msg = rospy.wait_for_message("/odom", Odometry)
        init_orientation = np.array([init_msg.pose.pose.orientation.x,
                                     init_msg.pose.pose.orientation.y,
                                     init_msg.pose.pose.orientation.z,
                                     init_msg.pose.pose.orientation.w])
        _, _, init_theta = transformations.euler_from_quaternion(init_orientation)
        init_x = init_msg.pose.pose.position.x + b * np.cos(init_theta)
        init_y = init_msg.pose.pose.position.y + b * np.sin(init_theta)
        rotation_matrix = np.array([[np.cos(init_theta), -np.sin(init_theta)],
                                    [np.sin(init_theta), np.cos(init_theta)]])
        set_point_list = []
        set_point_dot_list = []
        compute_reference_trajectory()

        rospy.on_shutdown(stop_node)

        ref_trj = []
        xb_trj = []

        for set_point in set_point_list:
            cmd_vel_msg = Twist()
            v, omega = control(set_point[0], set_point[1])
            cmd_vel_msg.linear.x = v
            cmd_vel_msg.angular.z = omega
            cmd_vel_publisher.publish(cmd_vel_msg)

            current_time = rospy.Time.now()

            current_ref_msg = PointStamped()
            current_ref_msg.header.stamp = current_time
            current_ref = Point()
            current_ref.x = set_point[0]
            current_ref.y = set_point[1]
            current_ref_msg.point = current_ref
            ref_signal_publisher.publish(current_ref_msg)

            current_xb_msg = PointStamped()
            current_xb_msg.header.stamp = current_time
            current_xb = Point()
            current_xb.x = current_pose[0] + b * np.cos(current_pose[2])
            current_xb.y = current_pose[1] + b * np.sin(current_pose[2])
            current_xb_msg.point = current_xb
            xb_signal_publisher.publish(current_xb_msg)

            ref_trj.append(current_ref)
            xb_trj.append(current_xb)

            rate.sleep()

        stop_node()

        total_samples = len(ref_trj)
        quarter = total_samples // 4

        rmse_first_quarter = rmse_calculation(ref_trj, xb_trj, 0, quarter)
        rmse_last_quarter = rmse_calculation(ref_trj, xb_trj, total_samples - quarter, total_samples)

        print(f'RMSE initial segment (0-1/4) [mm]: {1000.0 * rmse_first_quarter:.3f}')
        print(f'RMSE final segment (3/4-1) [mm]: {1000.0 * rmse_last_quarter:.3f}')
        print(f'Ratio between RMSE (final/initial): {rmse_last_quarter / rmse_first_quarter:.3f}')
        final_error = distance_calculation(xb_trj)
        print(f'Final error [mm]: {1000.0 * final_error:.3f}')
