#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math
import numpy as np
import sympy
from nav_msgs.msg import Odometry
from landmark_msgs.msg import LandmarkArray
from geometry_msgs.msg import Twist
from task2_lab4_pkg.utils.utils import residual
from task2_lab4_pkg.utils.ekf import RobotEKF
from sensor_msgs.msg import Imu

from task2_lab4_pkg.utils.probabilistic_models_task2 import sample_velocity_motion_model, velocity_mm_simpy, landmark_range_bearing_model, landmark_sm_simpy
from task2_lab4_pkg.utils.probabilistic_models_task2 import ht_odom, ht_odom_mm_simpy
from task2_lab4_pkg.utils.probabilistic_models_task2 import ht_imu, ht_imu_mm_simpy
from tf_transformations import quaternion_from_euler

class EKFLocalization(Node):
    def __init__(self):
        super().__init__('EKF_Localization')
        self.get_logger().info("EKF node initialized")
        #subscription to /odom to update the state of the filter
        self.cmdvel_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_update_callback, 10)
        #subscription to /cmd_vel to get the velocity from twist
        self.odom_sub = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        #subscription to /odom to update the state of the filter
        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_update_callback, 10)
        self.imu_sub
        #initial velocity
        self.v = 0.0
        self.w = 1e-9
        #subscription to /landmarks to get the range and bearing of the landmarks
        self.lmark_sub = self.create_subscription(LandmarkArray, '/landmarks', self.update, 10)
        #period of the filter
        self.ekf_dt = 1/20.0
        #timer for the prediction call
        self.timer = self.create_timer(self.ekf_dt, self.predict)
        #publisher to publish the result of the ekf
        self.ekf_pub = self.create_publisher(Odometry, '/ekf', 10)
        #message of which we publish the result
        self.ekf_msg = Odometry()
        #parameters needed to the EKF class:
        self.eval_gux = sample_velocity_motion_model
        _, self.eval_Gt, self.eval_Vt = velocity_mm_simpy()
        self.std_lin_vel = 0.05
        self.std_ang_vel = 0.02
        self.Mt = np.diag([self.std_lin_vel**2, self.std_ang_vel**2])
        self.ekf = RobotEKF(dim_x=5, #(x,y,theta)
                            dim_u=2, #(v,w=0)
                            eval_gux=self.eval_gux,
                            eval_Gt=self.eval_Gt,
                            eval_Vt=self.eval_Vt)
        self.ekf.mu = np.array([-2.0,-0.5,0.0, 0.0, 0.0]) #consider the origin as the initial position
        # print("SHAPE OF mu is", self.ekf.mu.shape)
        self.ekf.Sigma=np.diag([0.03,0.03,0.03,0.03,0.03])
        self.ekf.Mt = self.Mt

        #retrive landmarks position from yaml file:
        # Declare parameters with default values
        self.declare_parameter('landmarks.id',[11, 12, 13, 21, 22, 23, 31, 32, 33])
        self.declare_parameter('landmarks.x', [-1.1, -1.1, -1.1, 0.0, 0.0, 0.0, 1.1, 1.1, 1.1])
        self.declare_parameter('landmarks.y', [-1.1, 0.0, 1.1, -1.1, 0.0, 1.1, -1.1, 0.0, 1.1])
        self.declare_parameter('landmarks.z', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Retrieve parameters with default values or YAML overrides
        ids = self.get_parameter('landmarks.id').get_parameter_value().integer_array_value
        xs = self.get_parameter('landmarks.x').get_parameter_value().double_array_value
        ys = self.get_parameter('landmarks.y').get_parameter_value().double_array_value
        zs = self.get_parameter('landmarks.z').get_parameter_value().double_array_value

        # Combine into a list of dictionaries for each landmark
        self.landmarks = [
            {'id': int(ids[i]), 'x': xs[i], 'y': ys[i], 'z': zs[i]}
            for i in range(len(ids))
        ]

        #parameters needed for the update of the EKF
        self.eval_hx_landm = landmark_range_bearing_model
        _, self.eval_Ht_landm = landmark_sm_simpy()
        self.std_range = 0.01
        self.std_bearing = 0.01
        self.Q_landm = np.diag([self.std_range**2, self.std_bearing**2])
        self.sigma_z = np.array([self.std_range, self.std_bearing])

        #parameters needed for the update of the EKF from ODOM
        self.eval_hx_odom = ht_odom
        _, self.eval_Ht_odom = ht_odom_mm_simpy()
        self.std_v_hat = 0.05
        self.std_w_hat = 0.03
        self.Q_odom = np.diag([self.std_v_hat**2, self.std_w_hat**2])
        self.sigma_z_odom = np.array([self.std_v_hat, self.std_w_hat])

        #parameters needed for the update of the EKF from IMU
        self.eval_hx_imu = ht_imu
        _, self.eval_Ht_imu = ht_imu_mm_simpy()
        self.std_w_hat_imu = 0.03
        self.Q_imu = np.diag([self.std_w_hat_imu**2])
        self.sigma_z_imu = np.array([self.std_w_hat_imu])

    def cmd_vel_callback(self, msg : Twist):
        self.v = msg.linear.x
        self.w = msg.angular.z
        if self.w == 0:
            self.w = 1e-6

    def predict(self):
        u = np.array([self.v, self.w])
        sigma_u = np.array([self.std_lin_vel, self.std_ang_vel])
        self.ekf.predict(u=u, sigma_u=sigma_u,g_extra_args=(self.ekf_dt,))
        print("ekf.mu after odom predict is : ", self.ekf.mu)

    def imu_update_callback(self, msg : Imu ):
        z = msg.angular_velocity.z
        if z == 0:
            z = 1e-6
        z_mat = np.array([z])
        self.ekf.update(z , eval_hx = self.eval_hx_imu, eval_Ht = self.eval_Ht_imu, Qt = self.Q_imu,
                        Ht_args = (*self.ekf.mu,*z_mat), hx_args = (self.ekf.mu, z, self.sigma_z_imu),
                        residual = residual, angle_idx = -1)
        print("ekf.mu after imu update is : ", self.ekf.mu)

    def cmd_update_callback(self, msg : Twist):
        v_hat = msg.linear.x
        w_hat = msg.angular.z
        if w_hat == 0:
            w_hat = 1e-6
        z = np.array([v_hat, w_hat])
        self.ekf.update(z , eval_hx = self.eval_hx_odom, eval_Ht = self.eval_Ht_odom, Qt = self.Q_odom,
                        Ht_args = (*self.ekf.mu,*z), hx_args = (self.ekf.mu, z, self.sigma_z_odom),
                        residual = residual, angle_idx = -1)
        print("ekf.mu after odom update is : ", self.ekf.mu)

    def update(self, msg : LandmarkArray):
        #extract the necessary parameters from the message
        for landmark in msg.landmarks:
            landmark_id = landmark.id
            landmark_range = landmark.range
            landmark_bearing = landmark.bearing
            # Find the x, y coordinates for this landmark id
            if landmark_id in self.landmarks:
                x = self.landmarks[landmark_id]['x']
                y = self.landmarks[landmark_id]['y']
                # Prepare the lists: [range, bearing] and [x, y]
                z = [landmark_range, landmark_bearing]
                lmark = [x, y]
                self.ekf.update(z, eval_hx=self.eval_hx_landm, eval_Ht=self.eval_Ht_landm, Qt=self.Q_landm, 
                                Ht_args=(*self.ekf.mu, *lmark), 
                                hx_args=(self.ekf.mu, lmark, self.sigma_z),
                                residual=residual, 
                                angle_idx=-1)
        print("KALMAN GAIN FROM LMARK UPDATE IS: ", self.ekf.K)
        print("ekf.mu after landmark update is : ", self.ekf.mu)
        self.ekf_msg.header.stamp = self.get_clock().now().to_msg()
        #extract the position from the mu
        self.ekf_msg.pose.pose.position.x = self.ekf.mu[0]
        self.ekf_msg.pose.pose.position.y = self.ekf.mu[1]
        self.ekf_msg.pose.pose.position.z = 0.0
        #modify the theta angle as a quaternion in oder to publish the result on a Odometry message
        quat = quaternion_from_euler(0.0, 0.0, self.ekf.mu[2])
        self.ekf_msg.pose.pose.orientation.x = quat[0]
        self.ekf_msg.pose.pose.orientation.y = quat[1]
        self.ekf_msg.pose.pose.orientation.z = quat[2]
        self.ekf_msg.pose.pose.orientation.w = quat[3]
        self.ekf_msg.twist.twist.linear.x = self.ekf.mu[3]
        self.ekf_msg.twist.twist.linear.y = 0.0
        self.ekf_msg.twist.twist.linear.z = 0.0
        self.ekf_msg.twist.twist.angular.z = self.ekf.mu[4]
        self.ekf_msg.twist.twist.angular.x = 0.0
        self.ekf_msg.twist.twist.angular.y = 0.0
        self.ekf_pub.publish(self.ekf_msg)



def main(args=None):
    # Initialize the ROS2 Python client library
    rclpy.init(args=args)

    # Create the controller node
    controller = EKFLocalization()

    # Spin the node so it keeps running
    rclpy.spin(controller)

    # Clean up and shut down the node
    controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

        
        

            
            


