import math
from math import cos, sin, sqrt
import numpy as np
import sympy
from sympy import symbols, Matrix

def sample_velocity_motion_model(x, u, a, dt):
    """ Sample velocity motion model.
    Arguments:
    x -- pose of the robot before moving [x, y, theta]
    u -- velocity reading obtained from the robot [v, w]
    sigma -- noise parameters of the motion model [a1, a2, a3, a4, a5, a6] or [std_dev_v, std_dev_w]
    dt -- time interval of prediction
    """
    sigma = np.ones((3))
    sigma[:2] = a 
    sigma[2] = sigma[1] * 0.5      #1D array with 5 elements

    # sample noisy velocity commands to consider actuaction errors and unmodeled dynamics
    v_hat = u[0] + np.random.normal(0, sigma[0])
    w_hat = u[1] + np.random.normal(0, sigma[1])
    gamma_hat = np.random.normal(0, sigma[2])

    # compute the new pose of the robot according to the velocity motion model
    r = v_hat/w_hat
    x_prime = x[0] - r*sin(x[2]) + r*sin(x[2]+w_hat*dt)
    y_prime = x[1] + r*cos(x[2]) - r*cos(x[2]+w_hat*dt)
    theta_prime = x[2] + w_hat*dt + gamma_hat*dt

    return np.array([x_prime, y_prime, theta_prime,v_hat,w_hat])

def squeeze_sympy_out(func):
    # inner function
    def squeeze_out(*args):
        out = func(*args).squeeze()
        return out
    return squeeze_out

def velocity_mm_simpy():
    """
    Define Jacobian Gt w.r.t state x=[x, y, theta] and Vt w.r.t command u=[v, w]
    """
    x, y, theta, v, w, v_hat, w_hat, dt = symbols('x y theta v w v_hat w_hat dt')
    R = v_hat / w_hat
    beta = theta + w * dt
    gux = Matrix(
        [
            [x - R * sympy.sin(theta) + R * sympy.sin(beta)],
            [y + R * sympy.cos(theta) - R * sympy.cos(beta)],
            [beta],
            [v_hat],
            [w_hat],
        ]
    )

    eval_gux = squeeze_sympy_out(sympy.lambdify((x, y, theta, v, w, v_hat, w_hat, dt), gux, 'numpy'))

    Gt = gux.jacobian(Matrix([x, y, theta,v,w]))
    eval_Gt = squeeze_sympy_out(sympy.lambdify((x, y, theta, v, w,v_hat,w_hat, dt), Gt, "numpy"))

    Vt = gux.jacobian(Matrix([v_hat, w_hat]))
    eval_Vt = squeeze_sympy_out(sympy.lambdify((x, y, theta, v, w,v_hat,w_hat, dt), Vt, "numpy"))

    return eval_gux, eval_Gt, eval_Vt

def landmark_range_bearing_model(robot_pose, landmark, sigma):
    """""
    Sampling z from landmark model for range and bearing
    """""
    m_x, m_y = landmark[:]
    x, y, theta,v,w = robot_pose[:]

    r_ = math.dist([x, y], [m_x, m_y]) + np.random.normal(0., sigma[0])
    phi_ = math.atan2(m_y - y, m_x - x) - theta + np.random.normal(0., sigma[1])
    return np.array([r_, phi_])

def landmark_sm_simpy():
    x, y, theta,v,w, mx, my = symbols("x y theta v w m_x m_y")

    hx = Matrix(
        [
            [sympy.sqrt((mx - x) ** 2 + (my - y) ** 2)],
            [sympy.atan2(my - y, mx - x) - theta],
        ]
    )
    eval_hx = squeeze_sympy_out(sympy.lambdify((x, y, theta, mx, my), hx, "numpy"))
    
    Ht = hx.jacobian(Matrix([x, y, theta,v,w]))
    eval_Ht = squeeze_sympy_out(sympy.lambdify((x, y, theta,v, w, mx, my), Ht, "numpy"))

    return eval_hx, eval_Ht

def ht_odom(x, u, a):
    """ Sample odometry motion model.
    Arguments:
    x -- initial state of the robot before update [x,y,theta,v,w]
    u -- odometry reading obtained from the robot [v_hat, w_hat]
    a -- noise parameters of the motion model [std_trans, std_rot]
    """

    sigma = np.ones((2))
    sigma = a
    
    # velocity updates from /odom
    v_hat = u[0] +  np.random.normal(0, sigma[0])
    w_hat = u[1] + np.random.normal(0, sigma[1])

    return np.array([v_hat, w_hat])


def ht_odom_mm_simpy():
    """
    Define Jacobian Gt and Vt for the odometry motion model
    """
    v_hat, w_hat = symbols(r"\delta_{v_hat} \delta_{w_hat}")
    x, y, theta, v, w = symbols(r"x y \theta v w")
    gux_odom = Matrix([
        [v_hat],
        [w_hat],
    ])
    eval_gux_odom = squeeze_sympy_out(sympy.lambdify((x, y, theta, v, w,v_hat,w_hat), gux_odom, "numpy"))
    Gt_odom = gux_odom.jacobian(Matrix([x, y, theta,v,w]))
    args = (x, y, theta, v, w, v_hat, w_hat)
    eval_gux_odom = squeeze_sympy_out(sympy.lambdify(args, gux_odom, "numpy"))
    eval_Ht_odom = squeeze_sympy_out(sympy.lambdify(args, Gt_odom, "numpy"))

    return eval_gux_odom, eval_Ht_odom

def ht_imu(x, u ,a):
    """ Sample odometry motion model.
    Arguments:
    x -- initial state of the robot before update [x,y,theta,v,w]
    u -- IMU reading obtained from the robot [v_hat, w_hat]
    a -- noise parameters of the motion model [std_trans, std_rot]
    """
    sigma = a
    
    # velocity updates from /odom
    w_hat = u + np.random.normal(0, sigma)

    return np.array([w_hat])

def ht_imu_mm_simpy():
    """
    Define Jacobian Gt and Vt for the odometry motion model
    """
    w_hat = symbols(r"\delta_{w_hat}")
    x, y, theta, v, w = symbols(r"x y \theta v w")
    gux_imu = Matrix([
        [w_hat],
    ])
    Gt_imu = gux_imu.jacobian(Matrix([x, y, theta,v,w]))
    Vt_imu = gux_imu.jacobian(Matrix([w_hat]))
    Gt_imu = Matrix([[entry] for entry in Gt_imu.row(0)])

    args = (x, y, theta, v, w, w_hat)
    eval_gux_imu = squeeze_sympy_out(sympy.lambdify(args, gux_imu, "numpy"))
    eval_Ht_imu = squeeze_sympy_out(sympy.lambdify(args, Gt_imu, "numpy"))

    return eval_gux_imu, eval_Ht_imu