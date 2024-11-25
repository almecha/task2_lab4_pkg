import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/nukharetrd/ros2_ws/src/task2_lab4_pkg/install/task2_lab4_pkg'
