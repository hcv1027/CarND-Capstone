import rospy
from pid import PID
from yaw_controller import YawController


GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, *args, **kwargs):
        # TODO: Implement
        # Initialize pid-controller
        kp = 0.171
        ki = 0.000015
        kd = 1.0305
        self.pid = PID(kp, ki, kd, mn=0.0, mx=1.0)

        # Initialize yaw-controller
        try:
            wheel_base = kwargs['wheel_base']
            steer_ratio = kwargs['steer_ratio']
            min_speed = kwargs['min_speed']
            max_lat_accel = kwargs['max_lat_accel']
            max_steer_angle = kwargs['max_steer_angle']
        except:
            rospy.logerr("Initialize yaw-controller fail!")
            self.yaw_controller = None
        else:
            self.yaw_controller = YawController(
                wheel_base, steer_ratio, min_speed, max_lat_accel, max_steer_angle)

    def control(self, *args, **kwargs):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        # Get throttle
        try:
            vel_error = kwargs['vel_error']
            sample_time = kwargs['sample_time']
        except:
            rospy.logerr("PID controller parameters are not provided!")
            throttle = 0.0
        else:
            throttle = self.pid.step(vel_error, sample_time)
            # rospy.loginfo("throttle: %f, error: %f, sample_time: %f",
            #               throttle, vel_error, sample_time)

        # Get steering angle
        try:
            linear_velocity = kwargs['linear_velocity']
            angular_velocity = kwargs['angular_velocity']
            current_velocity = kwargs['current_velocity']
        except:
            rospy.logerr("yaw-controller parameters are not provided!")
            steer = 0.0
        else:
            steer = self.yaw_controller.get_steering(
                linear_velocity, angular_velocity, current_velocity)
            # rospy.loginfo("steer: %f", steer)

        # try:
        #     dbw_enable = kwargs['dbw_enable']
        # except:
        #     rospy.logerr("No key: %s", 'dbw_enable')
        #     dbw_enable = False

        # Reset PID-controller when dbw_enable is False
        return throttle, 0., steer

    def reset(self):
        # rospy.loginfo("Reset pid")
        self.pid.reset()
