import rospy
from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter


GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit,
                 wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
        # TODO: Implement
        self.vehicle_mass = vehicle_mass
        self.fuel_capacity = fuel_capacity
        self.brake_deadband = brake_deadband
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius
        self.wheel_base = wheel_base
        self.steer_ratio = steer_ratio
        self.max_lat_accel = max_lat_accel
        self.max_steer_angle = max_steer_angle
        self.last_time = rospy.get_time()
        self.last_vel = 0.0
        # rospy.loginfo("Controller type(vehicle_mass): %s, %f",
        #               type(self.vehicle_mass), self.vehicle_mass)

        # Initialize pid-controller
        kp = 0.3
        ki = 0.1
        kd = 0.001
        self.throttle_controller = PID(kp, ki, kd, mn=-1.0, mx=1.0)

        # Initialize low pass filter
        tau = 0.5
        ts = 0.02
        self.vel_lpf = LowPassFilter(tau, ts)

        # Initialize yaw-controller
        self.steer_controller = YawController(
            wheel_base=wheel_base,
            steer_ratio=steer_ratio,
            min_speed=0.2,
            max_lat_accel=max_lat_accel,
            max_steer_angle=max_steer_angle)

    def control(self, dbw_enable, curr_twist_cmd, target_twist_cmd):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        if dbw_enable is not True:
            self.reset()
            return 0.0, 0.0, 0.0

        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time

        curr_velocity = self.vel_lpf.filt(curr_twist_cmd.twist.linear.x)

        # Get throttle
        vel_error = target_twist_cmd.twist.linear.x - curr_twist_cmd.twist.linear.x
        # print("{:f}, ".format(vel_error))
        throttle = self.throttle_controller.step(vel_error, sample_time)
        # rospy.loginfo("throttle: %f, error: %f, sample_time: %f",
        #               throttle, vel_error, sample_time)

        # Get steering angle
        linear_vel = target_twist_cmd.twist.linear.x
        angular_vel = target_twist_cmd.twist.angular.z
        steer = self.steer_controller.get_steering(
            linear_vel, angular_vel, curr_velocity)
        # rospy.loginfo("control linear_vel: %f, angular_vel: %f, curr_velocity: %f",
        #               linear_vel, angular_vel, curr_velocity)
        # rospy.loginfo("steer: %f", steer)

        # Get brake
        brake = 0.0
        if linear_vel == 0 and curr_velocity < 1e-4:
            throttle = 0.0
            brake = 700
        # elif vel_error < 0.0:  # elif throttle < 0.1 and vel_error < 0.0:
        elif throttle < 0.1 and vel_error < 0.0:
            throttle = 0.0
            # decel = max(abs(vel_error), abs(self.decel_limit))
            decel = max(vel_error, self.decel_limit)
            # rospy.loginfo("Controller type(vehicle_mass): %s, %f",
            #               type(self.vehicle_mass), self.vehicle_mass)
            # rospy.loginfo("self.wheel_radius: %f", self.wheel_radius)
            # rospy.loginfo("abs(decel): %f", abs(decel))
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius
        return throttle, brake, steer

    def reset(self):
        # rospy.loginfo("Reset controller")
        self.throttle_controller.reset()
