import rospy
from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter


GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit, accel_limit,
                 wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle, hz):
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
        self.hz = hz
        self.last_time = rospy.get_time()
        self.last_vel = 0.0

        # Initialize pid-controller
        kp = 0.3
        ki = 0.1
        kd = 0.001
        # kp = 0.55
        # ki = 0.03696
        # kd = 0.002772
        self.throttle_controller = PID(kp, ki, kd, mn=-1.0, mx=1.0)

        # Initialize low pass filter
        ts = 0.02  # 1/self.hz ?
        tau = 2 * ts
        self.vel_lpf = LowPassFilter(tau, ts)

        # Initialize yaw-controller
        self.steer_controller = YawController(
            wheel_base=wheel_base,
            steer_ratio=steer_ratio,
            min_speed=0.1,  # 0.2
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
        # rospy.loginfo("curr_velocity: %f, target: %f",
        #               curr_velocity, target_twist_cmd.twist.linear.x)

        # Get throttle
        vel_error = target_twist_cmd.twist.linear.x - curr_twist_cmd.twist.linear.x
        throttle = self.throttle_controller.step(vel_error, sample_time)

        # Get steering angle
        linear_vel = target_twist_cmd.twist.linear.x
        angular_vel = target_twist_cmd.twist.angular.z
        steer = self.steer_controller.get_steering(
            linear_vel, angular_vel, curr_velocity)

        # Get brake
        brake = 0.0
        if linear_vel <= 1e-1 and curr_velocity <= 1e-1:
            throttle = 0.0
            steer = 0.0
            brake = 700
        elif throttle > 0.0:
            brake = 0.0
        else:
            decel = -throttle
            throttle = 0
            if decel < self.brake_deadband:
                decel = 0.
            brake = decel * self.wheel_radius * \
                (self.vehicle_mass + self.fuel_capacity * GAS_DENSITY)

        return throttle, brake, steer

    def reset(self):
        # rospy.loginfo("Reset controller")
        self.throttle_controller.reset()
        self.last_time = rospy.get_time()
