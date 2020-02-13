#!/usr/bin/env python

import os
import csv

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport


'''
You can use this file to test your DBW code against a bag recorded with a reference implementation.
The bag can be found at https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/files/reference.bag.zip

To use the downloaded bag file, rename it to 'dbw_test.rosbag.bag' and place it in the CarND-Capstone/data folder.
Then with roscore running, you can then use roslaunch with the dbw_test.launch file found in 
<project_repo>/ros/src/twist_controller/launch.

This file will produce 3 csv files which you can process to figure out how your DBW node is
performing on various commands.

`/actual/*` are commands from the recorded bag while `/vehicle/*` are the output of your node.

'''


class DBWTestNode(object):
    def __init__(self):
        rospy.init_node('dbw_test_node')

        rospy.Subscriber('/vehicle/steering_cmd', SteeringCmd, self.steer_cb)
        rospy.Subscriber('/vehicle/throttle_cmd',
                         ThrottleCmd, self.throttle_cb)
        rospy.Subscriber('/vehicle/brake_cmd', BrakeCmd, self.brake_cb)

        rospy.Subscriber('/actual/steering_cmd',
                         SteeringCmd, self.actual_steer_cb)
        rospy.Subscriber('/actual/throttle_cmd', ThrottleCmd,
                         self.actual_throttle_cb)
        rospy.Subscriber('/actual/brake_cmd', BrakeCmd, self.actual_brake_cb)

        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)

        self.steer = self.throttle = self.brake = None

        self.steer_data = []
        self.throttle_data = []
        self.brake_data = []

        self.dbw_enabled = False

        base_path = os.path.dirname(os.path.abspath(__file__))
        self.steerfile = os.path.join(base_path, 'steers.csv')
        self.throttlefile = os.path.join(base_path, 'throttles.csv')
        self.brakefile = os.path.join(base_path, 'brakes.csv')

        self.loop()

    def loop(self):
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            rate.sleep()
        fieldnames = ['actual', 'proposed']

        throttle_err = 0.0
        for data in self.throttle_data:
            throttle_err += abs(data['actual'] - data['proposed'])
        rospy.loginfo("throttle_err: %f", throttle_err)

        steer_err = 0.0
        for data in self.steer_data:
            steer_err += abs(data['actual'] - data['proposed'])
        rospy.loginfo("steer_err: %f", steer_err)

        brake_err = 0.0
        for data in self.brake_data:
            brake_err += abs(data['actual'] - data['proposed'])
        rospy.loginfo("brake_err: %f", brake_err)

        with open(self.steerfile, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.steer_data)

        with open(self.throttlefile, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.throttle_data)

        with open(self.brakefile, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.brake_data)

    def dbw_enabled_cb(self, msg):
        self.dbw_enabled = msg.data
        # rospy.loginfo("dbw_enabled: %d", self.dbw_enabled)

    def steer_cb(self, msg):
        self.steer = msg.steering_wheel_angle_cmd
        # rospy.loginfo("steer: {}".format(self.steer))

    def throttle_cb(self, msg):
        self.throttle = msg.pedal_cmd
        # rospy.loginfo("throttle: {}".format(self.throttle))

    def brake_cb(self, msg):
        self.brake = msg.pedal_cmd
        # rospy.loginfo("brake: {}".format(self.brake))

    def actual_steer_cb(self, msg):
        if self.dbw_enabled and self.steer is not None:
            # rospy.loginfo("actual_steer: {}".format(
            #     msg.steering_wheel_angle_cmd))
            self.steer_data.append({'actual': msg.steering_wheel_angle_cmd,
                                    'proposed': self.steer})
            self.steer = None

    def actual_throttle_cb(self, msg):
        if self.dbw_enabled and self.throttle is not None:
            # rospy.loginfo("actual_throttle: {}".format(msg.pedal_cmd))
            self.throttle_data.append({'actual': msg.pedal_cmd,
                                       'proposed': self.throttle})
            self.throttle = None

    def actual_brake_cb(self, msg):
        if self.dbw_enabled and self.brake is not None:
            # rospy.loginfo("actual_brake: {}".format(msg.pedal_cmd))
            self.brake_data.append({'actual': msg.pedal_cmd,
                                    'proposed': self.brake})
            self.brake = None


if __name__ == '__main__':
    DBWTestNode()
