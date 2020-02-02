#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Int32
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
from scipy.interpolate import splev, splrep, CubicSpline
import numpy as np
import tf
import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200  # Number of waypoints we will publish. You can change this number
MAX_DECEL = 9.5


def derivative(coeffs):
    derivative_coeffs = []
    for i in range(1, len(coeffs)):
        derivative_coeffs.append(i * coeffs[i])
    return derivative_coeffs


def poly_eval(x, coeffs):
    result = 0.0
    t = 1.0
    for i in range(len(coeffs)):
        result += coeffs[i] * t
        t *= x
    return result


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher(
            'final_waypoints', Lane, queue_size=1)
        self.stopline_pub = rospy.Publisher(
            'stop_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoints_tree = None
        self.curr_pose = None
        self.curr_twist = []
        self.curr_acc = 0.0
        self.stopline_wp_idx = -1
        self.prev_stopline_wp_idx = -1
        self.stop_buffer = 10
        # self.decel_limit = rospy.get_param('/dbw_node/decel_limit', -5)
        self.accel_limit = rospy.get_param('/dbw_node/accel_limit', 1.)
        self.decel_limit = -1.0
        self.accel_limit = 9.0
        self.max_vel = float(rospy.get_param(
            '/waypoint_loader/velocity', 40.0)) * 1000.0 / 3600.0
        self.hz = 30
        self.delta_time = 1 / self.hz
        self.prev_final_waypoints = []

        # start_x = [0.0, 11.1111, 0.0]
        # end_x = [29.0, 0.0, 0.0]
        # jmt = self.get_jmt_params(start_x, end_x, 5.0)
        # rospy.loginfo("jmt: %f, %f, %f, %f, %f, %f",
        #               jmt[0], jmt[1], jmt[2], jmt[3], jmt[4], jmt[5])

        self.loop()

    def loop(self):
        rate = rospy.Rate(self.hz)
        while not rospy.is_shutdown():
            if self.curr_pose is not None and self.waypoints_tree is not None and len(self.curr_twist) > 0:
                # Get closest waypoint
                x = self.curr_pose.pose.position.x
                y = self.curr_pose.pose.position.y
                closest_wp_idx = self.get_closest_waypoint_id(
                    x, y, self.waypoints_2d, self.waypoints_tree)
                self.publish_waypoints(closest_wp_idx)
            rate.sleep()

    def get_closest_waypoint_id(self, x, y, waypoints_2d, waypoints_tree):
        # x = self.curr_pose.pose.position.x
        # y = self.curr_pose.pose.position.y
        if waypoints_tree is None:
            rospy.logerr("waypoints_tree is None")
        closest_idx = waypoints_tree.query([x, y], 1)[1]

        # Check if closest waypoint is ahead or behind vehicle
        closest_coord = waypoints_2d[closest_idx]
        prev_coord = waypoints_2d[closest_idx - 1]
        if closest_idx == 0:
            next_coord = waypoints_2d[closest_idx + 1]
            temp_vect = [next_coord[0] - closest_coord[0],
                         next_coord[1] - closest_coord[1]]
            prev_coord = [closest_coord[0] - temp_vect[0],
                          closest_coord[1] - temp_vect[1]]

        # Equation for hyperplan through closest coordinate
        closest_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        curr_vect = np.array([x, y])

        val = np.dot(closest_vect - prev_vect, curr_vect - closest_vect)
        if val > 0:
            closest_idx = (closest_idx + 1) % len(waypoints_2d)
        return closest_idx

    def publish_waypoints(self, closest_wp_idx):
        # final_lane = self.generate_lane(closest_wp_idx)
        # self.final_waypoints_pub.publish(final_lane)

        # Using jmt to generate final_waypoints
        if self.stopline_wp_idx >= 0:
            buffer_dist = self.distance(self.base_waypoints.waypoints,
                                        closest_wp_idx, self.stopline_wp_idx - self.stop_buffer)
            if self.curr_twist[-1].twist.linear.x < 0.1:
                waypoints = self.base_waypoints.waypoints[closest_wp_idx:closest_wp_idx+5]
                for waypoint in waypoints:
                    waypoint.twist.twist.linear.x = 0.0
                final_lane = Lane()
                final_lane.header = self.curr_pose.header
                final_lane.waypoints = waypoints
                self.prev_final_waypoints = []
            elif buffer_dist > 30.0:
                final_lane = Lane()
                final_lane.header = self.curr_pose.header
                final_lane.waypoints = self.base_waypoints.waypoints[
                    closest_wp_idx:closest_wp_idx+LOOKAHEAD_WPS]
                self.prev_final_waypoints = []
            else:
                # Decelerate to stop before stop_line_wp
                final_lane = self.generate_jmt_waypoints(
                    closest_wp_idx, self.stopline_wp_idx - self.stop_buffer, 0.0)
                if len(final_lane.waypoints) == 0:
                    final_lane.waypoints = self.base_waypoints.waypoints[
                        closest_wp_idx:closest_wp_idx+LOOKAHEAD_WPS]
                self.prev_final_waypoints = final_lane.waypoints
        else:
            # Try to follow default base_waypoints
            end_wp_idx = closest_wp_idx + LOOKAHEAD_WPS
            if end_wp_idx >= len(self.base_waypoints.waypoints):
                rospy.logerr("end_wp_idx %d out of bound! Max: %d",
                             end_wp_idx, len(self.base_waypoints.waypoints)-1)
                end_wp_idx = len(self.base_waypoints.waypoints) - 1
            end_vel = self.base_waypoints.waypoints[end_wp_idx].twist.twist.linear.x
            if self.curr_twist[-1].twist.linear.x < end_vel - 2.0:
                final_lane = self.generate_jmt_waypoints(
                    closest_wp_idx, end_wp_idx, end_vel)
                self.prev_final_waypoints = final_lane.waypoints
            else:
                final_lane = self.generate_lane(closest_wp_idx)
                self.prev_final_waypoints = []
        # rospy.loginfo("closest_wp_idx: %d", closest_wp_idx)
        # rospy.loginfo("final_lane size: %d", len(final_lane.waypoints))
        # rospy.loginfo("wp[0] vel: %f",
        #               final_lane.waypoints[0].twist.twist.linear.x)
        self.final_waypoints_pub.publish(final_lane)
        if self.stopline_wp_idx >= 0:
            lane = Lane()
            lane.header = final_lane.header
            stop_wp = self.base_waypoints.waypoints[self.stopline_wp_idx - self.stop_buffer]
            lane.waypoints.append(stop_wp)
            self.stopline_pub.publish(lane)

    def generate_jmt_waypoints(self, closest_wp_idx, end_wp_idx, end_vel):
        # rospy.loginfo("closest_wp_idx: %d, end_wp_idx: %d, curr_vel: %f, end_vel: %f",
        #               closest_wp_idx, end_wp_idx, self.curr_twist[-1].twist.linear.x, end_vel)
        # rospy.loginfo("start: (%f, %f), end: (%f, %f)",
        #               self.base_waypoints.waypoints[closest_wp_idx].pose.pose.position.x,
        #               self.base_waypoints.waypoints[closest_wp_idx].pose.pose.position.y,
        #               self.base_waypoints.waypoints[end_wp_idx].pose.pose.position.x,
        #               self.base_waypoints.waypoints[end_wp_idx].pose.pose.position.y)
        lane = Lane()
        lane.header = self.curr_pose.header
        lane.waypoints = []

        if end_wp_idx - closest_wp_idx < 2:
            # lane.waypoints = self.base_waypoints.waypoints[closest_wp_idx, end_wp_idx]
            lane.waypoints = []
            return lane

        # Compute jmt parameters
        prev_closest_idx = -1
        if len(self.prev_final_waypoints) > 0:
            waypoints_2d = [[waypoint.pose.pose.position.x,
                             waypoint.pose.pose.position.y] for waypoint in self.prev_final_waypoints]
            waypoints_tree = KDTree(waypoints_2d)
            curr_x = self.curr_pose.pose.position.x
            curr_y = self.curr_pose.pose.position.y
            prev_closest_idx = self.get_closest_waypoint_id(
                curr_x, curr_y, waypoints_2d, waypoints_tree)
            # min_dist = 1e10  # Just a big value
            # for i, wp in enumerate(self.prev_final_waypoints):
            #     dist = pow(wp.pose.pose.position.x - curr_x, 2) + \
            #         pow(wp.pose.pose.position.y - curr_y, 2)
            #     if dist < min_dist:
            #         prev_closest_idx = i
            #         min_dist = dist

        if prev_closest_idx >= 0 and abs(self.prev_final_waypoints[-1].twist.twist.linear.x - end_vel) < 0.5:
            lane.waypoints = self.prev_final_waypoints[prev_closest_idx:]
            # rospy.loginfo("prev_closest_idx: %d, vel: %f",
            #               prev_closest_idx, lane.waypoints[0].twist.twist.linear.x)
            if len(lane.waypoints) < LOOKAHEAD_WPS:
                extend_size = LOOKAHEAD_WPS - len(lane.waypoints)
                last_x = lane.waypoints[-1].pose.pose.position.x
                last_y = lane.waypoints[-1].pose.pose.position.y
                last_wp_idx = self.get_closest_waypoint_id(
                    last_x, last_y, self.waypoints_2d, self.waypoints_tree)
                extend_waypoints = self.base_waypoints.waypoints[last_wp_idx:last_wp_idx+extend_size]
                for wp in extend_waypoints:
                    wp.twist.twist.linear.x = end_vel
                    lane.waypoints.append(wp)
            # rospy.loginfo("prev_closest_idx: %d, size: %d",
            #               prev_closest_idx, len(lane.waypoints))
            return lane

        dist = self.distance(self.base_waypoints.waypoints,
                             closest_wp_idx, end_wp_idx)
        waypoints = []
        # if self.curr_twist[-1].twist.linear.x < end_vel and len(self.prev_final_waypoints) > 0:
        #     waypoints = self.prev_final_waypoints[prev_closest_idx:]
        #     # rospy.loginfo("self.prev_final_waypoints size: %d, %d, %d", len(self.prev_final_waypoints), prev_closest_idx, )
        #     prev_dist = self.distance(
        #         self.prev_final_waypoints, prev_closest_idx, len(self.prev_final_waypoints) - 1)
        #     dist -= prev_dist
        # if len(self.prev_final_waypoints) - prev_closest_idx >= 5:
        #     waypoints = self.prev_final_waypoints[prev_closest_idx: prev_closest_idx+5]
        #     prev_dist = self.distance(
        #         self.prev_final_waypoints, prev_closest_idx, prev_closest_idx+5)
        #     dist -= prev_dist
        start = [0.0, 0.0, 0.0]
        end = [dist, end_vel, 0.0]
        TIME_STEP = 0.1
        if len(waypoints) == 0:
            start[1] = self.curr_twist[-1].twist.linear.x
            # start[2] = self.curr_acc
        else:
            prev_size = len(waypoints)
            vel_0 = waypoints[0].twist.twist.linear.x
            vel_1 = waypoints[-1].twist.twist.linear.x
            # dist_0 = self.distance(waypoints, prev_size - 2, prev_size - 1)
            # dist_1 = self.distance(waypoints, prev_size - 3, prev_size - 2)
            # vel_0 = dist_0 / TIME_STEP
            # vel_1 = dist_1 / TIME_STEP
            # acc_0 = (vel_0 - vel_1) / TIME_STEP
            start[1] = vel_0
            # start[2] = (vel_1 - vel_0) / (TIME_STEP * prev_size)
        rospy.loginfo("jmt start: %f, %f, %f", start[0], start[1], start[2])
        rospy.loginfo("jmt end: %f, %f, %f", end[0], end[1], end[2])

        # jmt_candidate = []
        jmt_params = None
        max_duration = 40.0
        jmt_duration = 20.0
        for duration in np.arange(max_duration, 0.2, -0.1):
            jmt = self.get_jmt_params(start, end, duration)
            if self.check_jmt(jmt, duration):
                jmt_duration = duration
                jmt_params = jmt
                break
                # jmt_candidate.append()
        if jmt_params is not None:
            # rospy.loginfo("jmt: %f, %f, %f, %f, %f, %f",
            #               jmt_params[0], jmt_params[1], jmt_params[2], jmt_params[3], jmt_params[4], jmt_params[5])
            vel_params = derivative(jmt_params)
            sample_xy = []
            for i in range(closest_wp_idx, end_wp_idx):
                sample_xy.append([self.base_waypoints.waypoints[i].pose.pose.position.x,
                                  self.base_waypoints.waypoints[i].pose.pose.position.y])
            # Sort waypoint accroding to x coordinate, spline needs x to be sorted.
            sample_xy = np.array(sorted(sample_xy, key=lambda x: x[0]))
            try:
                cs = CubicSpline(sample_xy[:, 0], sample_xy[:, 1])
            except:
                rospy.logerr("CubicSpline fail:")
                for x in sample_xy[:, 0]:
                    print("x: %f", x)

            # sample_x = sample_xy[:, 0].tolist()
            # sample_y = sample_xy[:, 1].tolist()
            # total_x = sample_x[-1] - sample_x[0]
            start_x = waypoints[-1].pose.pose.position.x if len(
                waypoints) > 0 else self.base_waypoints.waypoints[closest_wp_idx].pose.pose.position.x
            end_x = self.base_waypoints.waypoints[end_wp_idx].pose.pose.position.x
            dist_x = end_x - start_x
            new_x = []
            new_vel = []
            # new_yaw = []
            prev_d = 0.0
            delta_time = 0.0
            dist_threshold = dist * 0.05
            for dt in np.arange(0.1, jmt_duration, TIME_STEP):
                d = poly_eval(dt, jmt_params)
                next_vel = poly_eval(dt, vel_params)
                delta_dist = (d - prev_d)
                delta_time += TIME_STEP
                # if True:
                if next_vel > 0.5 or delta_dist > 1.0:
                    dist_ratio = d / dist
                    next_x = start_x + dist_x * dist_ratio
                    # rospy.loginfo("dt: %f, dist_x: %f, dist_ratio: %f, delta_dist: %f, delta_time: %f, next_x: %f, next_vel: %f",
                    #               dt, dist_x, dist_ratio, delta_dist, delta_time, next_x, next_vel)
                    new_x.append(next_x)
                    new_vel.append(next_vel)
                    prev_d = d
                    delta_time = 0.0
                    if ((self.curr_twist[-1].twist.linear.x < end_vel) and (next_vel >= end_vel)):
                        # rospy.loginfo("break 1, curr_vel: %f, end_vel: %f, next_vel: %f", self.curr_twist[-1].twist.linear.x , end_vel, next_vel)
                        break
                    elif end_vel == 0.0 and abs(next_vel - end_vel) < 0.1:
                        # rospy.loginfo("break 2, end_vel: %f, next_vel: %f",end_vel, next_vel)
                        break
            new_x = np.array(new_x)
            new_y = cs(new_x)
            # Skip the angular velocity and yaw, because waypoint_follower doesn't use it.
            # It computes angular velocity accroding to the curvature and linear velocity.
            for i in range(len(new_vel)):
                waypoint = Waypoint()
                waypoint.pose.header = self.curr_pose.header
                waypoint.pose.pose.position.x = new_x[i]
                waypoint.pose.pose.position.y = new_y[i]
                waypoint.pose.pose.position.z = 0.0
                # Transform from yaw to quaternion
                # q = tf.transformations.quaternion_from_euler(0., 0., yaw)
                # waypoint.pose.pose.orientation = Quaternion(*q)
                waypoint.twist.header = self.curr_pose.header
                waypoint.twist.twist.linear.x = new_vel[i]
                # if end_vel == 0.0 and new_vel[i] < 0.05:
                #     rospy.loginfo("i: %d, xy: (%f, %f), vel: %f",
                #                   i, new_x[i], new_y[i], new_vel[i])
                waypoints.append(waypoint)
            if len(waypoints) < LOOKAHEAD_WPS:
                extend_size = LOOKAHEAD_WPS - len(waypoints)
                last_x = waypoints[-1].pose.pose.position.x
                last_y = waypoints[-1].pose.pose.position.y
                last_wp_idx = self.get_closest_waypoint_id(
                    last_x, last_y, self.waypoints_2d, self.waypoints_tree)
                extend_waypoints = self.base_waypoints.waypoints[last_wp_idx:last_wp_idx+extend_size]
                for wp in extend_waypoints:
                    wp.twist.twist.linear.x = end_vel
                    waypoints.append(wp)
            # rospy.loginfo("len(new_vel): %d", len(new_vel))
            # if end_vel == 0.0:
            #     for i in range(len(waypoints)):
            #         wp = waypoints[i]
            #         rospy.loginfo("i: %d, xy: (%f, %f), vel: %f",
            #                       i, wp.pose.pose.position.x, wp.pose.pose.position.y, wp.twist.twist.linear.x)
            # if len(waypoints) > 0:
            #     rospy.loginfo("Final start: (%f, %f), end: (%f, %f)",
            #                   waypoints[0].pose.pose.position.x,
            #                   waypoints[0].pose.pose.position.y,
            #                   waypoints[-1].pose.pose.position.x,
            #                   waypoints[-1].pose.pose.position.y,)
        else:
            rospy.logwarn("Fail to find jmt parameters!")

        lane.waypoints = waypoints
        return lane

    def get_jmt_params(self, start, end, duration):
        t1 = duration
        t2 = math.pow(t1, 2)
        t3 = math.pow(t1, 3)
        t4 = math.pow(t1, 4)
        t5 = math.pow(t1, 5)
        a = np.array([[t3, t4, t5], [3 * t2, 4 * t3, 5 * t4],
                      [6 * t1, 12 * t2, 20 * t3]])
        a_inv = np.linalg.inv(a)

        b = np.array([0.0, 0.0, 0.0])
        b[0] = end[0] - (start[0] + start[1] * t1 + .5 * start[2] * t1 * t1)
        b[1] = end[1] - (start[1] + start[2] * t1)
        b[2] = end[2] - start[2]

        c = np.dot(a_inv, b)
        return np.array([start[0], start[1], 0.5 * start[2], c[0], c[1], c[2]])

    def check_jmt(self, jmt_params, duration):
        vel_params = derivative(jmt_params)
        acc_params = derivative(vel_params)
        jerk_params = derivative(acc_params)
        MAX_VEL = self.max_vel + 5.0
        # rospy.loginfo("check_jmt duration: %f", duration)
        result = True
        # jmt_log = ""
        for t in np.arange(0.1, duration, 0.1):
            dist = poly_eval(t, jmt_params)
            vel = poly_eval(t, vel_params)
            acc = abs(poly_eval(t, acc_params))
            jerk = abs(poly_eval(t, jerk_params))
            # jmt_log += "t: {:f}, dist: {:f}, vel: {:f}, acc: {:f}, jerk: {:f}\n".format(
            #     t, dist, vel, acc, jerk)
            if vel < 0.0 or acc < 0.1 or acc > self.accel_limit or jerk > 9.9:
                result = False
        if result:
            # rospy.loginfo("check_jmt duration: %f pass\n%s", duration, jmt_log)
            rospy.loginfo("check_jmt duration: %f pass", duration)
        # else:
        #     rospy.loginfo("check_jmt duration: %f fail", duration)
        return result

    def test_jmt_params(self):
        # Test case 1
        start1 = [0, 10, 0]
        end1 = [10, 10, 0]
        duration1 = 1
        ans_1 = [0.0, 10.0, 0.0, 0.0, 0.0, 0.0]
        jmt_param1 = self.get_jmt_params(start1, end1, duration1)
        test_case_result_1 = True
        for i in range(6):
            if abs(jmt_param1[i] - ans_1[i]) > 0.01:
                print("Test case 1 fail")
                test_case_result_1 = False
                break
        if test_case_result_1 is True:
            print("Test case 1 pass")

        # Test case 2
        start2 = [0, 10, 0]
        end2 = [20, 15, 20]
        duration2 = 2
        ans_2 = [0.0, 10.0, 0.0, 0.0, -0.625, 0.3125]
        jmt_param2 = self.get_jmt_params(start2, end2, duration2)
        test_case_result_2 = True
        for i in range(6):
            if abs(jmt_param2[i] - ans_2[i]) > 0.01:
                print("Test case 2 fail")
                test_case_result_2 = False
                break
        if test_case_result_2 is True:
            print("Test case 2 pass")

        # Test case 3
        start3 = [5, 10, 2]
        end3 = [-30, -20, -4]
        duration3 = 5
        ans_3 = [5.0, 10.0, 1.0, -3.0, 0.64, -0.0432]
        jmt_param3 = self.get_jmt_params(start3, end3, duration3)
        test_case_result_3 = True
        for i in range(6):
            if abs(jmt_param3[i] - ans_3[i]) > 0.01:
                print("Test case 3 fail")
                test_case_result_3 = False
                break
        if test_case_result_3 is True:
            print("Test case 3 pass")

    def generate_lane(self, closest_wp_idx):
        lane = Lane()
        lane.header = self.curr_pose.header

        lookahead_wp_idx = closest_wp_idx + LOOKAHEAD_WPS
        end_wp_idx = lookahead_wp_idx
        if self.stopline_wp_idx >= 0:
            end_wp_idx = min(lookahead_wp_idx, self.stopline_wp_idx)
        waypoints = self.base_waypoints.waypoints[closest_wp_idx: end_wp_idx]
        if self.stopline_wp_idx == -1 or lookahead_wp_idx <= self.stopline_wp_idx:
            lane.waypoints = waypoints
        else:
            # rospy.loginfo("Stop before: %d", self.stopline_wp_idx)
            lane.waypoints = self.decelerate_waypoints(
                waypoints, closest_wp_idx)

        # end_wp_idx = closest_wp_idx + LOOKAHEAD_WPS
        # waypoints = self.base_waypoints.waypoints[closest_wp_idx: end_wp_idx]
        # if self.stopline_wp_idx == -1 or self.stopline_wp_idx >= end_wp_idx:
        #     lane.waypoints = waypoints
        # else:
        #     # rospy.loginfo("Stop before: %d", self.stopline_wp_idx)
        #     lane.waypoints = self.decelerate_waypoints(
        #         waypoints, closest_wp_idx)
        return lane

    def decelerate_waypoints(self, waypoints, closest_wp_idx):
        temp = []
        # Because closest_wp_idx is at the center of the vehicle,
        # we want the vehicle's head stop at stop line waypoint,
        # that is the reason for "-2"
        # stop_wp_idx = max(0, self.stopline_wp_idx - closest_wp_idx - self.stop_buffer)
        stop_wp_idx = max(0, len(waypoints) - self.stop_buffer)

        # if self.stopline_wp_idx >= 0 and self.base_waypoints is not None and self.curr_pose is not None:
        #     lane = Lane()
        #     lane.header = self.curr_pose.header
        #     waypoints = []
        #     new_waypoint = self.base_waypoints.waypoints[self.stopline_wp_idx]
        #     waypoints.append(new_waypoint)
        #     new_waypoint = self.base_waypoints.waypoints[stop_wp_idx]
        #     waypoints.append(new_waypoint)
        #     lane.waypoints = waypoints
        #     self.stopline_pub.publish(lane)
        lane = Lane()
        lane.header = self.curr_pose.header
        show_waypoints = []
        rospy.loginfo("Stop before: %d, curr_wp: %d, stop_wp_idx: %d",
                      self.stopline_wp_idx, closest_wp_idx, stop_wp_idx)

        # found_plan = False
        # speed = 0
        # decel = 0.1
        # while found_plan is not True:
        #     for wp_idx in range(stop_wp_idx, -1, -1):
        #         pass
        # wp_s = waypoints[0]
        # wp_e = waypoints[stop_wp_idx]
        # start_x = wp_s

        for wp_idx, waypoint in enumerate(waypoints):
            new_waypoint = Waypoint()
            new_waypoint.pose = waypoint.pose
            dist = self.distance(waypoints, wp_idx, stop_wp_idx)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            if wp_idx == 0 or (stop_wp_idx - 5 <= wp_idx and wp_idx <= stop_wp_idx):
                rospy.loginfo("wp_idx: %d, dist: %f, vel: %f",
                              wp_idx, dist, vel)
            if vel < 1.0:
                # rospy.loginfo("Set to 0")
                vel = 0.0
            new_waypoint.twist.twist.linear.x = min(
                vel, waypoint.twist.twist.linear.x)
            temp.append(new_waypoint)
            # if vel == 0.0:
            #     show_waypoints.append(waypoint)
        show_waypoints.append(
            self.base_waypoints.waypoints[self.stopline_wp_idx])
        lane.waypoints = show_waypoints
        # self.stopline_pub.publish(lane)
        return temp

    def pose_cb(self, msg):
        # TODO: Implement
        self.curr_pose = msg

    def velocity_cb(self, msg):
        if len(self.curr_twist) == 50:
            self.curr_twist = self.curr_twist[1:len(self.curr_twist)]
        self.curr_twist.append(msg)
        # rospy.loginfo("Curr vel: %f", self.curr_twist[-1].twist.linear.x)
        # Estimate current acceleration
        if len(self.curr_twist) >= 2:
            twist_0 = self.curr_twist[0]
            twist_1 = self.curr_twist[-1]
            duration = twist_1.header.stamp.to_sec() - twist_0.header.stamp.to_sec()
            vel_diff = twist_1.twist.linear.x - twist_0.twist.linear.x
            self.curr_acc = vel_diff / duration
            # rospy.loginfo("acc: %f, vel_diff: %f, duration: %f",
            #               self.curr_acc, vel_diff, duration)

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x,
                                  waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoints_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        self.prev_stopline_wp_idx = self.stopline_wp_idx
        self.stopline_wp_idx = msg.data
        # pass
        # if self.stopline_wp_idx >= 0:
        #     rospy.loginfo("Receive traffic wp: %d", self.stopline_wp_idx)

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    # Get the distance from wp1 to wp2, the order of wp1 and wp2 can't change
    def distance(self, waypoints, wp1, wp2):
        dist = 0
        def dl(a, b): return math.sqrt(
            (a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position,
                       waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
