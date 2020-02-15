#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Int32
from std_msgs.msg import Bool
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree
from scipy.interpolate import splev, splrep, CubicSpline
import numpy as np
import tf
import threading
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

LOOKAHEAD_WPS = 100  # Number of waypoints we will publish. You can change this number
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

        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)
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
        self.dbw_enable = False
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoints_tree = None
        self.curr_pose = None
        self.curr_twist = []
        self.curr_acc = 0.0
        self.stopline_wp_idx = -1
        self.prev_stopline_wp_idx = -1
        self.stop_buffer = 10
        self.change_plan = True
        self.decel_limit = rospy.get_param('/dbw_node/decel_limit', -5)
        self.accel_limit = rospy.get_param('/dbw_node/accel_limit', 1.)
        self.max_vel = float(rospy.get_param(
            '/waypoint_loader/velocity', 40.0)) * 1000.0 / 3600.0
        rospy.loginfo("max_vel: %f", self.max_vel)
        self.hz = 30
        self.prev_final_waypoints = []
        self.max_jmt_duration = 40.0
        self.jmt_duration_dict = None
        self.generate_jmt_duration_dict()

        # start_x = [0.0, 11.1111, 0.0]
        # end_x = [200.0, 11.1111, 0.0]
        # jmt = self.get_jmt_params(start_x, end_x, 20.0)
        # rospy.loginfo("jmt: %f, %f, %f, %f, %f, %f",
        #               jmt[0], jmt[1], jmt[2], jmt[3], jmt[4], jmt[5])
        # self.test_hyperplane()
        # total = 20
        # start = 6
        # end = 23
        # idx_list = []
        # idx = start
        # while idx != end % total:
        #     rospy.loginfo("idx: %d", idx)
        #     idx_list.append(idx)
        #     idx = (idx + 1) % total
        # print(idx_list)

        self.loop()

    def dbw_enabled_cb(self, msg):
        self.dbw_enable = msg.data
        # rospy.loginfo("dbw_enable: %d", self.dbw_enable)

    def generate_jmt_duration_dict(self):
        max_duration = self.max_jmt_duration
        self.jmt_duration_dict = {}
        for duration in np.arange(0.2, max_duration, 0.1):
            t1 = duration
            t2 = math.pow(t1, 2)
            t3 = math.pow(t1, 3)
            t4 = math.pow(t1, 4)
            t5 = math.pow(t1, 5)
            a = np.array([[t3, t4, t5], [3 * t2, 4 * t3, 5 * t4],
                        [6 * t1, 12 * t2, 20 * t3]])
            a_inv = np.linalg.inv(a)
            # We only need t1 and a_inv later.
            self.jmt_duration_dict[str(duration)] = {'t1': t1, 'a_inv': a_inv}
            # self.jmt_duration_dict[str(duration)] =
            #     {'t1': t1, 't2': t2, 't3': t3, 't4': t4, 't5': t5, 'a': a, 'a_inv': a_inv}

    def loop(self):
        rate = rospy.Rate(self.hz)
        while not rospy.is_shutdown():
            if self.curr_pose is not None and self.waypoints_tree is not None and len(self.curr_twist) > 0:
                # Get closest waypoint
                if self.dbw_enable:
                    # time1 = rospy.Time.now()
                    x = self.curr_pose.pose.position.x
                    y = self.curr_pose.pose.position.y
                    closest_wp_idx = self.get_closest_waypoint_id(
                        x, y, self.waypoints_2d, self.waypoints_tree)
                    self.publish_waypoints(closest_wp_idx)
                    # time2 = rospy.Time.now()
                    # duration = time2.to_sec() - time1.to_sec()
                    # rospy.loginfo("loop took time: %f", duration)
                else:
                    self.change_plan = True
            rate.sleep()

    def get_closest_waypoint_id(self, x, y, waypoints_2d, waypoints_tree):
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
        self.test_ahead_or_behind([x, y], closest_coord, prev_coord)
        if self.test_ahead_or_behind([x, y], closest_coord, prev_coord):
            closest_idx = (closest_idx + 1) % len(waypoints_2d)
        return closest_idx

    def get_closest_waypoint_id_no_kdtree(self, x, y, waypoints):
        closest_idx = -1
        min_dist_2 = 1e10
        # far_away_counter = 0
        for i in range(len(waypoints)):
            x1 = waypoints[i].pose.pose.position.x
            y1 = waypoints[i].pose.pose.position.y
            dist_2 = pow(x - x1, 2) + pow(y - y1, 2)
            if dist_2 < min_dist_2:
                min_dist_2 = dist_2
                closest_idx = i
            #     far_away_counter = 0
            # else:
            #     far_away_counter += 1
            # if far_away_counter > 10:
            #     break
        if closest_idx == 0:
            pose_next = [waypoints[1].pose.pose.position.x,
                waypoints[1].pose.pose.position.y]
            pose_1 = [waypoints[0].pose.pose.position.x,
                waypoints[0].pose.pose.position.y]
            vec = [pose_next[0] - pose_1[0], pose_next[1] - pose_1[1]]
            pose_2 = [pose_1[0] - vec[0], pose_1[1] - vec[1]]
            if self.test_ahead_or_behind([x, y], pose_1, pose_2) is True:
                closest_idx += 1
        elif closest_idx > 0:
            pose_1 = [waypoints[closest_idx].pose.pose.position.x,
                waypoints[closest_idx].pose.pose.position.y]
            pose_2 = [waypoints[closest_idx-1].pose.pose.position.x,
                waypoints[closest_idx-1].pose.pose.position.y]
            if self.test_ahead_or_behind([x, y], pose_1, pose_2) is True:
                closest_idx += 1
        return closest_idx

    # Test whether pose is ahead or behind pose_1
    # Format of pose: [x, y]
    # pose is ahead pose_1: Return True
    # pose is behind pose_1: Return False
    def test_ahead_or_behind(self, pose, pose_1, pose_2):
        # Equation for hyperplan through closest coordinate
        pose_vect = np.array(pose)
        pose_1_vec = np.array(pose_1)
        pose_2_vec = np.array(pose_2)

        val = np.dot(pose_1_vec - pose_2_vec, pose_vect - pose_1_vec)
        if val > 0:
            # Less than 90 degrees
            return True
        else:
            # Larger than 90 degrees
            return False

    def test_hyperplane(self):
        pose_a = [3, 4]
        pose_b = [7, 6]
        pose_1 = [5, 5]
        pose_2 = [1, 1]
        vec_a = np.array(pose_a)
        vec_b = np.array(pose_b)
        vec_1 = np.array(pose_1)
        vec_2 = np.array(pose_2)

        # Test a, 1, 2
        rospy.loginfo("Test a, 1, 2:")
        val = np.dot(vec_1 - vec_2, vec_a - vec_1)
        if val > 0:
            rospy.loginfo("a is ahead 1")
        else:
            rospy.loginfo("a is behind 1")
        
        rospy.loginfo("Test a, 1, 2 by func:")
        if self.test_ahead_or_behind(pose_a, pose_1, pose_2):
            rospy.loginfo("a is ahead 1")
        else:
            rospy.loginfo("a is behind 1")

        # Test b, 1, 2
        rospy.loginfo("Test b, 1, 2:")
        val = np.dot(vec_1 - vec_2, vec_b - vec_1)
        if val > 0:
            rospy.loginfo("b is ahead 1")
        else:
            rospy.loginfo("b is behind 1")

        rospy.loginfo("Test b, 1, 2 by func:")
        if self.test_ahead_or_behind(pose_b, pose_1, pose_2):
            rospy.loginfo("b is ahead 1")
        else:
            rospy.loginfo("b is behind 1")

    def publish_waypoints(self, closest_wp_idx):
        # Using jmt to generate final_waypoints
        if self.stopline_wp_idx >= 0:
            stop_dist = self.distance(self.base_waypoints.waypoints,
                                        closest_wp_idx, self.stopline_wp_idx - self.stop_buffer)
            # rospy.loginfo("stop_dist: %f, change_plan: %d", stop_dist, self.change_plan)
            if stop_dist > 50.0:
                vel = max(self.max_vel * 0.7, self.curr_twist[-1].twist.linear.x - 0.1)
                # rospy.loginfo("Slow down to: %f", vel)
                final_lane = self.generate_normal_waypoints(closest_wp_idx, vel)
            elif self.change_plan == False:
                # time1 = rospy.Time.now()
                final_lane = self.extend_stop_waypoints()
                # time2 = rospy.Time.now()
                # duration = time2.to_sec() - time1.to_sec()
                # rospy.loginfo("extend_stop_waypoints took time: %f", duration)
                self.prev_final_waypoints = final_lane.waypoints
            else:
                if stop_dist > 35.0 and stop_dist <= 50.0:
                    # Try to generate a stop plan in this distance range
                    final_lane = self.generate_jmt_waypoints(
                        closest_wp_idx, self.stopline_wp_idx - self.stop_buffer, 0.0)
                    # final_lane = self.generate_stop_waypoints(
                    #     closest_wp_idx, self.stopline_wp_idx - self.stop_buffer)
                    if len(final_lane.waypoints) == 0:
                        vel = max(self.max_vel * 0.7, self.curr_twist[-1].twist.linear.x - 0.1)
                        # rospy.loginfo("Slow down to: %f", vel)
                        final_lane = self.generate_normal_waypoints(closest_wp_idx, vel)
                    else:
                        self.prev_final_waypoints = final_lane.waypoints
                        self.change_plan = False
                        # rospy.loginfo("Get a jmt stop plan: %d", len(self.prev_final_waypoints))
                        # for idx, wp in enumerate(self.prev_final_waypoints):
                        #     rospy.loginfo("idx: %d, vel: %f", idx, wp.twist.twist.linear.x)
                else:
                    # We must generate a stop plan in this distance range!
                    if self.curr_twist[-1].twist.linear.x < 1e-2 and stop_dist < 10.0:
                        final_lane = self.generate_stop_waypoints(
                            closest_wp_idx, self.stopline_wp_idx - self.stop_buffer)
                    else:
                        final_lane = self.generate_jmt_waypoints(
                            closest_wp_idx, self.stopline_wp_idx - self.stop_buffer, 0.0)
                    
                    if len(final_lane.waypoints) == 0:
                        # rospy.logwarn("JMT fail to find a stop plan, just stop it!")
                        end_wp_idx = self.stopline_wp_idx - self.stop_buffer
                        final_lane = self.generate_stop_waypoints(closest_wp_idx, end_wp_idx)
                    # else:
                    #     rospy.loginfo("Get a jmt stop plan: %d", len(final_lane.waypoints))
                    #     for idx, wp in enumerate(final_lane.waypoints):
                    #         rospy.loginfo("idx: %d, vel: %f", idx, wp.twist.twist.linear.x)
                    self.prev_final_waypoints = final_lane.waypoints
                    self.change_plan = False
                    
        else:
            # Try to follow default base_waypoints
            if self.change_plan == True:
                end_wp_idx = closest_wp_idx + LOOKAHEAD_WPS
                if end_wp_idx >= len(self.base_waypoints.waypoints):
                    end_wp_idx = end_wp_idx % len(self.base_waypoints.waypoints)
                # rospy.loginfo("end_wp_idx: %d", end_wp_idx)
                # if end_wp_idx >= len(self.base_waypoints.waypoints):
                #     rospy.logerr("end_wp_idx %d out of bound! Max: %d",
                #                 end_wp_idx, len(self.base_waypoints.waypoints)-1)
                #     end_wp_idx = len(self.base_waypoints.waypoints) - 1
                end_vel = self.max_vel
                final_lane = self.generate_jmt_waypoints(closest_wp_idx, end_wp_idx, end_vel)
                # final_lane = self.generate_normal_waypoints(closest_wp_idx, end_vel)
                if len(final_lane.waypoints) == 0 or final_lane.waypoints[0].twist.twist.linear.x < 1.0:
                    end_vel = min(self.max_vel, self.curr_twist[-1].twist.linear.x + 0.3)
                    # rospy.loginfo("end_vel: %f", end_vel)
                    final_lane = self.generate_normal_waypoints(closest_wp_idx, end_vel)
                    if abs(end_vel - self.max_vel) < 1e-3:
                        self.change_plan = False
                        self.prev_final_waypoints = final_lane.waypoints
                else:
                    # for i, wp in enumerate(final_lane.waypoints):
                    #     rospy.loginfo("i: %d, vel: %f", i, wp.twist.twist.linear.x)
                    self.change_plan = False
                    self.prev_final_waypoints = final_lane.waypoints
            else:
                final_lane = self.extend_normal_waypoints(closest_wp_idx)
                self.prev_final_waypoints = final_lane.waypoints
        
        # for wp in final_lane.waypoints:
        #     rospy.loginfo("final xy: %f, %f", wp.pose.pose.position.x, wp.pose.pose.position.y)
        self.final_waypoints_pub.publish(final_lane)
        # if self.stopline_wp_idx >= 0:
        #     lane = Lane()
        #     lane.header = final_lane.header
        #     stop_wp = self.base_waypoints.waypoints[self.stopline_wp_idx - self.stop_buffer]
        #     lane.waypoints.append(stop_wp)
        #     self.stopline_pub.publish(lane)

    def generate_normal_waypoints(self, closest_wp_idx, vel):
        lane = Lane()
        lane.header = self.curr_pose.header
        lane.waypoints = []
        waypoints = []
        if (closest_wp_idx + LOOKAHEAD_WPS) < len(self.base_waypoints.waypoints):
            waypoints = self.base_waypoints.waypoints[
                closest_wp_idx:closest_wp_idx+LOOKAHEAD_WPS]
        else:
            tail_size = LOOKAHEAD_WPS - (len(self.base_waypoints.waypoints) - closest_wp_idx)
            waypoints_1 = self.base_waypoints.waypoints[closest_wp_idx:]
            waypoints_2 = self.base_waypoints.waypoints[0:tail_size]
            waypoints = waypoints_1 + waypoints_2
        for wp in waypoints:
            new_waypoint = Waypoint()
            new_waypoint.pose.header = self.curr_pose.header
            new_waypoint.pose = wp.pose
            new_waypoint.twist.header = self.curr_pose.header
            new_waypoint.twist.twist.linear.x = vel
            lane.waypoints.append(new_waypoint)
            # rospy.loginfo("i: %d, vel: %f to %f",
            #     idx, wp.twist.twist.linear.x, new_waypoint.twist.twist.linear.x)
        return lane

    def generate_stop_waypoints(self, closest_wp_idx, end_wp_idx):
        lane = Lane()
        lane.header = self.curr_pose.header

        waypoints = []
        if closest_wp_idx <= end_wp_idx:
            waypoints = self.base_waypoints.waypoints[
                closest_wp_idx:end_wp_idx]
        else:
            waypoints_1 = self.base_waypoints.waypoints[closest_wp_idx:]
            waypoints_2 = self.base_waypoints.waypoints[0:end_wp_idx]
            waypoints = waypoints_1 + waypoints_2
        for wp in waypoints:
            new_waypoint = Waypoint()
            new_waypoint.pose.header = self.curr_pose.header
            new_waypoint.pose = wp.pose
            new_waypoint.twist.header = self.curr_pose.header
            new_waypoint.twist = wp.twist
            lane.waypoints.append(new_waypoint)

        wp_size = len(lane.waypoints)
        curr_vel = self.curr_twist[-1].twist.linear.x
        if wp_size > 0:
            delta_vel = curr_vel / wp_size
        else:
            delta_vel = self.max_vel
        # rospy.loginfo("generate_stop_waypoints:")
        for i, wp in enumerate(lane.waypoints):
            vel = curr_vel - (i+1) * delta_vel
            if vel < 0.1:
                vel = 0.0
            # rospy.loginfo("i: %d, vel: %f", i, vel)
            wp.twist.twist.linear.x = vel
        return lane

    def extend_stop_waypoints(self):
        lane = Lane()
        lane.header = self.curr_pose.header
        
        curr_x = self.curr_pose.pose.position.x
        curr_y = self.curr_pose.pose.position.y
        prev_closest_idx = self.get_closest_waypoint_id_no_kdtree(curr_x, curr_y,
            self.prev_final_waypoints)
        
        waypoints_1 = self.prev_final_waypoints[prev_closest_idx:] if prev_closest_idx >= 0 else []
        extend_size = LOOKAHEAD_WPS - len(waypoints_1)
        if extend_size + (self.stopline_wp_idx - self.stop_buffer + 1) < len(self.base_waypoints.waypoints):
            idx = self.stopline_wp_idx - self.stop_buffer + 1
            waypoints_2 = self.base_waypoints.waypoints[idx:idx + extend_size]
        else:
            stop_idx = self.stopline_wp_idx - self.stop_buffer + 1
            idx = extend_size - (len(self.base_waypoints.waypoints) - stop_idx)
            waypoints_2 = self.base_waypoints.waypoints[stop_idx:] + self.base_waypoints.waypoints[0:idx]
        
        # waypoints_1 is slicing from self.prev_final_waypoints
        # We can use them directlly.
        lane.waypoints = waypoints_1
        # waypoints_2 is slicing from self.base_waypoints.waypoints
        # We should't modify them directly, we should copy them.
        for wp in waypoints_2:
            new_waypoint = Waypoint()
            new_waypoint.pose = wp.pose
            new_waypoint.twist = wp.twist
            new_waypoint.twist.twist.linear.x = 0.0
            lane.waypoints.append(new_waypoint)
        return lane

    def extend_normal_waypoints(self, closest_wp_idx):
        lane = Lane()
        lane.header = self.curr_pose.header
        
        curr_x = self.curr_pose.pose.position.x
        curr_y = self.curr_pose.pose.position.y
        # rospy.loginfo("curr xy: %f, %f", curr_x, curr_y)
        rospy.loginfo("prev_final_waypoints size: %d", len(self.prev_final_waypoints))
        prev_closest_idx = self.get_closest_waypoint_id_no_kdtree(curr_x, curr_y,
            self.prev_final_waypoints)
        rospy.loginfo("prev_closest_idx: %d", prev_closest_idx)

        waypoints_1 = self.prev_final_waypoints[prev_closest_idx:] if prev_closest_idx >= 0 else []
        waypoints_2 = []
        if len(waypoints_1) > 0:
            prev_last_x = waypoints_1[-1].pose.pose.position.x
            prev_last_y = waypoints_1[-1].pose.pose.position.y
            prev_last_wp_idx = self.get_closest_waypoint_id(
                prev_last_x, prev_last_y, self.waypoints_2d, self.waypoints_tree)
            wp_idx_diff = prev_last_wp_idx - closest_wp_idx
            if wp_idx_diff < 0:
                wp_idx_diff += len(self.base_waypoints.waypoints)
            if wp_idx_diff < LOOKAHEAD_WPS:
                next_wp_idx = prev_last_wp_idx + 1
                extend_size = LOOKAHEAD_WPS - wp_idx_diff
                if extend_size + next_wp_idx <= len(self.base_waypoints.waypoints):
                    idx = next_wp_idx
                    waypoints_2 = self.base_waypoints.waypoints[idx:idx + extend_size]
                else:
                    idx = extend_size - (len(self.base_waypoints.waypoints) - next_wp_idx)
                    waypoints_2 = self.base_waypoints.waypoints[next_wp_idx:] + self.base_waypoints.waypoints[0:idx]
        else:
            end_idx = closest_wp_idx + LOOKAHEAD_WPS
            if end_idx < len(self.base_waypoints.waypoints):
                waypoints_1 = self.base_waypoints.waypoints[closest_wp_idx:end_idx]
            else:
                end_idx = LOOKAHEAD_WPS - (len(self.base_waypoints.waypoints) - closest_wp_idx)
                waypoints_1 = self.base_waypoints.waypoints[closest_wp_idx:]
                waypoints_2 = self.base_waypoints.waypoints[0:end_idx]
        
        lane.waypoints = waypoints_1 + waypoints_2
        return lane

    def get_cubic_spline(self, start_wp_idx, end_wp_idx):
        # rospy.loginfo("get_cubic_spline: %d, %d", start_wp_idx, end_wp_idx)
        idx = start_wp_idx
        idx_list = []
        while idx != end_wp_idx % len(self.base_waypoints.waypoints):
            # rospy.loginfo("add wp idx: %d", idx)
            idx_list.append(idx)
            idx = (idx + 1) % len(self.base_waypoints.waypoints)
        
        dist = 0.0
        sample_d = []
        sample_x = []
        sample_y = []
        sample_z = []
        for i in range(0, len(idx_list)):
            # rospy.loginfo("wp idx: %d", i)
            idx = idx_list[i]
            if i == 0:
                wp = self.base_waypoints.waypoints[idx]
                sample_d.append(dist)
                sample_x.append(wp.pose.pose.position.x)
                sample_y.append(wp.pose.pose.position.y)
                sample_z.append(wp.pose.pose.position.z)
                x = wp.pose.pose.position.x
                y = wp.pose.pose.position.y
                # rospy.loginfo("dist: %f, sample xy: %f, %f", dist, x, y)
            else:
                idx_1 = idx_list[i-1]
                idx_2 = idx_list[i]
                wp1 = self.base_waypoints.waypoints[idx_1]
                wp2 = self.base_waypoints.waypoints[idx_2]
                dist += math.sqrt(pow(wp1.pose.pose.position.x - wp2.pose.pose.position.x, 2) +
                    pow(wp1.pose.pose.position.y - wp2.pose.pose.position.y, 2))
                sample_d.append(dist)
                sample_x.append(wp2.pose.pose.position.x)
                sample_y.append(wp2.pose.pose.position.y)
                sample_z.append(wp2.pose.pose.position.z)
                x = wp2.pose.pose.position.x
                y = wp2.pose.pose.position.y
                # rospy.loginfo("dist: %f, sample xy: %f, %f", dist, x, y)
        cs_x = CubicSpline(sample_d, sample_x)
        cs_y = CubicSpline(sample_d, sample_y)
        cs_z = CubicSpline(sample_d, sample_z)
        return cs_x, cs_y, cs_z

    def generate_jmt_waypoints(self, closest_wp_idx, end_wp_idx, end_vel):
        # if self.curr_twist[-1].twist.linear.x < end_vel:
        rospy.loginfo("closest_wp_idx: %d, end_wp_idx: %d, curr_vel: %f, end_vel: %f",
                        closest_wp_idx, end_wp_idx, self.curr_twist[-1].twist.linear.x, end_vel)
        lane = Lane()
        lane.header = self.curr_pose.header
        lane.waypoints = []

        dist = self.distance(self.base_waypoints.waypoints, closest_wp_idx, end_wp_idx)
        waypoints = []
        start = [0.0, 0.0, 0.0]
        end = [dist, end_vel, 0.0]
        TIME_STEP = 0.1
        start[1] = self.curr_twist[-1].twist.linear.x
        # start[2] = self.curr_acc
        rospy.loginfo("jmt start: %f, %f, %f", start[0], start[1], start[2])
        rospy.loginfo("jmt end: %f, %f, %f", end[0], end[1], end[2])

        jmt_params = None
        max_duration = self.max_jmt_duration
        jmt_duration = max_duration
        for duration in np.arange(0.2, max_duration, 0.1):
            jmt = self.get_jmt_params(start, end, duration)
            if self.check_jmt(jmt, duration):
                jmt_duration = duration
                jmt_params = jmt
                break
        if jmt_params is not None:
            rospy.loginfo("jmt_params: %f, %f, %f, %f, %f, %f",
                          jmt_params[0], jmt_params[1], jmt_params[2],
                          jmt_params[3], jmt_params[4], jmt_params[5])
            vel_params = derivative(jmt_params)
            # sample_xy = []
            # if end_wp_idx >= closest_wp_idx:
            #     for i in range(closest_wp_idx, end_wp_idx):
            #         sample_xy.append([self.base_waypoints.waypoints[i].pose.pose.position.x,
            #                         self.base_waypoints.waypoints[i].pose.pose.position.y])
            # else:
            #     for i in range(closest_wp_idx, len(self.base_waypoints.waypoints)):
            #         sample_xy.append([self.base_waypoints.waypoints[i].pose.pose.position.x,
            #                         self.base_waypoints.waypoints[i].pose.pose.position.y])
            #     for i in range(0, end_wp_idx):
            #         sample_xy.append([self.base_waypoints.waypoints[i].pose.pose.position.x,
            #                         self.base_waypoints.waypoints[i].pose.pose.position.y])
            # # Sort waypoint accroding to x coordinate, spline needs x to be sorted.
            # sample_xy = sorted(sample_xy, key=lambda x: x[0])
            # # Remove the xy point whose x coordinate is too close to previous xy point.
            # safe_sample_xy = [sample_xy[0]]
            # for i in range(1, len(sample_xy)):
            #     if abs(sample_xy[i][0] - sample_xy[i-1][0]) > 0.01:
            #         safe_sample_xy.append(sample_xy[i])
            # sample_xy = np.array(safe_sample_xy)
            # try:
            #     cs = CubicSpline(sample_xy[:, 0], sample_xy[:, 1])
            # except:
            #     rospy.logerr("CubicSpline fail:")
            #     for x in sample_xy[:, 0]:
            #         print("x: %f", x)
            cs_x, cs_y, cs_z = self.get_cubic_spline(closest_wp_idx, end_wp_idx)

            # start_x = self.base_waypoints.waypoints[closest_wp_idx].pose.pose.position.x
            # end_x = self.base_waypoints.waypoints[end_wp_idx].pose.pose.position.x
            # dist_x = end_x - start_x
            # new_x = []
            # new_y = []
            query_d = []
            new_vel = []
            # prev_d = 0.0
            delta_time = 0.0
            for dt in np.arange(0.1, jmt_duration, TIME_STEP):
                d = poly_eval(dt, jmt_params)
                next_vel = poly_eval(dt, vel_params)
                # delta_dist = (d - prev_d)
                # delta_time += TIME_STEP
                # if True:
                # if next_vel > 0.5 or delta_dist > 1.0:
                if next_vel > 0.08 or dt == jmt_duration - TIME_STEP:
                    # dist_ratio = d / dist
                    # next_x = start_x + dist_x * dist_ratio
                    # new_x.append(next_x)
                    query_d.append(d)
                    new_vel.append(next_vel)
                    # prev_d = d
                    # delta_time = 0.0
            # new_x = np.array(new_x)
            # new_y = cs(new_x)
            # rospy.loginfo("query d: %f, %f, size: %d", query_d[0], query_d[-1], len(query_d))
            curr_x = self.curr_pose.pose.position.x
            curr_y = self.curr_pose.pose.position.y
            # rospy.loginfo("curr xy: %f, %f", curr_x, curr_y)
            new_x = cs_x(query_d)
            new_y = cs_y(query_d)
            new_z = cs_z(query_d)
            # Skip the angular velocity and yaw, because waypoint_follower doesn't use it.
            # It computes angular velocity accroding to the curvature and linear velocity.
            for i in range(len(new_vel)):
                waypoint = Waypoint()
                waypoint.pose.header = self.curr_pose.header
                waypoint.pose.pose.position.x = new_x[i]
                waypoint.pose.pose.position.y = new_y[i]
                waypoint.pose.pose.position.z = new_z[i]
                rospy.loginfo("jmt xy: %f, %f", new_x[i], new_y[i])
                # Transform from yaw to quaternion
                # q = tf.transformations.quaternion_from_euler(0., 0., yaw)
                # waypoint.pose.pose.orientation = Quaternion(*q)
                waypoint.twist.header = self.curr_pose.header
                waypoint.twist.twist.linear.x = new_vel[i]
                waypoints.append(waypoint)
            if len(waypoints) < LOOKAHEAD_WPS:
                extend_size = LOOKAHEAD_WPS - len(waypoints)
                last_x = waypoints[-1].pose.pose.position.x
                last_y = waypoints[-1].pose.pose.position.y
                last_wp_idx = self.get_closest_waypoint_id(
                    last_x, last_y, self.waypoints_2d, self.waypoints_tree)
                last_wp_idx = (last_wp_idx + 1) % len(self.base_waypoints.waypoints)
                extend_waypoints = self.base_waypoints.waypoints[last_wp_idx:last_wp_idx+extend_size]
                for wp in extend_waypoints:
                    wp.twist.twist.linear.x = end_vel
                    waypoints.append(wp)
        else:
            rospy.logwarn("Fail to find jmt parameters!")

        lane.waypoints = waypoints
        return lane

    def get_jmt_params(self, start, end, duration):
        jmt_duration_dict = self.jmt_duration_dict[str(duration)]
        t1 = jmt_duration_dict['t1']
        a_inv = jmt_duration_dict['a_inv']

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
        # rospy.loginfo("check_jmt duration: %f", duration)
        result = True
        for t in np.arange(0.1, duration, 0.1):
            # dist = poly_eval(t, jmt_params)
            vel = poly_eval(t, vel_params)
            acc = abs(poly_eval(t, acc_params))
            jerk = abs(poly_eval(t, jerk_params))
            if (vel < 0.0 and abs(vel) > 1e-5) or vel > self.max_vel + 1e-3 or acc > self.accel_limit + 1e-3 or jerk > 9.999:
                return False
        if result:
            rospy.loginfo("check_jmt duration: %f pass", duration)
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

    def pose_cb(self, msg):
        # TODO: Implement
        self.curr_pose = msg

    def velocity_cb(self, msg):
        if len(self.curr_twist) == 50:
            self.curr_twist = self.curr_twist[1:len(self.curr_twist)]
        self.curr_twist.append(msg)
        # rospy.loginfo("Curr vel: %f", self.curr_twist[-1].twist.linear.x)
        # Estimate current acceleration
        # if len(self.curr_twist) >= 2:
        #     twist_0 = self.curr_twist[0]
        #     twist_1 = self.curr_twist[-1]
        #     duration = twist_1.header.stamp.to_sec() - twist_0.header.stamp.to_sec()
        #     vel_diff = twist_1.twist.linear.x - twist_0.twist.linear.x
        #     self.curr_acc = vel_diff / duration
        #     rospy.loginfo("acc: %f, vel_diff: %f, duration: %f",
        #                   self.curr_acc, vel_diff, duration)

    def waypoints_cb(self, waypoints):
        # TODO: Implement
        self.base_waypoints = waypoints
        global LOOKAHEAD_WPS
        temp = LOOKAHEAD_WPS
        max_wp_count = min(LOOKAHEAD_WPS, len(waypoints.waypoints) / 2)
        LOOKAHEAD_WPS = max_wp_count
        if temp != LOOKAHEAD_WPS:
            rospy.loginfo("Change LOOKAHEAD_WPS: %d to %d", temp, LOOKAHEAD_WPS)
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x,
                                  waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoints_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        x = self.curr_pose.pose.position.x
        y = self.curr_pose.pose.position.y
        closest_wp_idx = self.get_closest_waypoint_id(
            x, y, self.waypoints_2d, self.waypoints_tree)
        # dist = self.distance(self.base_waypoints.waypoints, closest_wp_idx, msg.data)
        # if self.prev_stopline_wp_idx < 0 and msg.data >= 0 and dist < 10.0:
        if self.prev_stopline_wp_idx < 0 and msg.data >= 0 and msg.data - closest_wp_idx < 10:
            # The yellow light is too close, just go through this traffic light directlly.
            rospy.loginfo(
                "The yellow light is too close, just pass this traffic light directlly")
            return
        else:
            self.prev_stopline_wp_idx = self.stopline_wp_idx
            self.stopline_wp_idx = msg.data
            if self.prev_stopline_wp_idx != self.stopline_wp_idx:
                self.change_plan = True

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
        if wp2 >= wp1:
            for i in range(wp1, wp2 + 1):
                dist += dl(waypoints[wp1].pose.pose.position,
                        waypoints[i].pose.pose.position)
                wp1 = i
        else:
            for i in range(wp1, len(waypoints)):
                dist += dl(waypoints[wp1].pose.pose.position,
                        waypoints[i].pose.pose.position)
                wp1 = i
            dist += dl(waypoints[0].pose.pose.position,
                        waypoints[len(waypoints)-1].pose.pose.position)
            for i in range(0, wp2 + 1):
                dist += dl(waypoints[i].pose.pose.position,
                        waypoints[i+1].pose.pose.position)
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
