#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree

STATE_COUNT_THRESHOLD = 3


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoints_tree = None
        self.curr_pose = None
        self.camera_image = None
        self.lights = []

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        rospy.Subscriber('/vehicle/traffic_lights',
                         TrafficLightArray, self.traffic_cb)
        rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher(
            '/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.curr_pose = msg

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        if self.waypoints_2d is None:
            self.waypoints_2d = [[waypoint.pose.pose.position.x,
                                  waypoint.pose.pose.position.y] for waypoint in waypoints.waypoints]
            self.waypoints_tree = KDTree(self.waypoints_2d)
            # rospy.loginfo("Receive base_waypoints")

    def traffic_cb(self, msg):
        # rospy.loginfo("Receive traffic light")
        self.lights = msg.lights
        # TODO: This should not be called here!
        # self.image_cb(msg)

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()
        if light_wp >= 0:
            if state == TrafficLight.UNKNOWN:
                rospy.loginfo("TL state: UNKNOWN")
            elif state == TrafficLight.GREEN:
                rospy.loginfo("TL state: GREEN")
            elif state == TrafficLight.YELLOW:
                rospy.loginfo("TL state: YELLOW")
            elif state == TrafficLight.RED:
                rospy.loginfo("TL state: RED")

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
            # rospy.loginfo("Set state_count = 0")
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            # rospy.loginfo("Closest red light: %d", light_wp)
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1
        # rospy.loginfo("state_count: %d", self.state_count)

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        # TODO implement
        if self.waypoints_tree is None:
            rospy.logerr("self.waypoints_tree is None")
            return -1
        else:
            closest_idx = self.waypoints_tree.query([x, y], 1)[1]
            return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # For testing, just return the traffic light state
        # return light.state

        if self.has_image is False:
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # (rows, cols, channels) = cv_image.shape
        # rospy.loginfo("Img, size: %d, %d, %d,", rows, cols, channels)
        # rospy.loginfo("Img, type: {}".format(type(cv_image)))

        # Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = None

        if self.curr_pose is not None and self.base_waypoints is not None:
            car_wp_idx = self.get_closest_waypoint(
                self.curr_pose.pose.position.x, self.curr_pose.pose.position.y)

            # TODO find the closest visible traffic light (if one exists)
            min_dist = len(self.base_waypoints.waypoints)
            # List of positions that correspond to the line to stop in front of for a given intersection
            stop_line_positions = self.config['stop_line_positions']
            for idx, light in enumerate(self.lights):
                line = stop_line_positions[idx]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                dist = temp_wp_idx - car_wp_idx
                if dist >= 0 and dist < min_dist:
                    min_dist = dist
                    closest_light = light
                    line_wp_idx = temp_wp_idx

        if closest_light is not None:
            state = self.get_light_state(closest_light)
            # rospy.loginfo("Closest traffic state: %d", state)
            return line_wp_idx, state
        return -1, TrafficLight.UNKNOWN


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
