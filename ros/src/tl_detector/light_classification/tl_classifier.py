from styx_msgs.msg import TrafficLight
from std_msgs.msg import Bool
import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
import time
from scipy.stats import norm
import os
import rospy


# Colors (one for each class)
cmap = ImageColor.colormap
# print("Number of colors =", len(cmap))
COLOR_LIST = sorted([c for c in cmap.keys()])


class TLClassifier(object):
    def __init__(self, is_site):
        # TODO load classifier
        self.is_site = is_site
        self.output_img = False
        rospy.Subscriber('/show_tl_classifier', Bool, self.output_image_cb)

        self.cwd = os.path.dirname(os.path.realpath(__file__))
        SITE_SSD_INCEPTION = self.cwd + '/model/ssd_inception_v2_coco_real/'
        # SIM_SSD_INCEPTION = self.cwd + '/model/ssd_inception_v2_coco_sim/'
        # SITE_SSD_MOBILENET = self.cwd + '/model/ssd_mobilenet_v1_coco_real/'
        SIM_SSD_MOBILENET = self.cwd + '/model/ssd_mobilenet_v1_coco_sim/'
        GRAPH_FILE = 'frozen_inference_graph.pb'
        MODEL_PATH = SITE_SSD_INCEPTION if is_site else SIM_SSD_MOBILENET
        LOAD_FILE = MODEL_PATH + GRAPH_FILE

        rospy.loginfo("Model path is : %s", LOAD_FILE)
        self.graph = self.load_graph(LOAD_FILE)
        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        self.image_tensor = self.graph.get_tensor_by_name(
            'image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.graph.get_tensor_by_name(
            'detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.graph.get_tensor_by_name(
            'detection_scores:0')
        # The classification of the object (integer id).
        self.detection_classes = self.graph.get_tensor_by_name(
            'detection_classes:0')
        rospy.loginfo("Load graph complete")
        self.idx = 0
        # self.test_model()

    def output_image_cb(self, msg):
        self.output_img = msg.data

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # TODO implement light color prediction
        # Convert bgr to rgb
        # image = np.dstack((image[:, :, 2], image[:, :, 1], image[:, :, 0]))
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_expanded = np.expand_dims(image, axis=0)
        with tf.Session(graph=self.graph) as sess:
            # Actual detection.
            (boxes, scores, classes) = sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                                feed_dict={self.image_tensor: image_expanded})

            # Remove unnecessary dimensions
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes)

            confidence_cutoff = 0.6
            # Filter boxes with a confidence score less than `confidence_cutoff`
            boxes, scores, classes = self.filter_boxes(
                confidence_cutoff, boxes, scores, classes)

            # rospy.loginfo("Shape of classes: {}".format(classes.shape))
            state = TrafficLight.UNKNOWN
            state_count = [0, 0, 0, 0, 0]
            state_max_score = [0, 0, 0, 0, 0]
            # rospy.loginfo("Box count: {}".format(classes.shape[0]))
            for i in range(classes.shape[0]):
                # My sim version, red=1, yellow=2, green=3, unknown=4
                class_type = int(classes[i])
                if class_type == 1:
                    class_type = TrafficLight.RED
                elif class_type == 2:
                    class_type = TrafficLight.YELLOW
                elif class_type == 3:
                    class_type = TrafficLight.GREEN
                elif class_type == 4:
                    class_type = TrafficLight.UNKNOWN
                state_count[class_type] += 1
                if state_max_score[class_type] < scores[i]:
                    state_max_score[class_type] = scores[i]

            if state_count[TrafficLight.RED] >= 2:
                state = TrafficLight.RED
            elif state_count[TrafficLight.YELLOW] >= 2:
                state = TrafficLight.YELLOW
            elif state_count[TrafficLight.GREEN] >= 2:
                state = TrafficLight.GREEN

            if self.output_img:
                # The current box coordinates are normalized to a range between 0 and 1.
                # This converts the coordinates actual location on the image.
                # rospy.loginfo("image.shape: {}".format(image.shape))
                height, width, channels = image.shape
                box_coords = self.to_image_coords(boxes, height, width)
                # Each class with be represented by a differently colored box
                self.draw_boxes(image, box_coords, classes, scores)

        return state, state_max_score[state]

    def load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph

    def filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def to_image_coords(self, boxes, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1].

        This converts it back to the original coordinate based on the image
        size.
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width

        return box_coords

    def draw_boxes(self, image, boxes, classes, scores, thickness=4):
        """Draw bounding boxes on the image"""
        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        for i in range(len(boxes)):
            bot, left, top, right = boxes[i, ...]
            class_id = int(classes[i])
            state = 'u'
            if class_id == 1:
                color = ImageColor.getrgb("red")
                state = "r"
            elif class_id == 2:
                color = ImageColor.getrgb("yellow")
                state = 'y'
            elif class_id == 3:
                color = ImageColor.getrgb("green")
                state = 'g'
            elif class_id == 4:
                color = ImageColor.getrgb("white")
            draw.line([(left, top), (left, bot), (right, bot),
                       (right, top), (left, top)], width=thickness, fill=color)
            draw.text((left, top), "{}".format(scores[i]), fill=color)
            rospy.loginfo("id: %d, box: (%f, %f, %f, %f), state: %s, score: %f",
                          i, left, top, right, bot, state, scores[i])
        out_path = self.cwd + '/output/out_'
        image.save(out_path + str(self.idx) + '.jpg')
        self.idx += 1

    def test_model(self):
        # for i in range(672, 738):
        #     im = np.asarray(Image.open(
        #         self.cwd + "/test/v1/left{:04d}".format(i) + ".jpg"))
        #     self.get_classification(im)
        # for i in range(890, 927):
        #     im = np.asarray(Image.open(
        #         self.cwd + "/test/v1/left{:04d}".format(i) + ".jpg"))
        #     self.get_classification(im)
        # for i in range(1075, 1127):
        #     im = np.asarray(Image.open(
        #         self.cwd + "/test/v1/left{:04d}".format(i) + ".jpg"))
        #     self.get_classification(im)
        for i in range(231, 526):
            im = np.asarray(Image.open(
                self.cwd + "/test/v2/bag{:04d}".format(i) + ".png"))
            self.get_classification(im)
