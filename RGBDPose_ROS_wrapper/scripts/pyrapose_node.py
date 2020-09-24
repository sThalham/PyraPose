#!/usr/bin/env python

import sys

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PoseStamped, Point, Point32, PolygonStamped
from visualization_msgs.msg import MarkerArray, Marker
from cv_bridge import CvBridge, CvBridgeError

from scipy import ndimage, signal
import argparse
import os
import sys
import math
import cv2
import numpy as np
import copy
import transforms3d as tf3d

import keras
import tensorflow as tf
import open3d

sys.path.append("/RGBDPose")
from RGBDPose import models
from RGBDPose.utils.config import read_config_file, parse_anchor_parameters
from RGBDPose.utils.eval import evaluate
from RGBDPose.utils.keras_version import check_keras_version

from RGBDPose_ROS_wrapper.srv import returnPoses
from RGBDPose_ROS_wrapper.msg import PoseWithConfidence

###################################
##### Global Variable Space #######
######## aka. death zone ##########
###################################



# LineMOD
#fxkin = 572.41140
#fykin = 573.57043
#cxkin = 325.26110
#cykin = 242.04899

# our Kinect
fxkin = 575.81573
fykin = 575.81753
cxkin = 314.5
cykin = 235.5


def create_point_cloud(depth, fx, fy, cx, cy, ds):

    rows, cols = depth.shape

    depRe = depth.reshape(rows * cols)
    zP = np.multiply(depRe, ds)

    x, y = np.meshgrid(np.arange(0, cols, 1), np.arange(0, rows, 1), indexing='xy')
    yP = y.reshape(rows * cols) - cy
    xP = x.reshape(rows * cols) - cx
    yP = np.multiply(yP, zP)
    xP = np.multiply(xP, zP)
    yP = np.divide(yP, fy)
    xP = np.divide(xP, fx)

    cloud_final = np.transpose(np.array((xP, yP, zP)))
    cloud_final[cloud_final[:,2]==0] = np.NaN

    return cloud_final


def preprocess_image(x, mode='caffe'):
    x = x.astype(np.float32)

    if mode == 'tf':
        x /= 127.5
        x -= 1.
    elif mode == 'caffe':
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    return x



#################################
############### ROS #############
#################################
class PoseEstimationClass:
    def __init__(self, model, threshold, topic, graph):
        #event that will block until the info is received
        #attribute for storing the rx'd message
        self._model = model
        self._score_th = threshold
        self.graph = graph
        self.threeD_boxes 

        self._msg = None
        self.seq = None
        self.time = None
        self.frame_id = None
        self.bridge = CvBridge()
        self.pose_pub = rospy.Publisher("/object_recognition/poses", , queue_size=1)
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)

    def callback(self, data):
        self.seq = data.header.seq
        self.time = data.header.stamp
        self.frame_id = data.header.frame_id
        self._msg = self.bridge.imgmsg_to_cv2(data, "8UC3")
        det_poses = run_estimation(self._rgb, self._msg, self._model, self._score_th, self.graph, self.seq)
        #img = run_estimation(cc_rgb, self._msg, self._rgb, self._model, self._score_th, self.graph)

        self.publish_Poses(det_poses)

    def publish_pose(self, det_names, det_poses, det_confidences):

	msg = returnPoses()
	for idx in range(det_names):
            item = PoseWithConfidence()
            item.name = det_names[idx] 
            item.confidence = det_confidences[idx]
            item.pose = Pose()
            item.pose.position.x = pose[0]
            item.pose.position.y = pose[1]
            item.pose.position.z = pose[2]
            item.pose.orientation.w = pose[3]
            item.pose.orientation.x = pose[4]
            item.pose.orientation.y = pose[5]
            item.pose.orientation.z = pose[6]
            msg.append(item)

	self.pose_pub.publish(msg


class PoseEstimationServer:
    def __init__(self, model, threshold, service_name, graph):
        #event that will block until the info is received
        #attribute for storing the rx'd message
        self._model = model
        self._score_th = threshold
        self.graph = graph

        self._msg = None
        self.seq = None
        self.time = None
        self.frame_id = None
        self.bridge = CvBridge()
        self.pose_srv = rospy.Service(service_name, returnPoses, self.callback)

    def callback(self, data):
        #print(data)
        self.seq = data.image.header.seq
        self.time = data.image.header.stamp
        self.frame_id = data.image.header.frame_id
        self._msg = self.bridge.imgmsg_to_cv2(data.image, "8UC3")

        obj_names, obj_poses, obj_scores = run_estimation(self._msg, self._model, self._score_th, self.graph, self.frame_id)

        msg = self.fill_pose(det_poses)
        return msg

    def fill_pose(self, det_names, det_poses, det_confidences):

	msg = returnPoses()
	for idx in range(det_names):
            item = PoseWithConfidence()
            item.name = det_names[idx] 
            item.confidence = det_confidences[idx]
            item.pose = Pose()
            item.pose.position.x = det_poses[idx][0]
            item.pose.position.y = det_poses[idx][1]
            item.pose.position.z = det_poses[idx][2]
            item.pose.orientation.w = det_poses[idx][3]
            item.pose.orientation.x = det_poses[idx][4]
            item.pose.orientation.y = det_poses[idx][5]
            item.pose.orientation.z = det_poses[idx][6]
            msg.append(item)

	return msg

        
#################################
########## RetNetPose ###########
#################################
def get_session():
    """ Construct a modified tf session.
    """
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    return tf.Session(config=config)


def parse_args(args):

    parser     = argparse.ArgumentParser(description='Evaluation script for a RetinaNet network.')
    parser.add_argument('model',              help='Path to RetinaNet model.')
    parser.add_argument('--backbone',         help='The backbone of the model.', default='resnet50')
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--score-threshold',  help='Threshold on score to filter detections with (defaults to 0.5).', default=0.5, type=float)
    parser.add_argument('--max-detections',   help='Max Detections per image (defaults to 100).', default=100, type=int)
    parser.add_argument('--image-min-side',   help='Rescale the image so the smallest side is min_side.', type=int, default=480)
    parser.add_argument('--image-max-side',   help='Rescale the image if the largest side is larger than max_side.', type=int, default=640)
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file (only used with --convert-model).')

    return parser.parse_args(args)


def load_model(model_path):

    check_keras_version()

    #if args.gpu:
    #    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    keras.backend.tensorflow_backend.set_session(get_session())

    anchor_params = None
    backbone = 'resnet50'

    print('Loading model, this may take a second...')
    model = models.load_model(model_path, backbone_name=backbone)
    graph = tf.get_default_graph()
    model = models.convert_model(model, anchor_params=anchor_params) # convert model

    # print model summary
    print(model.summary())

    return model, graph


def run_estimation(image, image_dep, model, score_threshold, graph, frame_id):
    obj_names = []
    obj_poses = []
    obj_confs = []

    image = preprocess_image(image)

    #cv2.imwrite('/home/sthalham/retnetpose_image.jpg', image)

    if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

    with graph.as_default():
        boxes3D, scores, mask = model.predict_on_batch(np.expand_dims(image, axis=0))

    for inv_cls in range(scores.shape[2]):

	true_cat = inv_cls + 1
        cls = true_cat

        cls_mask = scores[0, :, inv_cls]

        cls_indices = np.where(cls_mask > threshold)

        if len(cls_indices[0]) < 10:
            continue

	if true_cls == 5:
            name = '006_mustard_bottle'
        if true_cls == 8:
            name = '009_gelatin_box'
        if true_cls == 9:
            name = '010_potted_meat_can'
        if true_cls == 10:
            name = '011_banana'
        if true_cls == 21:
            name = '061_foam_brick'
        else:
            continue 

	obj_names.append(name)
        obj_confs.append(np.sum(cls_mask[cls_indices[0]]))


        k_hyp = len(cls_indices[0])
        ori_points = np.ascontiguousarray(self.threeD_boxes[cls, :, :], dtype=np.float32)  # .reshape((8, 1, 3))
        K = np.float32([fxkin, 0., cxkin, 0., fykin, cykin, 0., 0., 1.]).reshape(3, 3)

        ##############################
        # pnp
        pose_votes = boxes3D[0, cls_indices, :]
        est_points = np.ascontiguousarray(pose_votes, dtype=np.float32).reshape((int(k_hyp * 8), 1, 2))
        obj_points = np.repeat(ori_points[np.newaxis, :, :], k_hyp, axis=0)
        obj_points = obj_points.reshape((int(k_hyp * 8), 1, 3))
        retval, orvec, otvec, inliers = cv2.solvePnPRansac(objectPoints=obj_points,
                                                               imagePoints=est_points, cameraMatrix=K,
                                                               distCoeffs=None, rvec=None, tvec=None,
                                                               useExtrinsicGuess=False, iterationsCount=300,
                                                               reprojectionError=5.0, confidence=0.99,
                                                               flags=cv2.SOLVEPNP_ITERATIVE)
        R_est, _ = cv2.Rodrigues(orvec)
        t_est = otvec

    return name


if __name__ == '__main__':

    # ROS params
    mesh_path = ''
    msg_topic = '/camera/rgb/image_color'
    score_threshold = 0.5
    icp_threshold = 0.15
    service_name = 'returnPoses'
    try:
        model_path = rospy.get_param('/PyraPose/model_path')
    except KeyError:
        print("please set path to model! example:/home/desired/path/to/resnet_xy.h5")
    try:
        mesh_path = rospy.get_param('/PyraPose/meshes_path')
    except KeyError:
        print("please set path to meshes! example:/home/desired/path/to/meshes/")
    if rospy.has_param('/PyraPose/detection_threshold'):
        score_threshold = rospy.get_param("/PyraPose/detection_threshold")
        print('Detection threshold set to: ', score_threshold)
    if rospy.has_param('/PyraPose/image_topic'):
        msg_topic = rospy.get_param("/PyraPose/image_topic")
        print("Subscribing to msg topic: ", msg_topic)
    if rospy.has_param('/PyraPose/icp_threshold'):
        icp_threshold = rospy.get_param("/PyraPose/icp_threshold")
        print("icp threshold set to: ", icp_threshold)
    if rospy.has_param('/PyraPose/service_call'):
        service_name = rospy.get_param("/PyraPose/service_call")
        print("service call set to: ", service_name)

    model, graph = load_model(model_path)
    try:
        if rospy.get_param('/PyraPose/node_type') == 'continuous':
            print("node type set to continuous")
            pose_estimation = PoseEstimationClass(model, score_threshold, msg_topic, graph)
        elif rospy.get_param('/PyraPose/node_type') == 'service':
            print("node type set to service")
            pose_estimation = PoseEstimationServer(model, score_threshold, service_name, graph)
    except KeyError:
        print("node_type should either be continuous or service.")
    rospy.init_node('RetNetPose', anonymous=True)

    rospy.spin()






