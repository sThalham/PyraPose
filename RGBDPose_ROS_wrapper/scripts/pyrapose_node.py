#!/usr/bin/env python

import sys

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PoseStamped, Point, Point32, PolygonStamped
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
import json

import keras
import tensorflow as tf
import open3d

sys.path.append("/RGBDPose")
from RGBDPose import models
from RGBDPose.utils.config import read_config_file, parse_anchor_parameters
from RGBDPose.utils.eval import evaluate
from RGBDPose.utils.keras_version import check_keras_version
from RGBDPose.utils import ply_loader

from object_detector_msgs.srv import get_poses, get_posesResponse
from object_detector_msgs.msg import PoseWithConfidence

###################################
##### Global Variable Space #######
######## aka. death zone ##########
###################################



# LineMOD
#fxkin = 572.41140
#fykin = 573.57043
#cxkin = 325.26110
#cykin = 242.04899

# YCB-video
#fxkin = 1066.778
#fykin = 1067.487
#cxkin = 312.9869
#cykin = 241.3109

# our Kinect
#fxkin = 575.81573
#fykin = 575.81753
#cxkin = 314.5
#cykin = 235.5

# HSRB
fxkin = 538.391033
fykin = 538.085452
cxkin = 315.30747
cykin = 233.048356


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
    #def __init__(self, model, mesh_path, threshold, topic, graph):
    def __init__(self, model, mesh_path, threshold, topic):
        #event that will block until the info is received
        #attribute for storing the rx'd message
        self._model = model
        self._score_th = threshold
        #self.graph = graph

        self._msg = None
        self.seq = None
        self.time = None
        self.frame_id = None
        self.bridge = CvBridge()
        self.pose_pub = rospy.Publisher("/pyrapose/poses", get_posesResponse, queue_size=1)
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)

        self.threeD_boxes = np.ndarray((22, 8, 3), dtype=np.float32)
        mesh_info = os.path.join(mesh_path, 'models_info.json')
        for key, value in json.load(open(mesh_info)).items():
            fac = 0.001
            x_minus = value['min_x'] * fac
            y_minus = value['min_y'] * fac
            z_minus = value['min_z'] * fac
            x_plus = value['size_x'] * fac + x_minus
            y_plus = value['size_y'] * fac + y_minus
            z_plus = value['size_z'] * fac + z_minus
            three_box_solo = np.array([[x_plus, y_plus, z_plus],
                                   [x_plus, y_plus, z_minus],
                                   [x_plus, y_minus, z_minus],
                                   [x_plus, y_minus, z_plus],
                                   [x_minus, y_plus, z_plus],
                                   [x_minus, y_plus, z_minus],
                                   [x_minus, y_minus, z_minus],
                                   [x_minus, y_minus, z_plus]])
            self.threeD_boxes[int(key), :, :] = three_box_solo
        ply_path = os.path.join(mesh_path, 'obj_000005.ply')
        model_vsd = ply_loader.load_ply(ply_path)
        self.model_6 = open3d.PointCloud()
        self.model_6.points = open3d.Vector3dVector(model_vsd['pts'])
        ply_path = mesh_path + '/obj_000008.ply'
        model_vsd = ply_loader.load_ply(ply_path)
        self.model_9 = open3d.PointCloud()
        self.model_9.points = open3d.Vector3dVector(model_vsd['pts'])
        ply_path = mesh_path + '/obj_000009.ply'
        model_vsd = ply_loader.load_ply(ply_path)
        self.model_10 = open3d.PointCloud()
        self.model_10.points = open3d.Vector3dVector(model_vsd['pts'])
        ply_path = mesh_path + '/obj_000010.ply'
        model_vsd = ply_loader.load_ply(ply_path)
        self.model_11 = open3d.PointCloud()
        self.model_11.points = open3d.Vector3dVector(model_vsd['pts'])
        ply_path = mesh_path + '/obj_000021.ply'
        model_vsd = ply_loader.load_ply(ply_path)
        self.model_61 = open3d.PointCloud()
        self.model_61.points = open3d.Vector3dVector(model_vsd['pts'])

    def callback(self, data):
        self.seq = data.header.seq
        self.time = data.header.stamp
        self.frame_id = data.header.frame_id
        self._msg = self.bridge.imgmsg_to_cv2(data, "8UC3")

        det_objs, det_poses, det_confs = run_estimation(self._msg, self._model, self._score_th, self.threeD_boxes)#, self.seq)


        self.publish_pose(det_objs, det_poses, det_confs)

    def publish_pose(self, det_names, det_poses, det_confidences):

        msg = get_posesResponse()
        for idx in range(len(det_names)):
            item = PoseWithConfidence()
            item.name = det_names[idx] 
            item.confidence = det_confidences[idx]
            item.pose = Pose()
            det_pose = det_poses[idx]
            item.pose.position.x = det_pose[0]
            item.pose.position.y = det_pose[1]
            item.pose.position.z = det_pose[2]
            item.pose.orientation.w = det_pose[3]
            item.pose.orientation.x = det_pose[4]
            item.pose.orientation.y = det_pose[5]
            item.pose.orientation.z = det_pose[6]
            msg.poses.append(item)

        self.pose_pub.publish(msg)


class PoseEstimationServer:
    def __init__(self, model, mesh_path, threshold, topic, service_name):
        #event that will block until the info is received
        #attribute for storing the rx'd message
        self._model = model
        self._score_th = threshold

        self._msg = None
        self.seq = None
        self.time = None
        self.frame_id = None
        self.bridge = CvBridge()
        self.topic = topic
        self.pose_srv = rospy.Service(service_name, get_poses, self.callback)
        self.image_sub = rospy.Subscriber(topic, Image, self.image_callback)


        self.threeD_boxes = np.ndarray((22, 8, 3), dtype=np.float32)
        mesh_info = os.path.join(mesh_path, 'models_info.json')
        for key, value in json.load(open(mesh_info)).items():
            fac = 0.001
            x_minus = value['min_x'] * fac
            y_minus = value['min_y'] * fac
            z_minus = value['min_z'] * fac
            x_plus = value['size_x'] * fac + x_minus
            y_plus = value['size_y'] * fac + y_minus
            z_plus = value['size_z'] * fac + z_minus
            three_box_solo = np.array([[x_plus, y_plus, z_plus],
                                   [x_plus, y_plus, z_minus],
                                   [x_plus, y_minus, z_minus],
                                   [x_plus, y_minus, z_plus],
                                   [x_minus, y_plus, z_plus],
                                   [x_minus, y_plus, z_minus],
                                   [x_minus, y_minus, z_minus],
                                   [x_minus, y_minus, z_plus]])
            self.threeD_boxes[int(key), :, :] = three_box_solo
        ply_path = os.path.join(mesh_path, 'obj_000005.ply')
        model_vsd = ply_loader.load_ply(ply_path)
        self.model_6 = open3d.PointCloud()
        self.model_6.points = open3d.Vector3dVector(model_vsd['pts'])
        ply_path = mesh_path + '/obj_000008.ply'
        model_vsd = ply_loader.load_ply(ply_path)
        self.model_9 = open3d.PointCloud()
        self.model_9.points = open3d.Vector3dVector(model_vsd['pts'])
        ply_path = mesh_path + '/obj_000009.ply'
        model_vsd = ply_loader.load_ply(ply_path)
        self.model_10 = open3d.PointCloud()
        self.model_10.points = open3d.Vector3dVector(model_vsd['pts'])
        ply_path = mesh_path + '/obj_000010.ply'
        model_vsd = ply_loader.load_ply(ply_path)
        self.model_11 = open3d.PointCloud()
        self.model_11.points = open3d.Vector3dVector(model_vsd['pts'])
        ply_path = mesh_path + '/obj_000021.ply'
        model_vsd = ply_loader.load_ply(ply_path)
        self.model_61 = open3d.PointCloud()
        self.model_61.points = open3d.Vector3dVector(model_vsd['pts'])
    
    def image_callback(self, data):
        self.image = data

    def callback(self, req):
        #print(data)
        rospy.wait_for_message(self.topic, Image)
        data = self.image
        self.seq = data.header.seq
        self.time = data.header.stamp
        self.frame_id = data.header.frame_id
        self._msg = self.bridge.imgmsg_to_cv2(data, "8UC3")

        det_objs, det_poses, det_confs = run_estimation(self._msg, self._model, self._score_th, self.threeD_boxes)#, self.seq)

        msg = self.fill_pose(det_objs, det_poses, det_confs)
        return msg

    def fill_pose(self, det_names, det_poses, det_confidences):

        msg = get_posesResponse()
        for idx in range(len(det_names)):
            item = PoseWithConfidence()
            item.name = det_names[idx] 
            item.confidence = det_confidences[idx]
            item.pose = Pose()
            det_pose = det_poses[idx]
            item.pose.position.x = det_pose[0]
            item.pose.position.y = det_pose[1]
            item.pose.position.z = det_pose[2]
            item.pose.orientation.w = det_pose[3]
            item.pose.orientation.x = det_pose[4]
            item.pose.orientation.y = det_pose[5]
            item.pose.orientation.z = det_pose[6]
            msg.poses.append(item)

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
    parser.add_argument('--gpu',              help='Id of the GPU to use (as reported by nvidia-smi).')
    parser.add_argument('--config',           help='Path to a configuration parameters .ini file (only used with --convert-model).')

    return parser.parse_args(args)


def load_model(model_path):

    check_keras_version()

    #if args.gpu:
    #    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    #keras.backend.tensorflow_backend.set_session(get_session())

    anchor_params = None
    backbone = 'resnet50'

    print('Loading model, this may take a second...')
    print(model_path)
    model = models.load_model(model_path, backbone_name=backbone)
    #graph = tf.compat.v1.get_default_graph()
    model = models.convert_model(model, anchor_params=anchor_params) # convert model

    # print model summary
    print(model.summary())

    return model#, graph


#def run_estimation(image, model, score_threshold, graph, frame_id):
def run_estimation(image, model, score_threshold, threeD_boxes):
    obj_names = []
    obj_poses = []
    obj_confs = []

    image = preprocess_image(image)

    #cv2.imwrite('/home/sthalham/retnetpose_image.jpg', image)

    if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

    #with graph.as_default():
    boxes3D, scores, mask = model.predict_on_batch(np.expand_dims(image, axis=0))

    for inv_cls in range(scores.shape[2]):

        true_cat = inv_cls + 1
        true_cls = true_cat

        cls_mask = scores[0, :, inv_cls]

        cls_indices = np.where(cls_mask > score_threshold)

        if len(cls_indices[0]) < 1:
            continue

        if true_cls == 5:
                name = '006_mustard_bottle'
        elif true_cls == 8:
                name = '009_gelatin_box'
        elif true_cls == 9:
                name = '010_potted_meat_can'
        elif true_cls == 10:
                name = '011_banana'
        elif true_cls == 21:
                name = '061_foam_brick'
        else:
                continue 

        obj_names.append(name)
        obj_confs.append(np.sum(cls_mask[cls_indices[0]]))


        k_hyp = len(cls_indices[0])
        ori_points = np.ascontiguousarray(threeD_boxes[true_cls, :, :], dtype=np.float32)  # .reshape((8, 1, 3))
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
        est_pose = np.zeros((7), dtype=np.float32)
        est_pose[:3] = t_est[:, 0]
        est_pose[3:] = tf3d.quaternions.mat2quat(R_est)
        obj_poses.append(est_pose)

    return obj_names, obj_poses, obj_confs


if __name__ == '__main__':

    # ROS params
    mesh_path = ''
    msg_topic = '/camera/rgb/image_color'
    score_threshold = 0.5
    icp_threshold = 0.15
    service_name = 'get_poses'
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

    #model, graph = load_model(model_path)
    model = load_model(model_path)
    try:
        if rospy.get_param('/PyraPose/node_type') == 'continuous':
            print("node type set to continuous")
            pose_estimation = PoseEstimationClass(model, mesh_path, score_threshold, msg_topic)#, graph)
        elif rospy.get_param('/PyraPose/node_type') == 'service':
            print("node type set to service")
            pose_estimation = PoseEstimationServer(model, mesh_path, score_threshold, msg_topic, service_name)
    except KeyError:
        print("node_type should either be continuous or service.")
    rospy.init_node('PyraPose', anonymous=True)

    rospy.spin()






