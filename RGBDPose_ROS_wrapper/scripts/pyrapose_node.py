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
from RetNetPose import models
from RetNetPose.utils.config import read_config_file, parse_anchor_parameters
from RetNetPose.utils.eval import evaluate
from RetNetPose.utils.keras_version import check_keras_version

from RGBDPose_ROS_wrapper.srv import returnPoses

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

#model_radii = np.array([0.060, 0.064, 0.0121, 0.0127])
#objectNames = ['AC_Abdeckung', 'Deckel', 'Seite_links', 'Seite_rechts']
objectNames = ['ac_front', 'abdeckung_ac_anschluss', 'abdeckung_dc_anschluss', 'boden', 'dc_front', 'vorderfront', 'leistungsteil', 'mantel']
original_names = ['42,0405,1268', '42,0405,1249', '42,0405,1265', '42,0405,1261', '42,0405,1270', '45,0200,1402', '43,0001,1487', 'BY2,0201,4807']
# fronius_6DoF: Dc-Abdeckung=3, Front=6

ply_path = '/home/sthalham/data/MMAssist_Fronius/CAD_models/ply_models_blender_all/Seite_AC.ply'
pcd_model_1 = open3d.read_point_cloud(ply_path)
ply_path = '/home/sthalham/data/MMAssist_Fronius/CAD_models/ply_models_blender_all/AC_Abdeckung.ply'
pcd_model_2 = open3d.read_point_cloud(ply_path)
ply_path = '/home/sthalham/data/MMAssist_Fronius/CAD_models/ply_models_blender_all/DC_Abdeckung.ply'
pcd_model_3 = open3d.read_point_cloud(ply_path)
ply_path = '/home/sthalham/data/MMAssist_Fronius/CAD_models/ply_models_blender_all/Boden.ply'
pcd_model_4 = open3d.read_point_cloud(ply_path)
ply_path = '/home/sthalham/data/MMAssist_Fronius/CAD_models/ply_models_blender_all/Seite_DC.ply'
pcd_model_5 = open3d.read_point_cloud(ply_path)
ply_path = '/home/sthalham/data/MMAssist_Fronius/CAD_models/ply_models_blender_all/Front.ply'
pcd_model_6 = open3d.read_point_cloud(ply_path)
ply_path = '/home/sthalham/data/MMAssist_Fronius/CAD_models/ply_models_blender_all/Leistungsteil.ply'
pcd_model_7 = open3d.read_point_cloud(ply_path)
ply_path = '/home/sthalham/data/MMAssist_Fronius/CAD_models/ply_models_blender_all/Mantel.ply'
pcd_model_8 = open3d.read_point_cloud(ply_path)


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

        self._msg = None
        self.seq = None
        self.time = None
        self.frame_id = None
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("/object_recognition/poses", MarkerArray, queue_size=1)
        self.dims_pub = rospy.Publisher("/object_recognition/dimensions", PolygonStamped, queue_size=1)
        self.image_sub = rospy.Subscriber(topic, Image, self.callback)
        self.image_sub = rospy.Subscriber("/camera/rgb/image_color", Image, self.callback_rgb)

    def callback(self, data):
        self.seq = data.header.seq
        self.time = data.header.stamp
        self.frame_id = data.header.frame_id
        self._msg = self.bridge.imgmsg_to_cv2(data, "32FC1")
        print(self._msg.shape)

    def callback_rgb(self, data):
        self._rgb = self.bridge.imgmsg_to_cv2(data, "8UC3")

        cc_rgb = None

        det_poses = run_estimation(self._rgb, self._msg, self._model, self._score_th, self.graph, self.seq)
        #img = run_estimation(cc_rgb, self._msg, self._rgb, self._model, self._score_th, self.graph)

        self.publish_Poses(det_poses)

    def publish_Poses(self, poses):
        #msg = AlvarMarkers()
        #msg.header.seq = self.seq
        #msg.header.stamp = self.time
        #msg.header.frame_id = self.frame_id
        #for idx, pose in enumerate(poses):
        #    item = AlvarMarker()
        #    item.id = int(pose[0])
        #    item.confidence = int(pose[1] * 100.0)
        #     item.header.frame_id = original_names[int(pose[0])-1]

        #    item.pose.pose.position.x = pose[2]
        #    item.pose.pose.position.y = pose[3]
        #    item.pose.pose.position.z = pose[4]
        #    item.pose.pose.orientation.w = pose[5]
        #    item.pose.pose.orientation.x = pose[6]
        #    item.pose.pose.orientation.y = pose[7]
        #    item.pose.pose.orientation.z = pose[8]
        #    msg.markers.append(item)

        msg = MarkerArray()
        for idx, pose in enumerate(poses):
            item = Marker()
            item.header.seq = self.seq
            item.header.stamp = self.time
            item.header.frame_id = original_names[int(pose[0])-1]
            #item.header.frame_id = self.frame_id
            item.id = int(pose[0])
            item.pose.position.x = pose[2]
            item.pose.position.y = pose[3]
            item.pose.position.z = pose[4]
            item.pose.orientation.w = pose[5]
            item.pose.orientation.x = pose[6]
            item.pose.orientation.y = pose[7]
            item.pose.orientation.z = pose[8]
            item.scale.x = threeD_dims[int(pose[0]) - 1, 0]
            item.scale.y = threeD_dims[int(pose[0]) - 1, 1]
            item.scale.z = threeD_dims[int(pose[0]) - 1, 2]
            msg.markers.append(item)

        msg_dims = PolygonStamped()
        msg_dims.header.seq = self.seq
        msg_dims.header.stamp = self.time
        msg_dims.header.frame_id = self.frame_id
        for idx in range(threeD_dims.shape[0]):
            item = Point32()
            item.x = threeD_dims[idx, 0]
            item.y = threeD_dims[idx, 1]
            item.z = threeD_dims[idx, 2]
            msg_dims.polygon.points.append(item)

        self.dims_pub.publish(msg_dims)
        self.image_pub.publish(msg)


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
        self.dims_pub = rospy.Publisher("/object_recognition/dimensions", PolygonStamped, queue_size=1)

    def callback(self, data):
        #print(data)
        self.seq = data.image.header.seq
        self.time = data.image.header.stamp
        self.frame_id = data.image.header.frame_id
        self._msg = self.bridge.imgmsg_to_cv2(data.image, "32FC1")

        cc_rgb = None

        det_poses = run_estimation(cc_rgb, self._msg, self._model, self._score_th, self.graph, self.frame_id)

        msg = self.publish_Poses(det_poses)
        return msg

    def publish_Poses(self, poses):
        msg = AlvarMarkers()
        msg.header.seq = self.seq
        msg.header.stamp = self.time
        msg.header.frame_id = self.frame_id
        for idx, pose in enumerate(poses):
            item = Marker()
            item.id = int(pose[0])
            item.confidence = int(pose[1] * 100.0)
            item.header.frame_id = original_names[int(pose[0])-1]
            item.pose.pose.position.x = pose[2]
            item.pose.pose.position.y = pose[3]
            item.pose.pose.position.z = pose[4]
            item.pose.pose.orientation.w = pose[5]
            item.pose.pose.orientation.x = pose[6]
            item.pose.pose.orientation.y = pose[7]
            item.pose.pose.orientation.z = pose[8]
            msg.markers.append(item)

        msg_dims = PolygonStamped()
        msg_dims.header.seq = self.seq
        msg_dims.header.stamp = self.time
        msg_dims.header.frame_id = self.frame_id
        for idx in range(threeD_dims.shape[0]):
            item = Point32()
            item.x = threeD_dims[idx, 0]
            item.y = threeD_dims[idx, 1]
            item.z = threeD_dims[idx, 2]
            msg_dims.polygon.points.append(item)

        self.dims_pub.publish(msg_dims)

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
    poses = []
    boxes2comp = []

    image_vis = image
    print(frame_id)

    #if np.nanmax(image_dep) < 1000.0: # orbbec
    #    image_dep * 1000.0

    image_dep[image_dep > 2000.0] = 0
    scaCro = 255.0 / np.nanmax(image_dep)
    cross = np.multiply(image_dep, scaCro)
    image = cross.astype(np.uint8)
    image = np.repeat(image[:, :, np.newaxis], repeats=3, axis=2)

    #cv2.imwrite('/home/sthalham/retnetpose_image.jpg', image)

    if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

    with graph.as_default():
        boxes, boxes3D, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

    # correct boxes for image scale
    #boxes /= scale   # may be relevant at some point

    print(scores[:, :8])
    #print(labels[:, :8])

    # change to (x, y, w, h) (MS COCO standard)
    boxes[:, :, 2] -= boxes[:, :, 0]
    boxes[:, :, 3] -= boxes[:, :, 1]

    det1 = False
    det2 = False
    det3 = False
    det4 = False
    det5 = False
    det6 = False
    det7 = False
    det8 = False

    #print('new image')

    # compute predicted labels and scores
    for box3D, score, mask in zip(boxes[0], boxes3D[0], scores[0], labels[0]):
        # scores are sorted, so we can break

        if score < score_threshold:
            continue

        # ugly workaround for IoU exception
        ov_detect = False
        for bb in boxes2comp:
            ovlap = boxoverlap(box, bb)
            if ovlap > 0.5:
                ov_detect = True

        boxes2comp.append(box)
        if ov_detect is True:
            continue

        if label < 0:
            continue
        elif label == 0 and det1 == False:
            det1 = True
        elif label == 1 and det2 == False:
            det2 = True
        elif label == 2 and det3 == False:
            det3 = True
        elif label == 3 and det4 == False:
            det4 = True
        elif label == 4 and det5 == False:
            det5 = True
        elif label == 5 and det6 == False:
            det6 = True
        elif label == 6 and det7 == False:
            det7 = True
        elif label == 7 and det8 == False:
            det8 = True
        else:
            continue

        control_points = box3D

        dC = label+1

        obj_points = np.ascontiguousarray(threeD_boxes[dC - 1, :, :], dtype=np.float32)
        est_points = np.ascontiguousarray(np.asarray(control_points, dtype=np.float32).T, dtype=np.float32).reshape(
            (8, 1, 2))

        K = np.float32([fxkin, 0., cxkin, 0., fykin, cykin, 0., 0., 1.]).reshape(3, 3)
        retval, orvec, otvec, inliers = cv2.solvePnPRansac(objectPoints=obj_points,
                                                           imagePoints=est_points, cameraMatrix=K,
                                                           distCoeffs=None, rvec=None, tvec=None,
                                                           useExtrinsicGuess=False, iterationsCount=100,
                                                           reprojectionError=8.0, confidence=0.99,
                                                           flags=cv2.SOLVEPNP_ITERATIVE)
        rmat, _ = cv2.Rodrigues(orvec)

        pcd_img = create_point_cloud(image_dep, fxkin, fykin, cxkin, cykin, 0.001)
        pcd_img = pcd_img.reshape((480, 640, 3))[int(box[1]):int(box[1]+box[3]), int(box[0]):int(box[0]+box[2]), :]
        pcd_img = pcd_img.reshape((pcd_img.shape[0] * pcd_img.shape[1], 3))
        pcd_crop = open3d.PointCloud()
        pcd_crop.points = open3d.Vector3dVector(pcd_img)
        pcd_crop.paint_uniform_color(np.array([0.99, 0.0, 0.00]))
        # open3d.draw_geometries([pcd_crop, pcd_model])

        guess = np.zeros((4, 4), dtype=np.float32)
        guess[:3, :3] = rmat
        guess[:3, 3] = otvec.T
        guess[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32).T

        if dC == 1:
            pcd_model = pcd_model_1
        elif dC == 2:
            pcd_model = pcd_model_2
        elif dC == 3:
            pcd_model = pcd_model_3
        elif dC == 4:
            pcd_model = pcd_model_4
        elif dC == 5:
            pcd_model = pcd_model_5
        elif dC == 6:
            pcd_model = pcd_model_6
        elif dC == 7:
            pcd_model = pcd_model_7
        elif dC == 8:
            pcd_model = pcd_model_8
        reg_p2p = open3d.registration_icp(pcd_model, pcd_crop, 0.015, guess,
                                          open3d.TransformationEstimationPointToPoint())
        R_est = reg_p2p.transformation[:3, :3]
        t_est = reg_p2p.transformation[:3, 3]
        rot = tf3d.quaternions.mat2quat(R_est)

        pose = [dC, score, t_est[0], t_est[1], t_est[2], rot[0], rot[1], rot[2], rot[3]]
        poses.append(pose)

        font = cv2.FONT_HERSHEY_COMPLEX
        bottomLeftCornerOfText = (int(box[0]) + 5, int(box[1]) + int(box[3]) - 5)
        fontScale = 0.5
        fontColor = (25, 215, 250)
        fontthickness = 2
        lineType = 2
        gtText = objectNames[dC-1]
        print(gtText)

        fontColor2 = (0, 0, 0)
        fontthickness2 = 4
        cv2.putText(image_vis, gtText,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor2,
                    fontthickness2,
                    lineType)

        cv2.putText(image_vis, gtText,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    fontColor,
                    fontthickness,
                    lineType)

        points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)
        axisPoints, _ = cv2.projectPoints(points, R_est, t_est*1000.0, K, (0, 0, 0, 0))

        image = cv2.line(image_vis, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255, 0, 0), 3)
        image = cv2.line(image_vis, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0, 255, 0), 3)
        image = cv2.line(image_vis, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0, 0, 255), 3)

    #scaCro = 255.0 / np.nanmax(image)
    #visImg = np.multiply(image, scaCro)
    #visImg = visImg.astype(np.uint8)
    cv2.imwrite('/home/sthalham/data/MMAssist_Fronius/test_results/17092019_99/'+str(frame_id)+'.jpg', image_vis)
    #cv2.imwrite('/home/mmassist/mmassist/Detection_TUWien/retnetpose_detects.jpg', visImg)

    return poses


if __name__ == '__main__':

    # ROS params
    mesh_path = ''
    msg_topic = '/camera/depth/image_rect'
    score_threshold = 0.4
    icp_threshold = 0.15
    service_name = 'estimate_poses'
    try:
        model_path = rospy.get_param('/RetNetPose/model_path')
    except KeyError:
        print("please set path to model! example:/home/desired/path/to/resnet_xy.h5")
    try:
        mesh_path = rospy.get_param('/RetNetPose/meshes_path')
    except KeyError:
        print("please set path to meshes! example:/home/desired/path/to/meshes/")
    if rospy.has_param('/RetNetPose/detection_threshold'):
        score_threshold = rospy.get_param("/RetNetPose/detection_threshold")
        print('Detection threshold set to: ', score_threshold)
    if rospy.has_param('/RetNetPose/image_topic'):
        msg_topic = rospy.get_param("/RetNetPose/image_topic")
        print("Subscribing to msg topic: ", msg_topic)
    if rospy.has_param('/RetNetPose/icp_threshold'):
        icp_threshold = rospy.get_param("/RetNetPose/icp_threshold")
        print("icp threshold set to: ", icp_threshold)
    if rospy.has_param('/RetNetPose/service_call'):
        service_name = rospy.get_param("/RetNetPose/service_call")
        print("service call set to: ", service_name)

    model, graph = load_model(model_path)
    try:
        if rospy.get_param('/RetNetPose/node_type') == 'continuous':
            print("node type set to continuous")
            pose_estimation = PoseEstimationClass(model, score_threshold, msg_topic, graph)
        elif rospy.get_param('/RetNetPose/node_type') == 'service':
            print("node type set to service")
            pose_estimation = PoseEstimationServer(model, score_threshold, service_name, graph)
    except KeyError:
        print("node_type should either be continuous or service.")
    rospy.init_node('RetNetPose', anonymous=True)

    rospy.spin()






