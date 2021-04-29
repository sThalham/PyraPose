"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

#from pycocotools.cocoeval import COCOeval

import numpy as np
import transforms3d as tf3d
import copy
import cv2
import open3d
from ..utils import ply_loader
from .pose_error import reproj, add, adi, re, te, vsd
import yaml
import datetime
import sys
from PIL import Image
import os
import json
from ..utils.anchors import anchors_for_shape

import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."

# Import bop_renderer and bop_toolkit.
# ------------------------------------------------------------------------------
bop_renderer_path = '/home/stefan/bop_renderer/build'
sys.path.append(bop_renderer_path)

import bop_renderer

# LineMOD
fxkin = 572.41140
fykin = 573.57043
cxkin = 325.26110
cykin = 242.04899


def get_evaluation_kiru(pcd_temp_,pcd_scene_,inlier_thres,tf,final_th, model_dia):#queue
    tf_pcd =np.eye(4)
    pcd_temp_.transform(tf)

    mean_temp = np.mean(np.array(pcd_temp_.points)[:, 2])
    mean_scene = np.median(np.array(pcd_scene_.points)[:, 2])
    pcd_diff = mean_scene - mean_temp

    #open3d.draw_geometries([pcd_temp_])
    # align model with median depth of scene
    new_pcd_trans = []
    for i, point in enumerate(pcd_temp_.points):
        poi = np.asarray(point)
        poi = poi + [0.0, 0.0, pcd_diff]
        new_pcd_trans.append(poi)
    tf = np.array(tf)
    tf[2, 3] = tf[2, 3] + pcd_diff
    pcd_temp_.points = open3d.Vector3dVector(np.asarray(new_pcd_trans))
    open3d.estimate_normals(pcd_temp_, search_param=open3d.KDTreeSearchParamHybrid(
        radius=5.0, max_nn=10))

    pcd_min = mean_scene - (model_dia * 2)
    pcd_max = mean_scene + (model_dia * 2)
    new_pcd_scene = []
    for i, point in enumerate(pcd_scene_.points):
        if point[2] > pcd_min or point[2] < pcd_max:
            new_pcd_scene.append(point)
    pcd_scene_.points = open3d.Vector3dVector(np.asarray(new_pcd_scene))
    #open3d.draw_geometries([pcd_scene_])
    open3d.estimate_normals(pcd_scene_, search_param=open3d.KDTreeSearchParamHybrid(
        radius=5.0, max_nn=10))

    reg_p2p = open3d.registration.registration_icp(pcd_temp_,pcd_scene_ , inlier_thres, np.eye(4),
                                                   open3d.registration.TransformationEstimationPointToPoint(),
                                                   open3d.registration.ICPConvergenceCriteria(max_iteration = 5)) #5?
    tf = np.matmul(reg_p2p.transformation,tf)
    tf_pcd = np.matmul(reg_p2p.transformation,tf_pcd)
    pcd_temp_.transform(reg_p2p.transformation)

    open3d.estimate_normals(pcd_temp_, search_param=open3d.KDTreeSearchParamHybrid(
        radius=2.0, max_nn=30))
    #open3d.draw_geometries([pcd_scene_])
    points_unfiltered = np.asarray(pcd_temp_.points)
    last_pcd_temp = []
    for i, normal in enumerate(pcd_temp_.normals):
        if normal[2] < 0:
            last_pcd_temp.append(points_unfiltered[i, :])
    if not last_pcd_temp:
        normal_array = np.asarray(pcd_temp_.normals) * -1
        pcd_temp_.normals = open3d.Vector3dVector(normal_array)
        points_unfiltered = np.asarray(pcd_temp_.points)
        last_pcd_temp = []
        for i, normal in enumerate(pcd_temp_.normals):
            if normal[2] < 0:
                last_pcd_temp.append(points_unfiltered[i, :])
    #print(np.asarray(last_pcd_temp))
    pcd_temp_.points = open3d.Vector3dVector(np.asarray(last_pcd_temp))

    open3d.estimate_normals(pcd_temp_, search_param=open3d.KDTreeSearchParamHybrid(
        radius=5.0, max_nn=30))

    hyper_tresh = inlier_thres
    for i in range(4):
        inlier_thres = reg_p2p.inlier_rmse*2
        hyper_thres = hyper_tresh * 0.75
        if inlier_thres < 1.0:
            inlier_thres = hyper_tresh * 0.75
            hyper_tresh = inlier_thres
        reg_p2p = open3d.registration.registration_icp(pcd_temp_,pcd_scene_ , inlier_thres, np.eye(4),
                                                       open3d.registration.TransformationEstimationPointToPlane(),
                                                       open3d.registration.ICPConvergenceCriteria(max_iteration = 1)) #5?
        tf = np.matmul(reg_p2p.transformation,tf)
        tf_pcd = np.matmul(reg_p2p.transformation,tf_pcd)
        pcd_temp_.transform(reg_p2p.transformation)
    inlier_rmse = reg_p2p.inlier_rmse

    #open3d.draw_geometries([pcd_temp_, pcd_scene_])

    ##Calculate fitness with depth_inlier_th
    if(final_th>0):

        inlier_thres = final_th #depth_inlier_th*2 #reg_p2p.inlier_rmse*3
        reg_p2p = open3d.registration.registration_icp(pcd_temp_,pcd_scene_, inlier_thres, np.eye(4),
                                                       open3d.registration.TransformationEstimationPointToPlane(),
                                                       open3d.registration.ICPConvergenceCriteria(max_iteration = 1)) #5?
        tf = np.matmul(reg_p2p.transformation, tf)
        tf_pcd = np.matmul(reg_p2p.transformation, tf_pcd)
        pcd_temp_.transform(reg_p2p.transformation)

    #open3d.draw_geometries([last_pcd_temp_, pcd_scene_])from ..utils.anchors import anchors_for_shape

    if( np.abs(np.linalg.det(tf[:3,:3])-1)>0.001):
        tf[:3,0]=tf[:3,0]/np.linalg.norm(tf[:3,0])
        tf[:3,1]=tf[:3,1]/np.linalg.norm(tf[:3,1])
        tf[:3,2]=tf[:3,2]/np.linalg.norm(tf[:3,2])
    if( np.linalg.det(tf) < 0) :
        tf[:3,2]=-tf[:3,2]

    return tf,inlier_rmse,tf_pcd,reg_p2p.fitness


def toPix_array(translation):

    xpix = ((translation[:, 0] * fxkin) / translation[:, 2]) + cxkin
    ypix = ((translation[:, 1] * fykin) / translation[:, 2]) + cykin
    #zpix = translation[2] * fxkin

    return np.stack((xpix, ypix), axis=1) #, zpix]


def load_pcd(cat):
    # load meshes
    #mesh_path ="/RGBDPose/Meshes/linemod_13/"
    mesh_path = "/home/stefan/data/Meshes/linemod_13/"
    #mesh_path = "/home/sthalham/data/Meshes/linemod_13/"
    ply_path = mesh_path + 'obj_' + cat + '.ply'
    model_vsd = ply_loader.load_ply(ply_path)
    pcd_model = open3d.geometry.PointCloud()
    pcd_model.points = open3d.utility.Vector3dVector(model_vsd['pts'])
    open3d.estimate_normals(pcd_model, search_param=open3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    # open3d.draw_geometries([pcd_model])
    model_vsd_mm = copy.deepcopy(model_vsd)
    model_vsd_mm['pts'] = model_vsd_mm['pts'] * 1000.0
    #pcd_model = open3d.read_point_cloud(ply_path)
    #pcd_model = None

    return pcd_model, model_vsd, model_vsd_mm

'''
def load_pcd(cat):
    # load meshes
    #mesh_path ="/RGBDPose/Meshes/linemod_13/"
    mesh_path = "/home/stefan/data/Meshes/linemod_13/"
    #mesh_path = "/home/sthalham/data/Meshes/linemod_13/"
    ply_path = mesh_path + 'obj_' + cat + '.ply'
    model_vsd = ply_loader.load_ply(ply_path)
    pcd_model = open3d.geometry.PointCloud()
    pcd_model.points = open3d.utility.Vector3dVector(model_vsd['pts'])
    pcd_model.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    # open3d.draw_geometries([pcd_model])
    model_vsd_mm = copy.deepcopy(model_vsd)
    model_vsd_mm['pts'] = model_vsd_mm['pts'] * 1000.0
    #pcd_model = open3d.read_point_cloud(ply_path)
    #pcd_model = None

    return pcd_model, model_vsd, model_vsd_mm
'''

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
    #cloud_final[cloud_final[:,2]==0] = np.NaN

    return cloud_final


def boxoverlap(a, b):
    a = np.array([a[0], a[1], a[0] + a[2], a[1] + a[3]])
    b = np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]])

    x1 = np.amax(np.array([a[0], b[0]]))
    y1 = np.amax(np.array([a[1], b[1]]))
    x2 = np.amin(np.array([a[2], b[2]]))
    y2 = np.amin(np.array([a[3], b[3]]))

    wid = x2-x1+1
    hei = y2-y1+1
    inter = wid * hei
    aarea = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    barea = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    # intersection over union overlap
    ovlap = inter / (aarea + barea - inter)
    # set invalid entries to 0 overlap
    maskwid = wid <= 0
    maskhei = hei <= 0
    np.where(ovlap, maskwid, 0)
    np.where(ovlap, maskhei, 0)

    return ovlap


def evaluate_linemod(generator, model, threshold=0.05):
    threshold = 0.5

    #mesh_info = '/RGBDPose/Meshes/linemod_13/models_info.yml'
    mesh_info = '/home/stefan/data/Meshes/linemod_13/models_info.yml'
    #mesh_info = '/home/sthalham/data/Meshes/linemod_13/models_info.yml'

    threeD_boxes = np.ndarray((31, 8, 3), dtype=np.float32)
    model_dia = np.zeros((31), dtype=np.float32)

    for key, value in yaml.load(open(mesh_info)).items():
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
        threeD_boxes[int(key), :, :] = three_box_solo
        model_dia[int(key)] = value['diameter'] * fac

    pc1, mv1, mv1_mm = load_pcd('01')
    pc2, mv2, mv2_mm = load_pcd('02')
    pc4, mv4, mv4_mm = load_pcd('04')
    pc5, mv5, mv5_mm = load_pcd('05')
    pc6, mv6, mv6_mm = load_pcd('06')
    pc8, mv8, mv8_mm = load_pcd('08')
    pc9, mv9, mv9_mm = load_pcd('09')
    pc10, mv10, mv10_mm = load_pcd('10')
    pc11, mv11, mv11_mm = load_pcd('11')
    pc12, mv12, mv12_mm = load_pcd('12')
    pc13, mv13, mv13_mm = load_pcd('13')
    pc14, mv14, mv14_mm = load_pcd('14')
    pc15, mv15, mv15_mm = load_pcd('15')

    allPoses = np.zeros((16), dtype=np.uint32)
    truePoses = np.zeros((16), dtype=np.uint32)
    falsePoses = np.zeros((16), dtype=np.uint32)
    trueDets = np.zeros((16), dtype=np.uint32)

    for index in progressbar.progressbar(range(generator.size()), prefix='LineMOD evaluation: '):
        image_raw = generator.load_image(index)
        image = generator.preprocess_image(image_raw)
        image, scale = generator.resize_image(image)


        #image_raw_dep = generator.load_image_dep(index)
        #image_raw_dep = np.where(image_raw_dep > 0, image_raw_dep, 0.0)
        #image_raw_dep = np.multiply(image_raw_dep, 255.0 / 2000.0)
        #image_raw_dep = np.repeat(image_raw_dep[:, :, np.newaxis], 3, 2)
        #image_raw_dep = get_normal(image_raw_dep, fxkin, fykin, cxkin, cykin)
        #image_dep = generator.preprocess_image(image_raw_dep)
        #image_dep, scale = generator.resize_image(image_dep)

        image_viz = copy.deepcopy(image_raw)

        anno = generator.load_annotations(index)

        print(anno['labels'])
        if len(anno['labels']) < 1:
            continue

        if anno['labels'] == 6 or anno['labels'] == 2:
            continue

        checkLab = anno['labels']  # +1 to real_class
        for lab in checkLab:
            allPoses[int(lab) + 1] += 1

        # if len(anno['labels']) > 1:
        #    t_cat = 2
        #    obj_name = '02'
        #    ent = np.where(anno['labels'] == 1.0)
        #    t_bbox = np.asarray(anno['bboxes'], dtype=np.float32)[ent][0]
        #    t_tra = anno['poses'][ent][0][:3]
        #    t_rot = anno['poses'][ent][0][3:]

        # else:
        #    t_cat = int(anno['labels']) + 1
        #    obj_name = str(t_cat)
        #    if len(obj_name) < 2:
        #        obj_name = '0' + obj_name
        #    t_bbox = np.asarray(anno['bboxes'], dtype=np.float32)[0]
        #    t_tra = anno['poses'][0][:3]
        #    t_rot = anno['poses'][0][3:]

        #if anno['labels'][0] != 13:
        #    continue

        # run network
        images = []
        images.append(image)
        #images.append(image_dep)
        boxes3D, scores, mask = model.predict_on_batch(np.expand_dims(image, axis=0))#, np.expand_dims(image_dep, axis=0)]

        for inv_cls in range(scores.shape[2]):

            true_cat = inv_cls + 1
            # if true_cat > 5:
            #    cls = true_cat + 2
            # elif true_cat > 2:
            #    cls = true_cat + 1
            # else:
            cls = true_cat

            cls_mask = scores[0, :, inv_cls]

            cls_indices = np.where(cls_mask > threshold)
            # print(' ')
            # print('true cat: ', checkLab)
            # print('query cat: ', true_cat)
            # print(len(cls_indices[0]))
            # print(cls_mask[cls_indices])
            # print(len(cls_mask[cls_indices]))

            if cls != (checkLab + 1):
                # falsePoses[int(cls)] += 1
                continue

            if len(cls_indices[0]) < 10:
                # print('not enough inlier')
                continue
            trueDets[int(cls)] += 1

            obj_mask = mask[0, :, inv_cls]
            print(np.nanmax(obj_mask))
            cls_img = np.where(obj_mask > 0.5, 255.0, 80.0)
            cls_img = cls_img.reshape((60, 80)).astype(np.uint8)
            cls_img = np.asarray(Image.fromarray(cls_img).resize((640, 480), Image.NEAREST))
            cls_img = np.repeat(cls_img[:, :, np.newaxis], 3, 2)
            cls_img = np.where(cls_img > 254, cls_img, image_raw)
            #cv2.imwrite('/home/stefan/head_mask_viz/pred_mask_' + str(index) + '_.jpg', cls_img)

            '''
            # mask from anchors
            pot_mask = scores[0, :, inv_cls]
            pot_mask_P3 = pot_mask[:43200]
            pot_mask_P4 = pot_mask[43200:54000]
            pot_mask_P4 = pot_mask[54000:]
            print(pot_mask.shape)

            sidx = 0
            eidx = 0
            mask_P3 = np.zeros((4800), dtype=np.float32)
            for idx in range(4800):
                eidx = eidx + 9
                mask_P3[idx] = np.sum(pot_mask_P3[sidx:eidx])
                sidx = eidx

            print(mask_P3.shape)
            print(np.nanmax(mask_P3))
            mask_P3 = np.where(mask_P3 > 0.5 * (np.nanmax(mask_P3)), 255, 0)
            cls_img = mask_P3.reshape((60, 80)).astype(np.uint8)
            cls_img = cv2.resize(cls_img, (640, 480), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite('/home/stefan/RGBDPose_viz/pot_mask.jpg', cls_img)
            cls_img = np.repeat(cls_img[:, :, np.newaxis], 3, 2)
            cls_img = np.where(cls_img > 254, cls_img, image_raw)
            cv2.imwrite('/home/stefan/RGBDPose_viz/pred_mask.jpg', cls_img)
            '''

            anno_ind = np.argwhere(anno['labels'] == checkLab)
            t_tra = anno['poses'][anno_ind[0][0]][:3]
            t_rot = anno['poses'][anno_ind[0][0]][3:]
            # print(t_rot)

            BOP_obj_id = np.asarray([true_cat], dtype=np.uint32)

            # print(cls)

            if cls == 1:
                model_vsd = mv1
                model_vsd_mm = mv1_mm
            elif cls == 2:
                model_vsd = mv2
                model_vsd_mm = mv2_mm
            elif cls == 4:
                model_vsd = mv4
                model_vsd_mm = mv4_mm
            elif cls == 5:
                model_vsd = mv5
                model_vsd_mm = mv5_mm
            elif cls == 6:
                model_vsd = mv6
                model_vsd_mm = mv6_mm
            elif cls == 8:
                model_vsd = mv8
                model_vsd_mm = mv8_mm
            elif cls == 9:
                model_vsd = mv9
                model_vsd_mm = mv9_mm
            elif cls == 10:
                model_vsd = mv10
                model_vsd_mm = mv10_mm
            elif cls == 11:
                model_vsd = mv11
                model_vsd_mm = mv11_mm
            elif cls == 12:
                model_vsd = mv12
                model_vsd_mm = mv12_mm
            elif cls == 13:
                model_vsd = mv13
                model_vsd_mm = mv13_mm
            elif cls == 14:
                model_vsd = mv14
                model_vsd_mm = mv14_mm
            elif cls == 15:
                model_vsd = mv15
                model_vsd_mm = mv15_mm

            k_hyp = len(cls_indices[0])
            ori_points = np.ascontiguousarray(threeD_boxes[cls, :, :], dtype=np.float32)  # .reshape((8, 1, 3))
            K = np.float32([fxkin, 0., cxkin, 0., fykin, cykin, 0., 0., 1.]).reshape(3, 3)

            ##############################
            # pnp
            pose_votes = boxes3D[0, cls_indices, :]
            est_points = np.ascontiguousarray(pose_votes, dtype=np.float32).reshape((int(k_hyp * 8), 1, 2))
            obj_points = np.repeat(ori_points[np.newaxis, :, :], k_hyp, axis=0)
            obj_points = obj_points.reshape((int(k_hyp * 8), 1, 3))

            ###############################
            # weighted mean
            #pose_weights = np.repeat(cls_mask[cls_indices, np.newaxis], 16, axis=2)
            #pose_mean = np.mean(boxes3D[0, cls_indices, :], axis=1)
            #pose_votes = boxes3D[0, cls_indices, :] - np.repeat(pose_mean, k_hyp, axis=0)
            #pose_votes = np.multiply(pose_votes, pose_weights)
            #pose_votes = np.divide(np.sum(pose_votes, axis=1), np.sum(pose_weights, axis=1))
            #pose_votes = pose_votes + pose_mean

            #est_points = np.ascontiguousarray(pose_votes, dtype=np.float32).reshape((8, 1, 2))
            #obj_points = ori_points.reshape((8, 1, 3))

            ################################
            # 1 sigma pnp
            #pose_weights = np.repeat(cls_mask[cls_indices, np.newaxis], 16, axis=2)
            #pose_mean = np.mean(boxes3D[0, cls_indices, :], axis=1)
            #pose_var = np.var(boxes3D[0, cls_indices, :], axis=1)[0, :]
            #pose_votes = boxes3D[0, cls_indices, :] - np.repeat(pose_mean, k_hyp, axis=0)

            #filt_votes = np.empty((0, 16))
            #while filt_votes.shape[0] < 1:
            #    for set_idx in range(pose_votes.shape[1]):
            #        box_query = pose_votes[0, set_idx, :]
            #        inlier_count = np.sum(np.where(np.abs(box_query) < pose_var, 1, 0))
            #        if inlier_count > 15:
            #            filt_votes = np.concatenate([filt_votes, box_query[np.newaxis, :]], axis=0)
            #    pose_var = pose_var * 2

            #new_k = filt_votes.shape[0]

            #pose_votes = np.multiply(pose_votes, pose_weights)
            #pose_votes = np.divide(np.sum(pose_votes, axis=1), np.sum(pose_weights, axis=1))
            #pose_mean = np.repeat(pose_mean, new_k, axis=0)
            #pose_votes = filt_votes + pose_mean
            #est_points = np.ascontiguousarray(pose_votes, dtype=np.float32).reshape((int(new_k * 8), 1, 2))
            #obj_points = np.repeat(ori_points[np.newaxis, :, :], new_k, axis=0)
            #obj_points = obj_points.reshape((int(new_k * 8), 1, 3))

            ############################
            # top n hypotheses
            #top_n = 10
            #vote_scores = np.argsort(cls_mask[cls_indices])[-top_n:]
            #pose_votes = boxes3D[0, cls_indices, :][0, vote_scores, :]
            #est_points = np.ascontiguousarray(pose_votes, dtype=np.float32).reshape((int(top_n * 8), 1, 2))
            #obj_points = np.repeat(ori_points[np.newaxis, :, :], top_n, axis=0)
            #obj_points = obj_points.reshape((int(top_n * 8), 1, 3))

            retval, orvec, otvec, inliers = cv2.solvePnPRansac(objectPoints=obj_points,
                                                               imagePoints=est_points, cameraMatrix=K,
                                                               distCoeffs=None, rvec=None, tvec=None,
                                                               useExtrinsicGuess=False, iterationsCount=300,
                                                               reprojectionError=5.0, confidence=0.99,
                                                               flags=cv2.SOLVEPNP_ITERATIVE)
            R_est, _ = cv2.Rodrigues(orvec)
            t_est = otvec

            ##############################
            # pnp
            #pose_votes = boxes3D[0, cls_indices, :]
            #pose_weights = scores[0, cls_indices, :]
            #print(pose_votes.shape)
            #print(pose_weights.shape)
            #print(ori_points.shape)

            #Rt = uncertainty_pnp(pose_votes, pose_weights, ori_points, K)

            # BOP_score = -1
            #R_est = tf3d.quaternions.quat2mat(pose[3:])
            #t_est = pose[:3]
            #t_est[:2] = t_est[:2] * 0.5
            #t_est[2] = (t_est[2] / 3 + 1.0)

            BOP_R = R_est.flatten().tolist()
            BOP_t = t_est.flatten().tolist()

            # result = [int(BOP_scene_id), int(BOP_im_id), int(BOP_obj_id), float(BOP_score), BOP_R[0], BOP_R[1], BOP_R[2], BOP_R[3], BOP_R[4], BOP_R[5], BOP_R[6], BOP_R[7], BOP_R[8], BOP_t[0], BOP_t[1], BOP_t[2]]
            # result = [int(BOP_scene_id), int(BOP_im_id), int(BOP_obj_id), float(BOP_score), BOP_R, BOP_t]
            # results_image.append(result)

            # t_rot = tf3d.euler.euler2mat(t_rot[0], t_rot[1], t_rot[2])
            t_rot = tf3d.quaternions.quat2mat(t_rot)
            R_gt = np.array(t_rot, dtype=np.float32).reshape(3, 3)
            t_gt = np.array(t_tra, dtype=np.float32)

            # print(t_est)
            # print(t_gt)

            t_gt = t_gt * 0.001
            t_est = t_est.T  # * 0.001
            #print('pose: ', pose)
            #print(t_gt)
            #print(t_est)

            if cls == 10 or cls == 11:
                err_add = adi(R_est, t_est, R_gt, t_gt, model_vsd["pts"])
            else:
                err_add = add(R_est, t_est, R_gt, t_gt, model_vsd["pts"])

            colEst = (0, 0, 250)
            if err_add < model_dia[true_cat] * 0.1:
                truePoses[int(true_cat)] += 1
                colEst = (0, 204, 0)

            print(' ')
            print('error: ', err_add, 'threshold', model_dia[cls] * 0.1)


            tDbox = R_gt.dot(ori_points.T).T
            tDbox = tDbox + np.repeat(t_gt[:, np.newaxis], 8, axis=1).T
            box3D = toPix_array(tDbox)
            tDbox = np.reshape(box3D, (16))
            tDbox = tDbox.astype(np.uint16)

            eDbox = R_est.dot(ori_points.T).T
            #eDbox = eDbox + np.repeat(t_est[:, np.newaxis], 8, axis=1).T
            eDbox = eDbox + np.repeat(t_est, 8, axis=0)
            est3D = toPix_array(eDbox)
            eDbox = np.reshape(est3D, (16))
            pose = eDbox.astype(np.uint16)

            colGT = (255, 0, 0)
            #colEst = colEst = (0, 204, 0)

            image_raw = cv2.line(image_raw, tuple(tDbox[0:2].ravel()), tuple(tDbox[2:4].ravel()), colGT, 2)
            image_raw = cv2.line(image_raw, tuple(tDbox[2:4].ravel()), tuple(tDbox[4:6].ravel()), colGT, 2)
            image_raw = cv2.line(image_raw, tuple(tDbox[4:6].ravel()), tuple(tDbox[6:8].ravel()), colGT,
                             2)
            image_raw = cv2.line(image_raw, tuple(tDbox[6:8].ravel()), tuple(tDbox[0:2].ravel()), colGT,
                             2)
            image_raw = cv2.line(image_raw, tuple(tDbox[0:2].ravel()), tuple(tDbox[8:10].ravel()), colGT,
                             2)
            image_raw = cv2.line(image_raw, tuple(tDbox[2:4].ravel()), tuple(tDbox[10:12].ravel()), colGT,
                             2)
            image_raw = cv2.line(image_raw, tuple(tDbox[4:6].ravel()), tuple(tDbox[12:14].ravel()), colGT,
                             2)
            image_raw = cv2.line(image_raw, tuple(tDbox[6:8].ravel()), tuple(tDbox[14:16].ravel()), colGT,
                             2)
            image_raw = cv2.line(image_raw, tuple(tDbox[8:10].ravel()), tuple(tDbox[10:12].ravel()),
                             colGT,
                             2)
            image_raw = cv2.line(image_raw, tuple(tDbox[10:12].ravel()), tuple(tDbox[12:14].ravel()),
                             colGT,
                             2)
            image_raw = cv2.line(image_raw, tuple(tDbox[12:14].ravel()), tuple(tDbox[14:16].ravel()),
                             colGT,
                             2)
            image_raw = cv2.line(image_raw, tuple(tDbox[14:16].ravel()), tuple(tDbox[8:10].ravel()),
                             colGT,
                             2)

            image_raw = cv2.line(image_raw, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), colEst, 2)
            image_raw = cv2.line(image_raw, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), colEst, 2)
            image_raw = cv2.line(image_raw, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), colEst, 2)
            image_raw = cv2.line(image_raw, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), colEst, 2)
            image_raw = cv2.line(image_raw, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), colEst, 2)
            image_raw = cv2.line(image_raw, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), colEst, 2)
            image_raw = cv2.line(image_raw, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), colEst, 2)
            image_raw = cv2.line(image_raw, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), colEst, 2)
            image_raw = cv2.line(image_raw, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()), colEst,
                             2)
            image_raw = cv2.line(image_raw, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), colEst,
                             2)
            image_raw = cv2.line(image_raw, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), colEst,
                             2)
            image_raw = cv2.line(image_raw, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), colEst,
                             2)


            #hyp_mask = np.zeros((640, 480), dtype=np.float32)
            #for idx in range(k_hyp):
            #    hyp_mask[int(est_points[idx, 0, 0]), int(est_points[idx, 0, 1])] += 1

            #hyp_mask = np.transpose(hyp_mask)
            #hyp_mask = (hyp_mask * (255.0 / np.nanmax(hyp_mask))).astype(np.uint8)

            #image_raw[:, :, 0] = np.where(hyp_mask > 0, 0, image_raw[:, :, 0])
            #image_raw[:, :, 1] = np.where(hyp_mask > 0, 0, image_raw[:, :, 1])
            #image_raw[:, :, 2] = np.where(hyp_mask > 0, hyp_mask, image_raw[:, :, 2])


            '''
            idx = 0
            for i in range(k_hyp):
                image = cv2.circle(image, (est_points[idx, 0, 0], est_points[idx, 0, 1]), 3, (13, 243, 207), -2)
                image = cv2.circle(image, (est_points[idx+1, 0, 0], est_points[idx+1, 0, 1]), 3, (251, 194, 213), -2)
                image = cv2.circle(image, (est_points[idx+2, 0, 0], est_points[idx+2, 0, 1]), 3, (222, 243, 41), -2)
                image = cv2.circle(image, (est_points[idx+3, 0, 0], est_points[idx+3, 0, 1]), 3, (209, 31, 201), -2)
                image = cv2.circle(image, (est_points[idx+4, 0, 0], est_points[idx+4, 0, 1]), 3, (8, 62, 53), -2)
                image = cv2.circle(image, (est_points[idx+5, 0, 0], est_points[idx+5, 0, 1]), 3, (13, 243, 207), -2)
                image = cv2.circle(image, (est_points[idx+6, 0, 0], est_points[idx+6, 0, 1]), 3, (215, 41, 29), -2)
                image = cv2.circle(image, (est_points[idx+7, 0, 0], est_points[idx+7, 0, 1]), 3, (78, 213, 16), -2)
                idx = idx+8
            '''
            '''
            max_x = int(np.max(est_points[:, :, 0]) + 5)
            min_x = int(np.min(est_points[:, :, 0]) - 5)
            max_y = int(np.max(est_points[:, :, 1]) + 5)
            min_y = int(np.min(est_points[:, :, 1]) - 5)

            print(max_x, min_x, max_y, min_y)

            image_crop = image_raw[min_y:max_y, min_x:max_x, :]
            image_crop = cv2.resize(image_crop, None, fx=2, fy=2)
            '''

            name = '/home/stefan/PyraPose_viz/img_' + str(index) + '.jpg'
            cv2.imwrite(name, image_raw)
            #print('break')

    recall = np.zeros((16), dtype=np.float32)
    precision = np.zeros((16), dtype=np.float32)
    detections = np.zeros((16), dtype=np.float32)
    for i in range(1, (allPoses.shape[0])):
        recall[i] = truePoses[i] / allPoses[i]
        precision[i] = truePoses[i] / (truePoses[i] + falsePoses[i])
        detections[i] = trueDets[i] / allPoses[i]

        if np.isnan(recall[i]):
            recall[i] = 0.0
        if np.isnan(precision[i]):
            precision[i] = 0.0

        print('CLS: ', i)
        print('true detections: ', detections[i])
        print('recall: ', recall[i])
        print('precision: ', precision[i])

    recall_all = np.sum(recall[1:]) / 13.0
    precision_all = np.sum(precision[1:]) / 13.0
    detections_all = np.sum(detections[1:]) / 13.0
    print('ALL: ')
    print('true detections: ', detections_all)
    print('recall: ', recall_all)
    print('precision: ', precision_all)


def reannotate_linemod(generator, model, threshold=0.5):

    #mesh_info = '/RGBDPose/Meshes/linemod_13/models_info.yml'
    mesh_info = generator.get_mesh_info()
    mesh_path = "/home/stefan/data/Meshes/linemod_13_invert/"
    #mesh_info = '/home/sthalham/data/Meshes/linemod_13/models_info.yml'

    threeD_boxes = np.ndarray((31, 8, 3), dtype=np.float32)
    model_dia = np.zeros((31), dtype=np.float32)

    for key, value in yaml.load(open(mesh_info)).items():
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
        threeD_boxes[int(key), :, :] = three_box_solo
        model_dia[int(key)] = value['diameter'] * fac

    '''
    print('loading with open3d')

    pc1, mv1, mv1_mm = load_pcd('01')
    pc2, mv2, mv2_mm = load_pcd('02')
    pc4, mv4, mv4_mm = load_pcd('04')
    pc5, mv5, mv5_mm = load_pcd('05')
    pc6, mv6, mv6_mm = load_pcd('06')
    pc8, mv8, mv8_mm = load_pcd('08')
    pc9, mv9, mv9_mm = load_pcd('09')
    pc10, mv10, mv10_mm = load_pcd('10')
    pc11, mv11, mv11_mm = load_pcd('11')
    pc12, mv12, mv12_mm = load_pcd('12')
    pc13, mv13, mv13_mm = load_pcd('13')
    pc14, mv14, mv14_mm = load_pcd('14')
    pc15, mv15, mv15_mm = load_pcd('15')
    '''

    ren = bop_renderer.Renderer()
    print(ren)
    ren.init(640, 480)
    mesh_id = 1
    categories = []

    for mesh_now in os.listdir(mesh_path):
        mesh_path_now = os.path.join(mesh_path, mesh_now)
        print(mesh_path_now)
        if mesh_now[-4:] != '.ply':
            continue
        mesh_id = int(mesh_now[-6:-4])
        ren.add_object(mesh_id, mesh_path_now)
        print(mesh_id)
        categories.append(mesh_id)
        mesh_id += 1

    print('Meshes loaded')

    allPoses = np.zeros((16), dtype=np.uint32)
    truePoses = np.zeros((16), dtype=np.uint32)
    falsePoses = np.zeros((16), dtype=np.uint32)
    trueDets = np.zeros((16), dtype=np.uint32)

    annoID_val = 0

    now = datetime.datetime.now()
    dateT = str(now)
    dict = {"info": {
        "description": "linemod",
        "version": "1.0",
        "year": 2021,
        "contributor": "Stefan Thalhammer",
        "date_created": dateT
    },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }

    anchor_params = anchors_for_shape((480, 640))

    for index in progressbar.progressbar(range(generator.size()), prefix='LineMOD evaluation: '):
        image_raw = generator.load_image(index)
        image = generator.preprocess_image(image_raw)
        image, scale = generator.resize_image(image)

        anno = generator.load_annotations(index)

        checkLab = anno['labels']  # +1 to real_class
        for lab in checkLab:
            allPoses[int(lab) + 1] += 1

        img_id = generator.get_img_id(index)
        iname = generator.get_image_path(index)
        data_dir_path = generator.get_path()
        #mask_path = os.path.join(data_path, iname[:-4], '_mask.png')
        mask_path = os.path.join(data_dir_path, 'images/pseudo', iname[:-4] + '_mask.png')
        img_path = os.path.join(data_dir_path, 'images/pseudo', iname[:-4] + '_rgb.png')

        #if len(anno['labels']) < 1:
        #    continue

        #checkLab = anno['labels']  # +1 to real_class
        #for lab in checkLab:
        #    allPoses[int(lab) + 1] += 1

        boxes3D, scores, mask = model.predict_on_batch(np.expand_dims(image, axis=0))#, np.expand_dims(image_dep, axis=0)])

        mask_ind = 0
        mask_img = np.zeros((480, 640), dtype=np.uint8)

        est_poses = []
        est_classes = []
        zeds = []

        for inv_cls in range(scores.shape[2]):

            true_cat = inv_cls + 1
            cls = true_cat

            cls_mask = scores[0, :, inv_cls]

            cls_indices = np.where(cls_mask > threshold)

            # multi-instance intermezzo for annotating
            pos_anchors = anchor_params[cls_indices, :]
            ind_anchors = cls_indices[0]
            pos_anchors = pos_anchors[0]
            per_obj_hyps = []

            while pos_anchors.shape[0] > 0:
                # make sure to separate objects
                start_i = np.random.randint(pos_anchors.shape[0])
                obj_ancs = [pos_anchors[start_i]]
                obj_inds = [ind_anchors[start_i]]
                pos_anchors = np.delete(pos_anchors, start_i, axis=0)
                ind_anchors = np.delete(ind_anchors, start_i, axis=0)
                # print('ind_anchors: ', ind_anchors)
                same_obj = True
                while same_obj == True:
                    # update matrices based on iou
                    same_obj = False
                    indcs2rm = []
                    for adx in range(pos_anchors.shape[0]):
                        # loop through anchors
                        box_b = pos_anchors[adx, :]
                        if not np.all((box_b > 0)):  # need x_max or y_max here? maybe irrelevant due to positivity
                            indcs2rm.append(adx)
                            continue
                        for qdx in range(len(obj_ancs)):
                            # loop through anchors belonging to instance
                            iou = boxoverlap(obj_ancs[qdx], box_b)
                            if iou > 0.4:
                                # print('anc_anchors: ', pos_anchors)
                                # print('ind_anchors: ', ind_anchors)
                                # print('adx: ', adx)
                                obj_ancs.append(box_b)
                                obj_inds.append(ind_anchors[adx])
                                indcs2rm.append(adx)
                                same_obj = True
                                break
                        if same_obj == True:
                            break

                    # print('pos_anchors: ', pos_anchors.shape)
                    # print('ind_anchors: ', len(ind_anchors))
                    # print('indcs2rm: ', indcs2rm)
                    pos_anchors = np.delete(pos_anchors, indcs2rm, axis=0)
                    ind_anchors = np.delete(ind_anchors, indcs2rm, axis=0)

                per_obj_hyps.append(obj_inds)

            for per_ins_indices in per_obj_hyps:

                if len(per_ins_indices) < 10:
                    # print('not enough inlier')
                    continue

                '''
                # all anchors heuristics
                true_anchors = anchor_params[per_ins_indices, :]
                a_min_x = np.nanmin(true_anchors[:, 0])
                a_min_y = np.nanmin(true_anchors[:, 1])
                a_max_x = np.nanmax(true_anchors[:, 2])
                a_max_y = np.nanmax(true_anchors[:, 3])
#
                os_box_w = a_max_x - a_min_x
                os_box_h = a_max_y - a_min_y
                obj_center_x = a_min_x + 0.5 * os_box_w
                obj_center_y = a_min_y + 0.5 * os_box_h
                # handling image boundaries
                #if a_min_x < 1:
                #    obj_center_x = a_min_x + 0.25 * os_box_w
                #elif a_min_x > 638:
                #    obj_center_x = a_min_x + 0.75 * os_box_w
                #if a_min_y < 1:
                #    obj_center_y = a_min_y + 0.25 * os_box_h
                #elif a_min_x > 478:
                #    obj_center_y = a_min_y + 0.75 * os_box_h
                box_min_x = int(obj_center_x - 0.25 * os_box_w)
                box_max_x = int(obj_center_x + 0.25 * os_box_w)
                box_min_y = int(obj_center_y - 0.25 * os_box_h)
                box_max_y = int(obj_center_y + 0.25 * os_box_h)
                obj_bb = [int(box_min_x), int(box_min_y), int(box_max_x - box_min_x), int(box_max_y - box_min_y)]
                area = obj_bb[2] * obj_bb[3]

                # object masking
                obj_mask = mask[0, :, inv_cls]
                box_min_x_lr = box_min_x * 0.125
                box_max_x_lr = box_max_x * 0.125
                box_min_y_lr = box_min_y * 0.125
                box_max_y_lr = box_max_y * 0.125
                obj_mask = obj_mask.reshape((60, 80))
                where_true = np.where(obj_mask > 0.5)
                #obj_y = where_true[0][(where_true[0] > box_min_y_lr) & (where_true[0] < box_max_y_lr)]
                #obj_x = where_true[1][(where_true[1] > box_min_x_lr) & (where_true[0] < box_max_x_lr)]
                obj_y = (where_true[0] > box_min_y_lr) & (where_true[0] < box_max_y_lr)
                obj_x = (where_true[1] > box_min_x_lr) & (where_true[1] < box_max_x_lr)
                #print('obj_y: ', where_true[0], box_min_y_lr, box_max_y_lr)
                #print('obj_x: ', where_true[1], box_min_x_lr, box_max_x_lr)
                both_true = np.where(np.logical_and(obj_x, obj_y))[0]

                obj_y = where_true[0][both_true]
                obj_x = where_true[1][both_true]

                obj_mask = np.zeros((60, 80))
                for pdx, y in enumerate(obj_y):
                    #print(y, obj_x[pdx])
                    obj_mask[y, obj_x[pdx]] = 1

                ##########
                # correct box with min and max of segmentation
                ##########

                #obj_mask = np.asarray(Image.fromarray(obj_mask).resize((640, 480), Image.NEAREST))
                #mask_id = mask_ind + 1
                #mask_img = np.where(obj_mask > 0.5, mask_id, mask_img)
                #mask_ind += 1
                '''
                anno_ind = np.argwhere(anno['labels'] == checkLab)
                t_tra = anno['poses'][anno_ind[0][0]][:3]
                t_rot = anno['poses'][anno_ind[0][0]][3:]

                k_hyp = len(per_ins_indices)
                #k_hyp = len(cls_indices[0])
                ori_points = np.ascontiguousarray(threeD_boxes[cls, :, :], dtype=np.float32)  # .reshape((8, 1, 3))
                K = np.float32([fxkin, 0., cxkin, 0., fykin, cykin, 0., 0., 1.]).reshape(3, 3)

                ##############################
                # pnp
                pose_votes = boxes3D[0, per_ins_indices, :]
                est_points = np.ascontiguousarray(pose_votes, dtype=np.float32).reshape((int(k_hyp * 8), 1, 2))
                obj_points = np.repeat(ori_points[np.newaxis, :, :], k_hyp, axis=0)
                obj_points = obj_points.reshape((int(k_hyp * 8), 1, 3))
                # single pnp
                #random_hyp = np.random.choice(per_ins_indices)
                #pose_votes = boxes3D[0, random_hyp, :]
                #est_points = np.ascontiguousarray(pose_votes, dtype=np.float32).reshape((8, 1, 2))
                #obj_points = np.repeat(ori_points[np.newaxis, :, :], k_hyp, axis=0)
                #obj_points = ori_points.reshape((8, 1, 3))

                retval, orvec, otvec, inliers = cv2.solvePnPRansac(objectPoints=obj_points,
                                                               imagePoints=est_points, cameraMatrix=K,
                                                               distCoeffs=None, rvec=None, tvec=None,
                                                               useExtrinsicGuess=False, iterationsCount=300,
                                                               reprojectionError=5.0, confidence=0.99,
                                                               flags=cv2.SOLVEPNP_EPNP)
                R_est, _ = cv2.Rodrigues(orvec)
                t_est = otvec
                est_pose = np.eye((4))
                est_pose[:3, :3] = R_est
                est_pose[:3, 3] = t_est[:, 0]

                est_poses.append(est_pose)
                est_classes.append(cls)
                zeds.append(t_est[2])

                '''
                # Eval
                if cls == 1:
                    model_vsd = mv1
                elif cls == 2:
                    model_vsd = mv2
                elif cls == 4:
                    model_vsd = mv4
                elif cls == 5:
                    model_vsd = mv5
                elif cls == 6:
                    model_vsd = mv6
                elif cls == 8:
                    model_vsd = mv8
                elif cls == 9:
                    model_vsd = mv9
                elif cls == 10:
                    model_vsd = mv10
                elif cls == 11:
                    model_vsd = mv11
                elif cls == 12:
                    model_vsd = mv12
                elif cls == 13:
                    model_vsd = mv13
                elif cls == 14:
                    model_vsd = mv14
                elif cls == 15:
                    model_vsd = mv15

                if cls == (checkLab + 1):

                    t_rot = tf3d.quaternions.quat2mat(t_rot)
                    R_gt = np.array(t_rot, dtype=np.float32).reshape(3, 3)
                    t_gt = np.array(t_tra, dtype=np.float32)
                    t_gt = t_gt * 0.001
                    t_est = t_est.T  # * 0.001

                    if cls == 10 or cls == 11:
                        err_add = adi(R_est, t_est, R_gt, t_gt, model_vsd["pts"])
                    else:
                        err_add = add(R_est, t_est, R_gt, t_gt, model_vsd["pts"])

                    if err_add < model_dia[true_cat] * 0.1:
                        truePoses[int(true_cat)] += 1

                    print(' ')
                    print('error: ', err_add, 'threshold', model_dia[cls] * 0.1)
                '''

        # sort poses with depth
        zeds = np.asarray(zeds, dtype=np.float32)
        if zeds.shape[0] == 0:
            continue
        low2high = np.argsort(zeds[:, 0])
        high2low = low2high[::-1]
        full_seg = []

        for v_idx in low2high:

            pose = est_poses[v_idx]
            R_est = pose[:3, :3]
            t_est = pose[:3, 3]
            cls = est_classes[v_idx]

            # light, render and append
            R_list = R_est.flatten().tolist()
            t_list = t_est.flatten().tolist()
            #light_pose = [np.random.rand() * 3 - 1.0, np.random.rand() * 2 - 1.0, 0.0]
            #light_color = [1.0, 1.0, 1.0]
            #light_ambient_weight = np.random.rand()
            #light_diffuse_weight = 0.75 + np.random.rand() * 0.25
            #light_spec_weight = 0.25 + np.random.rand() * 0.25
            #light_spec_shine = np.random.rand() * 3.0

            light_pose = [0.0, 0.0, 0.0]
            light_color = [1.0, 1.0, 1.0]
            light_ambient_weight = 0.8
            light_diffuse_weight = 0.8
            light_spec_weight = 0.2
            light_spec_shine = 1.0

            ren.set_light(light_pose, light_color, light_ambient_weight, light_diffuse_weight, light_spec_weight,
                              light_spec_shine)
            ren.render_object(cls, R_list, t_list, fxkin, fykin, cxkin, cykin)
            rgb_img = ren.get_color_image(cls)

            # pose for anno
            tra = t_est
            rot = tf3d.quaternions.mat2quat(R_est)
            pose = [tra[0], tra[1], tra[2], rot[0], rot[1], rot[2], rot[3]]
            trans = np.asarray([pose[0], pose[1], pose[2]], dtype=np.float32)
            R = tf3d.quaternions.quat2mat(np.asarray([pose[3], pose[4], pose[5], pose[6]], dtype=np.float32))
            tDbox = R.reshape(3, 3).dot(threeD_boxes[cls, :, :].T).T
            tDbox = tDbox + np.repeat(trans[np.newaxis, :], 8, axis=0)
            box3D = toPix_array(tDbox)
            box3D = np.reshape(box3D, (16))
            tDbox = box3D.astype(np.uint16)
            box3D = box3D.tolist()
            pose = [np.asscalar(pose[0]), np.asscalar(pose[1]), np.asscalar(pose[2]),
                    np.asscalar(pose[3]), np.asscalar(pose[4]), np.asscalar(pose[5]), np.asscalar(pose[6])]

            # mask
            mask_abs = np.where(rgb_img > 0, 1, 0)
            mask_all = np.where(mask_img == 0, 1, 0)
            mask_all = np.repeat(mask_all[:, :, np.newaxis], repeats=3, axis=2)
            mask_vis = np.where(np.logical_and(mask_abs == 1, mask_all == 1), 1, 0)
            image_raw = np.where(mask_vis == 1, rgb_img * 0.5 + image_raw * 0.5, image_raw)
            mask_id = mask_ind + 1
            mask_img = np.where(mask_vis[:, :, 0] == 1, mask_id, mask_img)
            mask_ind += 1

            # bbox + area
            mask_inds = np.where(mask_vis == 1)
            if mask_inds[0].shape[0] < 1:
                continue
            box_min_x = np.nanmin(mask_inds[1])
            box_min_y = np.nanmin(mask_inds[0])
            box_max_x = np.nanmax(mask_inds[1])
            box_max_y = np.nanmax(mask_inds[0])
            if box_min_x == 0 or box_min_y == 0 or box_max_x == 639 or box_max_y == 479:
                continue
            obj_bb = [int(box_min_x), int(box_min_y), int(box_max_x - box_min_x), int(box_max_y - box_min_y)]
            area = obj_bb[2] * obj_bb[3]

            image_raw = np.where(mask_vis > 0, rgb_img*0.5 + image_raw*0.5, image_raw)
            '''
            if cls == 10 or cls == 11:
                err_add = adi(R_est, t_est, R_gt, t_gt, model_vsd["pts"])
            else:
                err_add = add(R_est, t_est, R_gt, t_gt, model_vsd["pts"])

            if err_add < model_dia[true_cat] * 0.1:
                truePoses[int(true_cat)] += 1
            '''
            # create annotation

            colGT = (np.random.random()*255.0, np.random.random()*255.0, np.random.random()*255.0)
            image_raw = cv2.line(image_raw, tuple(tDbox[0:2].ravel()), tuple(tDbox[2:4].ravel()), colGT, 1)
            image_raw = cv2.line(image_raw, tuple(tDbox[2:4].ravel()), tuple(tDbox[4:6].ravel()), colGT, 1)
            image_raw = cv2.line(image_raw, tuple(tDbox[4:6].ravel()), tuple(tDbox[6:8].ravel()), colGT,
                                 1)
            image_raw = cv2.line(image_raw, tuple(tDbox[6:8].ravel()), tuple(tDbox[0:2].ravel()), colGT,
                                 1)
            image_raw = cv2.line(image_raw, tuple(tDbox[0:2].ravel()), tuple(tDbox[8:10].ravel()), colGT,
                                 1)
            image_raw = cv2.line(image_raw, tuple(tDbox[2:4].ravel()), tuple(tDbox[10:12].ravel()), colGT,
                                 1)
            image_raw = cv2.line(image_raw, tuple(tDbox[4:6].ravel()), tuple(tDbox[12:14].ravel()), colGT,
                                 1)
            image_raw = cv2.line(image_raw, tuple(tDbox[6:8].ravel()), tuple(tDbox[14:16].ravel()), colGT,
                                 1)
            image_raw = cv2.line(image_raw, tuple(tDbox[8:10].ravel()), tuple(tDbox[10:12].ravel()),
                                 colGT,
                                 1)
            image_raw = cv2.line(image_raw, tuple(tDbox[10:12].ravel()), tuple(tDbox[12:14].ravel()),
                                 colGT,
                                 1)
            image_raw = cv2.line(image_raw, tuple(tDbox[12:14].ravel()), tuple(tDbox[14:16].ravel()),
                                 colGT,
                                 1)
            image_raw = cv2.line(image_raw, tuple(tDbox[14:16].ravel()), tuple(tDbox[8:10].ravel()),
                                 colGT,
                                 1)
            bb = np.array(obj_bb)
            cv2.rectangle(image_raw, (int(bb[0]), int(bb[1])), (int(bb[0] + bb[2]), int(bb[1] + bb[3])),
                          (255, 255, 255), 2)

            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (int(bb[0]), int(bb[1]))
            fontScale = 1
            lineType = 2
            gtText = str(cls)

            fontColor2 = (255, 255, 255)
            fontthickness2 = 3
            cv2.putText(image_raw, gtText,
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor2,
                        fontthickness2,
                        lineType)


            annoID_val = annoID_val + 1
            tempTA = {
                "id": annoID_val,
                "image_id": img_id,
                "category_id": cls,
                "bbox": obj_bb,
                "pose": pose,
                "segmentation": box3D,
                "mask_id": mask_id,
                "area": area,
                "iscrowd": 0,
                "feature_visibility": 1.0
            }
            dict["annotations"].append(tempTA)
            #hyp_mask = np.zeros((640, 480), dtype=np.float32)
            #for idx in range(k_hyp):
            #    hyp_mask[int(est_points[idx, 0, 0]), int(est_points[idx, 0, 1])] += 1
            #hyp_mask = np.transpose(hyp_mask)
            #hyp_mask = (hyp_mask * (255.0 / np.nanmax(hyp_mask))).astype(np.uint8)
            #image_raw[:, :, 0] = np.where(hyp_mask > 0, 0, image_raw[:, :, 0])
            #image_raw[:, :, 1] = np.where(hyp_mask > 0, 0, image_raw[:, :, 1])
            #image_raw[:, :, 2] = np.where(hyp_mask > 0, hyp_mask, image_raw[:, :, 2])
            '''
            idx = 0
            for i in range(k_hyp):
                image_raw = cv2.circle(image_raw, (est_points[idx, 0, 0], est_points[idx, 0, 1]), 3, (13, 243, 207), -2)
                image_raw = cv2.circle(image_raw, (est_points[idx+1, 0, 0], est_points[idx+1, 0, 1]), 3, (251, 194, 213), -2)
                image_raw = cv2.circle(image_raw, (est_points[idx+2, 0, 0], est_points[idx+2, 0, 1]), 3, (222, 243, 41), -2)
                image_raw = cv2.circle(image_raw, (est_points[idx+3, 0, 0], est_points[idx+3, 0, 1]), 3, (209, 31, 201), -2)
                image_raw = cv2.circle(image_raw, (est_points[idx+4, 0, 0], est_points[idx+4, 0, 1]), 3, (8, 62, 53), -2)
                image_raw = cv2.circle(image_raw, (est_points[idx+5, 0, 0], est_points[idx+5, 0, 1]), 3, (13, 243, 207), -2)
                image_raw = cv2.circle(image_raw, (est_points[idx+6, 0, 0], est_points[idx+6, 0, 1]), 3, (215, 41, 29), -2)
                image_raw = cv2.circle(image_raw, (est_points[idx+7, 0, 0], est_points[idx+7, 0, 1]), 3, (78, 213, 16), -2)
                idx = idx+8
            
            max_x = int(np.max(est_points[:, :, 0]) + 5)
            min_x = int(np.min(est_points[:, :, 0]) - 5)
            max_y = int(np.max(est_points[:, :, 1]) + 5)
            min_y = int(np.min(est_points[:, :, 1]) - 5)

            print(max_x, min_x, max_y, min_y)

            image_crop = image_raw[min_y:max_y, min_x:max_x, :]
            image_crop = cv2.resize(image_crop, None, fx=2, fy=2)
            '''

        cv2.imwrite(mask_path, mask_img)
        cv2.imwrite(img_path, image_raw)

        name = '/home/stefan/PyraPose_viz/self_anno_' + str(index) + '.jpg'
        cls_img = np.repeat(mask_img[:, :, np.newaxis], 3, 2)
        img_con = np.concatenate([image_raw, (cls_img * 10.0).astype(np.uint8)], axis=1)
        cv2.imwrite(name, img_con)

        tempTL = {
            "url": "https://bop.felk.cvut.cz/home/",
            "id": img_id,
            "name": iname,
        }
        dict["licenses"].append(tempTL)

        tempTV = {
            "license": 2,
            "url": "https://bop.felk.cvut.cz/home/",
            "file_name": iname,
            "height": 480,
            "width": 640,
            "fx": fxkin,
            "fy": fykin,
            "cx": cxkin,
            "cy": cykin,
            "date_captured": dateT,
            "id": img_id,
        }
        dict["images"].append(tempTV)

    for s in range(1, 16):
        objName = str(s)
        tempC = {
            "id": s,
            "name": objName,
            "supercategory": "object"
        }
        dict["categories"].append(tempC)

    valAnno = generator.get_anno_path()
    valAnno = valAnno[:-8] + 'pseudo.json'

    with open(valAnno, 'w') as fpT:
        json.dump(dict, fpT)

    '''
    recall = np.zeros((16), dtype=np.float32)
    precision = np.zeros((16), dtype=np.float32)
    detections = np.zeros((16), dtype=np.float32)
    for i in range(1, (allPoses.shape[0])):
        recall[i] = truePoses[i] / allPoses[i]
        precision[i] = truePoses[i] / (truePoses[i] + falsePoses[i])
        detections[i] = trueDets[i] / allPoses[i]

        if np.isnan(recall[i]):
            recall[i] = 0.0
        if np.isnan(precision[i]):
            precision[i] = 0.0

    recall_all = np.sum(recall[1:]) / 13.0
    precision_all = np.sum(precision[1:]) / 13.0
    detections_all = np.sum(detections[1:]) / 13.0

    return recall_all
    '''

