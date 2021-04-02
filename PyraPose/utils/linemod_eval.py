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
import sys
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.covariance import MinCovDet


import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."


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

    #open3d.draw_geometries([last_pcd_temp_, pcd_scene_])

    if( np.abs(np.linalg.det(tf[:3,:3])-1)>0.001):
        tf[:3,0]=tf[:3,0]/np.linalg.norm(tf[:3,0])
        tf[:3,1]=tf[:3,1]/np.linalg.norm(tf[:3,1])
        tf[:3,2]=tf[:3,2]/np.linalg.norm(tf[:3,2])
    if( np.linalg.det(tf) < 0) :
        tf[:3,2]=-tf[:3,2]

    return tf,inlier_rmse,tf_pcd,reg_p2p.fitness


def plot_ellipses(img, means, covars, weights):
    norm_weights = (weights * (255/np.nanmax(weights))).astype(np.uint8)
    for n in range(means.shape[0]):
        center_coordinates = means[n]
        axesLength = covars[n]
        angle = 0
        startAngle = 0
        endAngle = 360
        color = norm_weights[n]
        color = (int(color), int(color), int(color))
        print(color, type(color))
        thickness = 1
        img = cv2.ellipse(img=img, box=(center_coordinates.astype(np.uint16), np.ceil(axesLength).astype(np.uint16), angle), color=color, thickness=thickness)

    return img


def toPix_array(translation):

    xpix = ((translation[:, 0] * fxkin) / translation[:, 2]) + cxkin
    ypix = ((translation[:, 1] * fykin) / translation[:, 2]) + cykin
    #zpix = translation[2] * fxkin

    return np.stack((xpix, ypix), axis=1) #, zpix]


def to3D_array(translation):

    xpix = ((translation[:, 0] * fxkin) / translation[:, 2]) + cxkin
    ypix = ((translation[:, 1] * fykin) / translation[:, 2]) + cykin
    #zpix = translation[2] * fxkin

    return np.stack((xpix, ypix), axis=1) #, zpix]

'''
def load_pcd(cat):
    # load meshes
    #mesh_path ="/RGBDPose/Meshes/linemod_13/"
    mesh_path = "/home/stefan/data/Meshes/linemod_13/"
    #mesh_path = "/home/sthalham/data/Meshes/linemod_13/"
    ply_path = mesh_path + 'obj_' + cat + '.ply'
    model_vsd = ply_loader.load_ply(ply_path)
    pcd_model = open3d.PointCloud()
    pcd_model.points = open3d.Vector3dVector(model_vsd['pts'])
    open3d.estimate_normals(pcd_model, search_param=open3d.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    # open3d.draw_geometries([pcd_model])
    model_vsd_mm = copy.deepcopy(model_vsd)
    model_vsd_mm['pts'] = model_vsd_mm['pts'] * 1000.0
    #pcd_model = open3d.read_point_cloud(ply_path)

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

    return pcd_model, model_vsd, model_vsd_mm


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

        if len(anno['labels']) < 1:
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

        if anno['labels'][0] == 2 or anno['labels'][0] == 6:
            continue

        # run network
        boxes3D, scores, mask = model.predict_on_batch(np.expand_dims(image, axis=0))#, np.expand_dims(image_dep, axis=0)])

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

            #obj_mask = mask[0, :, inv_cls]
            #print(np.nanmax(obj_mask))
            #cls_img = np.where(obj_mask > 0.5, 255.0, 80.0)
            #cls_img = cls_img.reshape((60, 80)).astype(np.uint8)
            #cls_img = np.asarray(Image.fromarray(cls_img).resize((640, 480), Image.NEAREST))
            #cls_img = np.repeat(cls_img[:, :, np.newaxis], 3, 2)
            #cls_img = np.where(cls_img > 254, cls_img, image_raw)
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

            ##########################
            # process every hypothesis separately
            ##########################
            ##########################
            # keep space
            # ----------------------
            # median box deviation for thresholding
            #med_box_dev = np.median(box_devs)
            #below_median = np.where(box_devs < med_box_dev)
            #filtered_hyps = cls_indices[0][below_median]
            # ---------------------------
            # gaussian process single hypotheses choice
            #X = np.arange(16)
            #X = np.repeat(X[:, np.newaxis], repeats=len(cls_indices[0]), axis=1)
            #X = X.flatten()
            #X = np.expand_dims(X, axis=1)
            #col_mean = np.mean(pose_votes, axis=0)
            #col_std = np.std(pose_votes, axis=0)
            #row_mean = np.repeat(col_mean[np.newaxis, :], repeats=len(cls_indices[0]), axis=0)
            #row_std = np.repeat(col_std[np.newaxis, :], repeats=len(cls_indices[0]), axis=0)
            #pose_votes = (pose_votes - row_mean) / row_std
            #y = pose_votes.T.flatten()
            #gpr = GaussianProcessRegressor().fit(X, y)
            #y_mean, y_std = gpr.predict(np.expand_dims(np.arange(16), axis=1), return_std=True)
            #y_mean = (y_mean * col_std) + col_mean
            # -------------------------------

            true_pose = 0
            top_error = 1
            k_hyp = len(cls_indices[0])
            errors = []
            box_devs = []
            loc_scores = []
            '''
            for hypdx in range(k_hyp):
                ori_points = np.ascontiguousarray(threeD_boxes[cls, :, :], dtype=np.float32)  # .reshape((8, 1, 3))
                K = np.float32([fxkin, 0., cxkin, 0., fykin, cykin, 0., 0., 1.]).reshape(3, 3)

                pose_votes = boxes3D[0, cls_indices[0][hypdx], :]
                est_points = np.ascontiguousarray(pose_votes, dtype=np.float32).reshape((8, 1, 2))

                retval, orvec, otvec, inliers = cv2.solvePnPRansac(objectPoints=ori_points,
                                                                   imagePoints=est_points, cameraMatrix=K,
                                                                   distCoeffs=None, rvec=None, tvec=None,
                                                                   useExtrinsicGuess=False, iterationsCount=300,
                                                                   reprojectionError=5.0, confidence=0.99,
                                                                   flags=cv2.SOLVEPNP_EPNP)
                R_est, _ = cv2.Rodrigues(orvec)
                t_est = otvec

                t_rot_q = tf3d.quaternions.quat2mat(t_rot)
                R_gt = np.array(t_rot_q, dtype=np.float32).reshape(3, 3)
                t_gt = np.array(t_tra, dtype=np.float32)

                t_gt = t_gt * 0.001
                t_est = t_est.T

                if cls == 10 or cls == 11:
                    err_add = adi(R_est, t_est, R_gt, t_gt, model_vsd["pts"])
                else:
                    err_add = add(R_est, t_est, R_gt, t_gt, model_vsd["pts"])

                if err_add < model_dia[true_cat] * 0.1:
                    true_pose = 1
                if err_add < top_error:
                    top_error = err_add
                errors.append(err_add)

                # box deviatio in camera frame
                tDbox = R_gt.dot(ori_points.T).T
                tDbox = tDbox + np.repeat(t_gt[:, np.newaxis], 8, axis=1).T
                box3D = toPix_array(tDbox)
                tDbox = np.reshape(box3D, (16))
                tDbox = tDbox.astype(np.uint16)

                eDbox = R_est.dot(ori_points.T).T
                # eDbox = eDbox + np.repeat(t_est[:, np.newaxis], 8, axis=1).T
                eDbox = eDbox + np.repeat(t_est, 8, axis=0)
                est3D = toPix_array(eDbox)
                eDbox = np.reshape(est3D, (16))
                pose = eDbox.astype(np.uint16)

                box_dev = np.linalg.norm(eDbox - pose_votes)

                # box deviation in object frame
                #box_cam_m = pose_votes.reshape((2, 8))
                #homo_series = np.repeat([0], repeats=8)
                #box_cam_m = np.concatenate([box_cam_m, homo_series.T], axis=0)

                box_devs.append(box_dev)
                loc_scores.append(cls_mask[cls_indices[0][hypdx]])

                colGT = (255, 0, 0)
                colEst = (0, 0, 215)
                #if err_add < model_dia[true_cat] * 0.1:
                #    colEst = (255, 255, 255)

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
            '''

            #pose_votes = boxes3D[0, cls_indices[0], :]
            #box_devs = np.asarray(box_devs)
            #gpr = GaussianProcessRegressor().fit(pose_votes, box_devs)
            #y_mean, y_std = gpr.predict(pose_votes, return_std=True)
            #print(y_std)
            #for idx in range(len(y_std)):
            #    y_std[idx, idx] = 0.0
            #y_std = y_std * (1/np.nanmax(y_std))

            #cov_sum = np.sum(y_std, axis=0)
            #print(cov_sum)
            #print(np.nanmax(cov_sum), np.argmax(cov_sum))
            #max_cov = np.argmax(cov_sum)
            #print(np.nanmax(y_std), np.argmax(y_std))

            #x_ax = np.arange(len(cls_indices[0]))
            #plt.plot(x_ax, y_mean, 'r', lw=1)#, zorder=9)
            #plt.fill(np.concatenate([x_ax, x_ax[::-1]], axis=0),
            #         np.concatenate([y_mean - (1.9600 * y_std * 50000.0),
            #                         (y_mean + (1.9600 * y_std * 50000.0))[::-1]], axis=0),
            #         alpha=.5, fc='b', ec='None', label='95% confidence interval')
            #plt.show()

            #X = np.arange(16)
            #X = np.repeat(X[:, np.newaxis], repeats=len(cls_indices[0]), axis=1)
            #X = X.flatten()
            #X = np.expand_dims(X, axis=1)
            #print(X.shape)

            #print(pose_votes.shape)
            #print(pose_votes[:5, 0])
            # normalize y
            pose_votes = boxes3D[0, cls_indices[0], :]
            col_mean = np.mean(pose_votes, axis=0)
            col_std = np.std(pose_votes, axis=0)
            row_mean = np.repeat(col_mean[np.newaxis, :], repeats=len(cls_indices[0]), axis=0)
            row_std = np.repeat(col_std[np.newaxis, :], repeats=len(cls_indices[0]), axis=0)
            pose_votes = (pose_votes - row_mean) / row_std

            y = pose_votes.T.flatten()
            #print(y[:5])

            #gpr = GaussianProcessRegressor().fit(X, y)
            #y_mean, y_std = gpr.predict(np.expand_dims(np.arange(16), axis=1), return_std=True)
            #print(y_std.shape, y_std)
            #y_mean = (y_mean * col_std) + col_mean
            #plt.imshow(y_std, cmap='hot', interpolation='nearest')
            #plt.show()

            #min_hyp = np.argmin(y_std)
            #print(min_hyp)

            #x_ax = np.arange(16)
            #x_ax = np.arange(len(cls_indices[0]))
            #col_min = np.nanmin(pose_votes, axis=0)
            #col_max = np.nanmax(pose_votes, axis=0)
            #row_min = np.repeat(col_min[np.newaxis, :], repeats=len(cls_indices[0]), axis=0)
            #row_max = np.repeat(col_max[np.newaxis, :], repeats=len(cls_indices[0]), axis=0)

            #col_mean = np.mean(pose_votes, axis=0)
            #col_std = np.std(pose_votes, axis=0)
            #row_mean = np.repeat(col_mean[np.newaxis, :], repeats=len(cls_indices[0]), axis=0)
            #row_std = np.repeat(col_std[np.newaxis, :], repeats=len(cls_indices[0]), axis=0)

            #for idx in range(len(cls_indices[0])):
            #    plt.plot(x_ax, (pose_votes[idx, :] - row_mean[idx, :]) / row_std[idx, :], 'r', lw=1)#, zorder=9)
                #plt.plot(x_ax, (pose_votes[idx, :] - row_min[idx, :]) * (1 / (row_max[idx, :] - row_min[idx, :])), 'r',lw=1)  # , zorder=9)
            #plt.show()

            #y_samples = gpr.sample_y(pose_votes, 10)
            #plt.plot(X_, y_samples, lw=1)
            #plt.scatter(X[:, 0], y, c='r', s=50, zorder=10, edgecolors=(0, 0, 0))
            #plt.xlim(0, 5)
            #plt.ylim(0, 8)
            #plt.title("Posterior (kernel: %s)\n Log-Likelihood: %.3f"
            #          % (gp.kernel_, gp.log_marginal_likelihood(gp.kernel_.theta)),
            #          fontsize=12)

            '''
            # Visulize errors loc_scores, box_devs
            norm_thres = np.asarray(model_dia[true_cat] * 0.1) * (1 / max(np.asarray(errors)))
            errors_norm = np.asarray(errors) * (1 / np.nanmax(np.asarray(errors)))
            box_devs = np.asarray(box_devs) * (1 / np.nanmax(np.asarray(box_devs)))
            #loc_scores = np.asarray(loc_scores)
            #hyps = 20
            sorting = np.argsort(errors)
            #if len(sorting) > hyps:
            #    sorting = sorting[-hyps:]
            #else:
            #    pass
            filtered_errors = errors_norm[sorting]
            filtered_box_devs = box_devs[sorting]
            #filtered_loc_scores = loc_scores[sorting]
            x_axis = range(len(errors_norm))
            plt.plot(x_axis, filtered_errors, 'bo:', linewidth=2, markersize=3, label="errors per hypothesis")
            plt.plot(x_axis, filtered_box_devs, 'rv-.', linewidth=2, markersize=3, label="box_devs")
            #plt.plot(x_axis, filtered_loc_scores, 'k*--', linewidth=2, markersize=3, label="loc scores")
            plt.axhline(y=norm_thres, color='r', linestyle='-', label="ADD-0.1 threshold")
            #plt.legend(loc="upper left")
            plt.show()
            '''
            '''
            min_box_dev = np.argmin(np.array(box_devs))

            #ori_points = np.ascontiguousarray(threeD_boxes[cls, :, :], dtype=np.float32)  # .reshape((8, 1, 3))
            #K = np.float32([fxkin, 0., cxkin, 0., fykin, cykin, 0., 0., 1.]).reshape(3, 3)

            #pose_votes = boxes3D[0, cls_indices[0][min_box_dev], :]
            #pose_votes = y_mean
            #est_points = np.ascontiguousarray(pose_votes, dtype=np.float32).reshape((8, 1, 2))
            '''

            '''
            k_hyp = len(cls_indices[0])
            ori_points = np.ascontiguousarray(threeD_boxes[cls, :, :], dtype=np.float32)  # .reshape((8, 1, 3))
            K = np.float32([fxkin, 0., cxkin, 0., fykin, cykin, 0., 0., 1.]).reshape(3, 3)

            ##############################
            # pnp
            #pose_votes = boxes3D[0, cls_indices, :]
            k_hyp = 2
            ori_points = np.ascontiguousarray(threeD_boxes[cls, :, :], dtype=np.float32)  # .reshape((8, 1, 3))
            K = np.float32([fxkin, 0., cxkin, 0., fykin, cykin, 0., 0., 1.]).reshape(3, 3)
            true_pose = 0
            for pdx in range(pose_votes.shape[0]):
                est_points = np.ascontiguousarray(pose_votes[pdx, :], dtype=np.float32).reshape((8, 1, 2))
                obj_points = np.repeat(ori_points[np.newaxis, :, :], k_hyp, axis=0)
                obj_points = obj_points.reshape((int(k_hyp * 8), 1, 3))

                retval, orvec, otvec, inliers = cv2.solvePnPRansac(objectPoints=ori_points,
                                                                   imagePoints=est_points, cameraMatrix=K,
                                                                   distCoeffs=None, rvec=None, tvec=None,
                                                                   useExtrinsicGuess=False, iterationsCount=300,
                                                                   reprojectionError=5.0, confidence=0.99,
                                                                   flags=cv2.SOLVEPNP_EPNP)
                R_est, _ = cv2.Rodrigues(orvec)
                t_est = otvec

                t_rot_q = tf3d.quaternions.quat2mat(t_rot)
                R_gt = np.array(t_rot_q, dtype=np.float32).reshape(3, 3)
                t_gt = np.array(t_tra, dtype=np.float32)

                t_gt = t_gt * 0.001
                t_est = t_est.T

                if cls == 10 or cls == 11:
                    err_add = adi(R_est, t_est, R_gt, t_gt, model_vsd["pts"])
                else:
                    err_add = add(R_est, t_est, R_gt, t_gt, model_vsd["pts"])

                if err_add < model_dia[true_cat] * 0.1:
                    true_pose = 1
            truePoses[int(true_cat)] += true_pose

            #plt.axhline(y=(err_add * (1 / max(np.asarray(errors)))), color='b', linestyle='-')
            #plt.show()

            #truePoses[int(true_cat)] += true_pose
            print(' ')
            print('error: ', err_add, 'threshold', model_dia[cls] * 0.1)
            #print('errors: ', errors)
            #print('box_devs: ', box_devs)

            '''

            '''
            # BGMM 2d: n=8, sample 200 hyps from cluster with highest separate weight: 61.13
            # variational hypotheses choice
            min_samples = 200
            components = 8
            if pose_votes.shape[0] < components:
                components = 2
            ori_points = np.ascontiguousarray(threeD_boxes[cls, :, :], dtype=np.float32)
            bgm0 = BayesianGaussianMixture(n_components=components, random_state=0).fit(pose_votes[:, :2])
            indices_0 = np.argmax(bgm0.weights_)
            pose_samples, sample_labels = bgm0.sample(min_samples)
            min_wp = np.where(sample_labels==indices_0)
            votes0 = (pose_samples[min_wp] * col_std[:2]) + col_mean[:2]
            corr0 = np.repeat(ori_points[0, :][np.newaxis, :], repeats=len(min_wp[0]), axis=0)
            #means = (bgm.means_ * col_std[:2]) + col_mean[:2]
            #covars = np.concatenate([bgm.covariances_[:, 0, 0][:, np.newaxis], bgm.covariances_[:, 1, 1][:, np.newaxis]], axis=1)
            #covars = (covars * col_std[:2]) * 3.0
            #weights = bgm.weights_
            bgm1 = BayesianGaussianMixture(n_components=components, random_state=0).fit(pose_votes[:, 2:4])
            indices_1 = np.argmax(np.array(bgm1.weights_))
            pose_samples, sample_labels = bgm1.sample(min_samples)
            min_wp = np.where(sample_labels == indices_1)
            votes1 = (pose_samples[min_wp] * col_std[2:4]) + col_mean[2:4]
            corr1 = np.repeat(ori_points[1, :][np.newaxis, :], repeats=len(min_wp[0]), axis=0)
            bgm2 = BayesianGaussianMixture(n_components=components, random_state=0).fit(pose_votes[:, 4:6])
            indices_2 = np.argmax(np.array(bgm2.weights_))
            pose_samples, sample_labels = bgm2.sample(min_samples)
            min_wp = np.where(sample_labels == indices_2)
            votes2 = (pose_samples[min_wp] * col_std[4:6]) + col_mean[4:6]
            corr2 = np.repeat(ori_points[2, :][np.newaxis, :], repeats=len(min_wp[0]), axis=0)
            bgm3 = BayesianGaussianMixture(n_components=components, random_state=0).fit(pose_votes[:, 6:8])
            indices_3 = np.argmax(np.array(bgm3.weights_))
            pose_samples, sample_labels = bgm3.sample(min_samples)
            min_wp = np.where(sample_labels == indices_3)
            votes3 = (pose_samples[min_wp] * col_std[6:8]) + col_mean[6:8]
            corr3 = np.repeat(ori_points[3, :][np.newaxis, :], repeats=len(min_wp[0]), axis=0)
            bgm4 = BayesianGaussianMixture(n_components=components, random_state=0).fit(pose_votes[:, 8:10])
            indices_4 = np.argmax(np.array(bgm4.weights_))
            pose_samples, sample_labels = bgm4.sample(min_samples)
            min_wp = np.where(sample_labels == indices_4)
            votes4 = (pose_samples[min_wp] * col_std[8:10] ) + col_mean[8:10]
            corr4 = np.repeat(ori_points[4, :][np.newaxis, :], repeats=len(min_wp[0]), axis=0)
            bgm5 = BayesianGaussianMixture(n_components=components, random_state=0).fit(pose_votes[:, 10:12])
            indices_5 = np.argmax(np.array(bgm5.weights_))
            pose_samples, sample_labels = bgm5.sample(min_samples)
            min_wp = np.where(sample_labels == indices_5)
            votes5 = (pose_samples[min_wp] * col_std[10:12]) + col_mean[10:12]
            corr5 = np.repeat(ori_points[5, :][np.newaxis, :], repeats=len(min_wp[0]), axis=0)
            bgm6 = BayesianGaussianMixture(n_components=components, random_state=0).fit(pose_votes[:, 12:14])
            indices_6 = np.argmax(np.array(bgm6.weights_))
            # sample from components
            pose_samples, sample_labels = bgm6.sample(min_samples)
            min_wp = np.where(sample_labels == indices_6)
            votes6 = (pose_samples[min_wp] * col_std[12:14]) + col_mean[12:14]
            corr6 = np.repeat(ori_points[6, :][np.newaxis, :], repeats=len(min_wp[0]), axis=0)
            bgm7 = BayesianGaussianMixture(n_components=components, random_state=0).fit(pose_votes[:, 14:16])
            indices_7 = np.argmax(np.array(bgm7.weights_))
            pose_samples, sample_labels = bgm7.sample(min_samples)
            min_wp = np.where(sample_labels == indices_7)
            votes7 = (pose_samples[min_wp] * col_std[14:16]) + col_mean[14:16]
            # sample from hyps
            #sample_labels = bgm7.predict(pose_votes[:, 14:16])
            #min_wp = np.where(sample_labels == indices_7)
            #votes7 = (pose_votes[min_wp, 14:16] * col_std[14:16]) + col_mean[14:16]
            #if min_samples > len(min_wp[0]):
            #    min_samples = len(min_wp[0])
            corr7 = np.repeat(ori_points[7, :][np.newaxis, :], repeats=len(min_wp[0]), axis=0)
            variational_votes = np.concatenate([votes0, votes1, votes2, votes3, votes4, votes5, votes6, votes7], axis=0)
            est_points = np.ascontiguousarray(variational_votes, dtype=np.float32).reshape(
                (variational_votes.shape[0], 1, 2))
            variational_corrs = np.concatenate([corr0, corr1, corr2, corr3, corr4, corr5, corr6, corr7], axis=0)
            obj_points = variational_corrs.reshape((variational_corrs.shape[0], 1, 3))
            '''
            '''
            # sample from hyps
            #votes_s = np.random.choice(np.arange(votes0.shape[1]), size=min_samples)
            #votes0 = votes0[:, votes_s, :]
            #votes_s = np.random.choice(np.arange(votes1.shape[1]), size=min_samples)
            #votes1 = votes1[:, votes_s, :]
            #votes_s = np.random.choice(np.arange(votes2.shape[1]), size=min_samples)
            #votes2 = votes2[:, votes_s, :]
            #votes_s = np.random.choice(np.arange(votes3.shape[1]), size=min_samples)
            #votes3 = votes3[:, votes_s, :]
            #votes_s = np.random.choice(np.arange(votes4.shape[1]), size=min_samples)
            #votes4 = votes4[:, votes_s, :]
            #votes_s = np.random.choice(np.arange(votes5.shape[1]), size=min_samples)
            #votes5 = votes5[:, votes_s, :]
            #votes_s = np.random.choice(np.arange(votes6.shape[1]), size=min_samples)
            #votes6 = votes6[:, votes_s, :]
            #votes_s = np.random.choice(np.arange(votes7.shape[1]), size=min_samples)
            #votes7 = votes7[:, votes_s, :]
            corr0 = np.repeat(ori_points[0, :][np.newaxis, :], repeats=min_samples, axis=0)
            corr1 = np.repeat(ori_points[1, :][np.newaxis, :], repeats=min_samples, axis=0)
            corr2 = np.repeat(ori_points[2, :][np.newaxis, :], repeats=min_samples, axis=0)
            corr3 = np.repeat(ori_points[3, :][np.newaxis, :], repeats=min_samples, axis=0)
            corr4 = np.repeat(ori_points[4, :][np.newaxis, :], repeats=min_samples, axis=0)
            corr5 = np.repeat(ori_points[5, :][np.newaxis, :], repeats=min_samples, axis=0)
            corr6 = np.repeat(ori_points[6, :][np.newaxis, :], repeats=min_samples, axis=0)
            corr7 = np.repeat(ori_points[7, :][np.newaxis, :], repeats=min_samples, axis=0)
            variational_votes = np.concatenate([votes0, votes1, votes2, votes3, votes4, votes5, votes6, votes7], axis=1)
            est_points = np.ascontiguousarray(variational_votes, dtype=np.float32).transpose((1, 0, 2))
            '''

            '''
            # SMM 2d: n=8, sample 200 hyps from cluster with highest separate weight: 61.13
            # variational hypotheses choice
            min_samples = 200
            components = 8
            if pose_votes.shape[0] < components:
                components = 2
            ori_points = np.ascontiguousarray(threeD_boxes[cls, :, :], dtype=np.float32)
            bgm0 = smm.smm.SMM(n_components=components, random_state=0).fit(pose_votes[:, :2])
            indices_0 = np.argmax(bgm0.weights_)
            # resample
            #pose_samples, sample_labels = bgm0.sample(min_samples)
            #min_wp = np.where(sample_labels==indices_0)
            #votes0 = (pose_samples[min_wp] * col_std[:2]) + col_mean[:2]
            # choice
            sample_labels = bgm0.predict(pose_votes[:, :2])
            min_wp = np.where(sample_labels == indices_0)
            votes0 = (pose_votes[min_wp, :2] * col_std[:2]) + col_mean[:2]
            corr0 = np.repeat(ori_points[0, :][np.newaxis, :], repeats=len(min_wp[0]), axis=0)
            #means = (bgm.means_ * col_std[:2]) + col_mean[:2]
            #covars = np.concatenate([bgm.covariances_[:, 0, 0][:, np.newaxis], bgm.covariances_[:, 1, 1][:, np.newaxis]], axis=1)
            #covars = (covars * col_std[:2]) * 3.0
            #weights = bgm.weights_
            bgm1 = smm.smm.SMM(n_components=components, random_state=0).fit(pose_votes[:, 2:4])
            indices_1 = np.argmax(np.array(bgm1.weights_))
            #pose_samples, sample_labels = bgm1.sample(min_samples)
            #min_wp = np.where(sample_labels == indices_1)
            #votes1 = (pose_samples[min_wp] * col_std[2:4]) + col_mean[2:4]
            # choice
            sample_labels = bgm1.predict(pose_votes[:, 2:4])
            min_wp = np.where(sample_labels == indices_1)
            votes1 = (pose_votes[min_wp, 2:4] * col_std[2:4]) + col_mean[2:4]
            corr1 = np.repeat(ori_points[1, :][np.newaxis, :], repeats=len(min_wp[0]), axis=0)
            bgm2 = smm.smm.SMM(n_components=components, random_state=0).fit(pose_votes[:, 4:6])
            indices_2 = np.argmax(np.array(bgm2.weights_))
            #pose_samples, sample_labels = bgm2.sample(min_samples)
            #min_wp = np.where(sample_labels == indices_2)
            #votes2 = (pose_samples[min_wp] * col_std[4:6]) + col_mean[4:6]
            # choice
            sample_labels = bgm2.predict(pose_votes[:, 4:6])
            min_wp = np.where(sample_labels == indices_2)
            votes2 = (pose_votes[min_wp, 4:6] * col_std[4:6]) + col_mean[4:6]
            corr2 = np.repeat(ori_points[2, :][np.newaxis, :], repeats=len(min_wp[0]), axis=0)
            bgm3 = smm.smm.SMM(n_components=components, random_state=0).fit(pose_votes[:, 6:8])
            indices_3 = np.argmax(np.array(bgm3.weights_))
            #pose_samples, sample_labels = bgm3.sample(min_samples)
            #min_wp = np.where(sample_labels == indices_3)
            #votes3 = (pose_samples[min_wp] * col_std[6:8]) + col_mean[6:8]
            # choice
            sample_labels = bgm3.predict(pose_votes[:, 6:8])
            min_wp = np.where(sample_labels == indices_3)
            votes3 = (pose_votes[min_wp, 6:8] * col_std[6:8]) + col_mean[6:8]
            corr3 = np.repeat(ori_points[3, :][np.newaxis, :], repeats=len(min_wp[0]), axis=0)
            bgm4 = smm.smm.SMM(n_components=components, random_state=0).fit(pose_votes[:, 8:10])
            indices_4 = np.argmax(np.array(bgm4.weights_))
            #pose_samples, sample_labels = bgm4.sample(min_samples)
            #min_wp = np.where(sample_labels == indices_4)
            #votes4 = (pose_samples[min_wp] * col_std[8:10] ) + col_mean[8:10]
            # choice
            sample_labels = bgm4.predict(pose_votes[:, 8:10])
            min_wp = np.where(sample_labels == indices_4)
            votes4 = (pose_votes[min_wp, 8:10] * col_std[8:10]) + col_mean[8:10]
            corr4 = np.repeat(ori_points[4, :][np.newaxis, :], repeats=len(min_wp[0]), axis=0)
            bgm5 = smm.smm.SMM(n_components=components, random_state=0).fit(pose_votes[:, 10:12])
            indices_5 = np.argmax(np.array(bgm5.weights_))
            #pose_samples, sample_labels = bgm5.sample(min_samples)
            #min_wp = np.where(sample_labels == indices_5)
            #votes5 = (pose_samples[min_wp] * col_std[10:12]) + col_mean[10:12]
            # choice
            sample_labels = bgm5.predict(pose_votes[:, 10:12])
            min_wp = np.where(sample_labels == indices_5)
            votes5 = (pose_votes[min_wp, 10:12] * col_std[10:12]) + col_mean[10:12]
            corr5 = np.repeat(ori_points[5, :][np.newaxis, :], repeats=len(min_wp[0]), axis=0)
            bgm6 = smm.smm.SMM(n_components=components, random_state=0).fit(pose_votes[:, 12:14])
            indices_6 = np.argmax(np.array(bgm6.weights_))
            # sample from components
            #pose_samples, sample_labels = bgm6.sample(min_samples)
            #min_wp = np.where(sample_labels == indices_6)
            #votes6 = (pose_samples[min_wp] * col_std[12:14]) + col_mean[12:14]
            # choice
            sample_labels = bgm6.predict(pose_votes[:, 12:14])
            min_wp = np.where(sample_labels == indices_6)
            votes6 = (pose_votes[min_wp, 12:14] * col_std[12:14]) + col_mean[12:14]
            corr6 = np.repeat(ori_points[6, :][np.newaxis, :], repeats=len(min_wp[0]), axis=0)
            bgm7 = smm.smm.SMM(n_components=components, random_state=0).fit(pose_votes[:, 14:16])
            indices_7 = np.argmax(np.array(bgm7.weights_))
            #pose_samples, sample_labels = bgm7.sample(min_samples)
            #min_wp = np.where(sample_labels == indices_7)
            #votes7 = (pose_samples[min_wp] * col_std[14:16]) + col_mean[14:16]
            # choice
            sample_labels = bgm7.predict(pose_votes[:, 14:16])
            min_wp = np.where(sample_labels == indices_7)
            votes7 = (pose_votes[min_wp, 14:16] * col_std[14:16]) + col_mean[14:16]
            corr7 = np.repeat(ori_points[7, :][np.newaxis, :], repeats=len(min_wp[0]), axis=0)
            print(votes0.shape)
            variational_votes = np.concatenate([votes0, votes1, votes2, votes3, votes4, votes5, votes6, votes7], axis=1)
            est_points = np.ascontiguousarray(variational_votes, dtype=np.float32).reshape(
                (variational_votes.shape[1], 1, 2))
            variational_corrs = np.concatenate([corr0, corr1, corr2, corr3, corr4, corr5, corr6, corr7], axis=0)
            obj_points = variational_corrs.reshape((variational_corrs.shape[0], 1, 3))
            '''

            '''
            # hyps of cluster with lowest weighted likelihood
            #components = int(pose_votes.shape[0] / 3)
            #ori_points = np.ascontiguousarray(threeD_boxes[cls, :, :], dtype=np.float32)
            #bgm = GaussianMixture(n_components=components, random_state=0).fit(pose_votes)
            #sample_labels = bgm.predict(pose_votes)
            #bgm_scores = bgm.score_samples(pose_votes)
            #comp_scores = []
            #for cdx in range(components):
            #    indices = np.where(sample_labels==cdx)
            #    comp_scores.append(np.sum(bgm_scores[indices]) / len(indices[0]))
            #min_wp = np.argmin(np.array(comp_scores))
            #hyp_indices = np.where(sample_labels==min_wp)
            #filtered_votes = boxes3D[0, cls_indices[0][hyp_indices], :]
            '''
            '''
            # BGMM with 16 dimensions
            #components = int(pose_votes.shape[0] / 6)
            components = 8
            if pose_votes.shape[0] < 8:
                components = 2
            ori_points = np.ascontiguousarray(threeD_boxes[cls, :, :], dtype=np.float32)
            bgm = BayesianGaussianMixture(n_components=components, random_state=0).fit(pose_votes)
            sample_labels = bgm.predict(pose_votes)
            min_wp = np.argmax(np.array(bgm.weights_))
            hyp_indices = np.where(sample_labels == min_wp)
            filtered_votes = boxes3D[0, cls_indices[0][hyp_indices], :]
            '''

            #col_mean = np.mean(filtered_votes, axis=0)
            #col_std = np.std(filtered_votes, axis=0)
            #row_mean = np.repeat(col_mean[np.newaxis, :], repeats=len(hyp_indices[0]), axis=0)
            #row_std = np.repeat(col_std[np.newaxis, :], repeats=len(hyp_indices[0]), axis=0)
            #filtered_votes = (filtered_votes - row_mean) / row_std

            # Minimum Covariance Determinant
            #ori_points = np.ascontiguousarray(threeD_boxes[cls, :, :], dtype=np.float32)


            # create emission sequences
            #sequence = []
            #components = 8
            #print(pose_votes.shape)
            #for idx in range(pose_votes.shape[0]):
            #    sequence.append(pose_votes[idx,:])
            #if pose_votes.shape[0] < 8:
            #    components = 2
            #hmm = bayesian_hmm.HDPHMM(sequence, sticky=False)
            #hmm.initialise(k=components)
            #results = hmm.mcmc(n=500, burn_in=100, ncores=3, save_every=10, verbose=True)
            #hmm.print_probabilities()
            #print(results)
            #print(hmm.n_features)
            #print(hmm.n_components)
            #print(hmm.means)

            #norm_thres = np.asarray(model_dia[true_cat] * 0.1) * (1 / max(np.asarray(errors)))
            #errors_norm = np.asarray(errors) * (1 / np.nanmax(np.asarray(errors)))
            #bgm_scores_norm = np.asarray(bgm_scores) * (1 / np.nanmax(np.asarray(bgm_scores)))

            #sorting = np.argsort(bgm_scores)
            #filtered_errors = errors_norm[sorting]
            #filtered_scores = bgm_scores_norm[sorting]
            #x_axis = range(len(errors_norm))
            #plt.plot(x_axis, filtered_errors, 'bo:', linewidth=2, markersize=3, label="errors per hypothesis")
            #plt.plot(x_axis, filtered_scores, 'rv-', linewidth=2, markersize=3, label="loc scores")
            #plt.show()

            #bgm = BayesianGaussianMixture(n_components=components, random_state=0).fit(pose_votes)
            #print('weights: ', bgm.weights_, np.sum(bgm.weights_))
            #print('covariances: ', bgm.covariances_.shape)
            #print('weights: ', bgm.weights_)
            #print('weights: ', bgm.weight_concentration_prior_)
            #print('concentrations: ', bgm.weight_concentration_)
            #print(mw_comp.shape, mw_comp)
            #print(bgm.means_[mw_comp,:])


            #print(variational_votes.shape)
            #print(obj_points.shape)

            # BGMM with 16 dimensions
            # using all components means
            components = 8
            if pose_votes.shape[0] < 8:
                components = 2
            # print('components: ', components)
            ori_points = np.ascontiguousarray(threeD_boxes[cls, :, :], dtype=np.float32)
            bgm = BayesianGaussianMixture(n_components=components, random_state=0).fit(pose_votes)
            smallest_indixes = np.argpartition(bgm.weights_, 2)

            if components > 3:
                hyps = bgm.means_[smallest_indixes[2:], :]
                weights = bgm.weights_[smallest_indixes[2:]]
                alphas = bgm.weight_concentration_[0][smallest_indixes[2:]]
                betas = bgm.weight_concentration_[1][smallest_indixes[2:]]
            else:
                hyps = bgm.means_
                weights = bgm.weights_
                alphas = bgm.weight_concentration_[0]
                betas = bgm.weight_concentration_[1]
            '''

            print('### post ###')
            print('hyps: ', hyps[:, 0])
            print('alphas: ', alphas)
            print('betas: ', betas)

            max_idx = np.argmax(alphas + betas)
            votes = (hyps[max_idx, :] * col_std) + col_mean
            print('max_idx: ', max_idx)
            '''

            #smallest_var = 10
            #k_idx = 0
            #for btx in range(len(hyps)):
            #    a = alphas[btx]
            #    b = betas[btx]
                #var = (a * b) / ((a + b + 1) * np.power((a + b), 2))
            #    var = a / (a+b)
            #    if np.abs(0.5 - var) < smallest_var:
            #        smallest_var = var
            #        k_idx = btx
            #print(k_idx, smallest_var)
            #print('weights: ', bgm.weights_)
            #print('higher weights: ', bgm.weights_[smallest_indixes[2:]])
            #max_con = np.argmax(bgm.weight_concentration_[0])
            #votes = (bgm.means_[k_idx, :] * col_std) + col_mean

            #indices_b_a = np.where(bgm.weight_concentration_[0] < bgm.weight_concentration_[1])
            #print(' ')
            #print('#############################################')
            #print('b>a: ', indices_b_a)
            #print('alpha: ', bgm.weight_concentration_[0])
            #print('beta: ', bgm.weight_concentration_[1])
            #print('b-a: ', bgm.weight_concentration_[1]-bgm.weight_concentration_[0])
            #abs_diffs = bgm.weight_concentration_[1]-bgm.weight_concentration_[0]
            #med_diff = np.where(abs_diffs==np.sort(abs_diffs)[len(abs_diffs)//2])
            #votes = (bgm.means_[med_diff, :] * col_std) + col_mean

            #hyps = bgm.means_
            col_std_exp = np.repeat(col_std[np.newaxis, :], repeats=hyps.shape[0], axis=0)
            col_mean_exp = np.repeat(col_mean[np.newaxis, :], repeats=hyps.shape[0], axis=0)
            votes = (hyps * col_std_exp) + col_mean_exp

            #########################
            # vanilla PyraPose
            #######################
            #k_hyp = len(hyp_indices[0])
            k_hyp = hyps.shape[0]
            true_pose = 0
            top_error = 1
            box_devs = []
            errors = []
            retvals = []
            highest_ind = 10

            #k_hyp = 1

            for pdx in range(k_hyp):
                est_points = np.ascontiguousarray(votes[pdx, :], dtype=np.float32).reshape((8, 1, 2))
                ori_points = np.ascontiguousarray(threeD_boxes[cls, :, :], dtype=np.float32)  # .reshape((8, 1, 3))
                K = np.float32([fxkin, 0., cxkin, 0., fykin, cykin, 0., 0., 1.]).reshape(3, 3)
                #pose_votes = boxes3D[0, hyp_indices, :]
                #est_points = np.ascontiguousarray(votes, dtype=np.float32).reshape((int(k_hyp * 8), 1, 2))
                #obj_points = np.repeat(ori_points[np.newaxis, :, :], k_hyp, axis=0)
                #obj_points = obj_points.reshape((int(k_hyp * 8), 1, 3))
                obj_points = np.repeat(ori_points[np.newaxis, :, :], 1, axis=0)
                obj_points = obj_points.reshape((8, 1, 3))
                retval, orvec, otvec, inliers = cv2.solvePnPRansac(objectPoints=obj_points,
                                                                   imagePoints=est_points, cameraMatrix=K,
                                                                   distCoeffs=None, rvec=None, tvec=None,
                                                                   useExtrinsicGuess=False, iterationsCount=300,
                                                                   reprojectionError=5.0, confidence=0.99,
                                                                   flags=cv2.SOLVEPNP_EPNP)

                R_est, _ = cv2.Rodrigues(orvec)
                t_est = otvec
                # t_rot = tf3d.euler.euler2mat(t_rot[0], t_rot[1], t_rot[2])
                t_rot_n = tf3d.quaternions.quat2mat(t_rot)
                R_gt = np.array(t_rot_n, dtype=np.float32).reshape(3, 3)
                t_gt = np.array(t_tra, dtype=np.float32)
                # print(t_est)
                # print(t_gt)
                t_gt = t_gt * 0.001
                t_est = t_est.T  # * 0.001
                if cls == 10 or cls == 11:
                    err_add = adi(R_est, t_est, R_gt, t_gt, model_vsd["pts"])
                else:
                    err_add = add(R_est, t_est, R_gt, t_gt, model_vsd["pts"])
                if err_add < model_dia[true_cat] * 0.1:
                    true_pose = 1
                if err_add < top_error:
                    top_error = err_add
                    highest_ind = pdx

                #print(pdx, bgm.weights_, bgm.weight_concentration_)
                #print(bgm.covariances_[pdx, :, :])
                #print('error: ', err_add)

                errors.append(err_add)
                #retvals.append(inliers)

                # box deviatio in camera frame
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
                box_dev = np.linalg.norm(eDbox - pose_votes)
                box_devs.append(box_dev)

                #print(' ')
                #print('error: ', err_add, 'threshold', model_dia[cls] * 0.1)


            #med_box_dev = np.median(box_devs)
            #below_median = np.where(box_devs < med_box_dev)
            #print(below_median)
            #filtered_hyps = hyp_indices[0][below_median]
            #print(len(box_devs))
            #ind_choice = np.argmin(box_devs)
            #print(filtered_votes.shape)
            #votes = filtered_votes[below_median, :]
            #print(votes.shape)

            #est_points = np.ascontiguousarray(votes, dtype=np.float32).reshape((int(8 * k_hyp), 1, 2))
            #obj_points = np.repeat(ori_points[np.newaxis, :, :], k_hyp, axis=0)
            #obj_points = obj_points.reshape((int(8 * k_hyp), 1, 3))

            '''
            est_points = np.ascontiguousarray(votes, dtype=np.float32).reshape((8, 1, 2))
            obj_points = np.repeat(ori_points[np.newaxis, :, :], 1, axis=0)
            obj_points = obj_points.reshape((8, 1, 3))

            K = np.float32([fxkin, 0., cxkin, 0., fykin, cykin, 0., 0., 1.]).reshape(3, 3)
            retval, orvec, otvec, inliers = cv2.solvePnPRansac(objectPoints=obj_points,
                                                               imagePoints=est_points, cameraMatrix=K,
                                                               distCoeffs=None, rvec=None, tvec=None,
                                                               useExtrinsicGuess=False, iterationsCount=300,
                                                               reprojectionError=5.0, confidence=0.99,
                                                               flags=cv2.SOLVEPNP_EPNP)


            R_est, _ = cv2.Rodrigues(orvec)
            t_est = otvec
            t_rot_n = tf3d.quaternions.quat2mat(t_rot)
            R_gt = np.array(t_rot_n, dtype=np.float32).reshape(3, 3)
            t_gt = np.array(t_tra, dtype=np.float32)

            t_gt = t_gt * 0.001
            t_est = t_est.T  # * 0.001
            if cls == 10 or cls == 11:
                err_add = adi(R_est, t_est, R_gt, t_gt, model_vsd["pts"])
            else:
                err_add = add(R_est, t_est, R_gt, t_gt, model_vsd["pts"])
            if err_add < model_dia[true_cat] * 0.1:
                truePoses[int(true_cat)] += 1
            '''


            covs = []
            covs_abs = []

            cov0 = bgm.covariances_[0, :, :]
            n_cov0 = np.diagonal(cov0)
            D = np.eye(16)
            np.fill_diagonal(D, n_cov0)
            D_inv = np.linalg.inv(np.sqrt(D))
            corr0 = D_inv @ cov0 @ D_inv
            #corrs.append(np.mean(corr0))
            cov_nd = cov0
            cov_nd[np.diag_indices(16)] = 0
            covs.append(np.sum(cov_nd)/2)
            covs_abs.append(np.sum(np.abs(cov_nd))/2)

            cov1 = bgm.covariances_[1, :, :]
            n_cov1 = np.diagonal(cov1)
            D = np.eye(16)
            np.fill_diagonal(D, n_cov1)
            D_inv = np.linalg.inv(np.sqrt(D))
            corr1 = D_inv @ cov1 @ D_inv
            #corrs.append(np.mean(corr1))
            cov_nd = cov1
            cov_nd[np.diag_indices(16)] = 0
            covs.append(np.sum(cov_nd) / 2)
            covs_abs.append(np.sum(np.abs(cov_nd)) / 2)

            cov2 = bgm.covariances_[2, :, :]
            n_cov2 = np.diagonal(cov2)
            D = np.eye(16)
            np.fill_diagonal(D, n_cov2)
            D_inv = np.linalg.inv(np.sqrt(D))
            corr2 = D_inv @ cov2 @ D_inv
            #corrs.append(np.mean(corr2))
            cov_nd = cov2
            cov_nd[np.diag_indices(16)] = 0
            covs.append(np.sum(cov_nd) / 2)
            covs_abs.append(np.sum(np.abs(cov_nd)) / 2)

            cov3 = bgm.covariances_[3, :, :]
            n_cov3 = np.diagonal(cov3)
            D = np.eye(16)
            np.fill_diagonal(D, n_cov3)
            D_inv = np.linalg.inv(np.sqrt(D))
            corr3 = D_inv @ cov3 @ D_inv
            #corrs.append(np.mean(corr3))
            cov_nd = cov3
            cov_nd[np.diag_indices(16)] = 0
            covs.append(np.sum(cov_nd) / 2)
            covs_abs.append(np.sum(np.abs(cov_nd)) / 2)

            cov4 = bgm.covariances_[4, :, :]
            n_cov4 = np.diagonal(cov4)
            D = np.eye(16)
            np.fill_diagonal(D, n_cov4)
            D_inv = np.linalg.inv(np.sqrt(D))
            corr4 = D_inv @ cov4 @ D_inv
            #corrs.append(np.mean(corr4))
            cov_nd = cov4
            cov_nd[np.diag_indices(16)] = 0
            covs.append(np.sum(cov_nd) / 2)
            covs_abs.append(np.sum(np.abs(cov_nd)) / 2)

            cov5 = bgm.covariances_[5, :, :]
            n_cov5 = np.diagonal(cov5)
            D = np.eye(16)
            np.fill_diagonal(D, n_cov5)
            D_inv = np.linalg.inv(np.sqrt(D))
            corr5 = D_inv @ cov5 @ D_inv
            #corrs.append(np.mean(corr5))
            cov_nd = cov5
            cov_nd[np.diag_indices(16)] = 0
            covs.append(np.sum(cov_nd) / 2)
            covs_abs.append(np.sum(np.abs(cov_nd)) / 2)

            fig, axs = plt.subplots(3, 2)
            i1 = axs[0 ,0].matshow(corr0, cmap='hot', interpolation='nearest')
            i2 = axs[0, 1].matshow(corr1, cmap='hot', interpolation='nearest')
            i3 = axs[1, 0].matshow(corr2, cmap='hot', interpolation='nearest')
            i4 = axs[1, 1].matshow(corr3, cmap='hot', interpolation='nearest')
            i5 = axs[2, 0].matshow(corr4, cmap='hot', interpolation='nearest')
            i6 = axs[2, 1].matshow(corr5, cmap='hot', interpolation='nearest')

            fig.colorbar(i1, ax=axs[0 ,0])
            fig.colorbar(i2, ax=axs[0 ,1])
            fig.colorbar(i3, ax=axs[1 ,0])
            fig.colorbar(i4, ax=axs[1 ,1])
            fig.colorbar(i5, ax=axs[2 ,0])
            fig.colorbar(i6, ax=axs[2 ,1])

            # truePoses[int(true_cat)] += true_pose
            print(' ')
            print('min error id: ', highest_ind)
            # print('weights: ', weights)
            # print('|a-b|: ', np.abs(alphas - betas))
            # aab = np.concatenate([alphas[np.newaxis, :], betas[np.newaxis, :]], axis=0)
            # normalizer = np.nanmax(aab, axis=0)
            # print('||a|-|b||: ', np.abs((alphas/normalizer) - (betas/normalizer)))
            print('errors: ', errors)
            #print('corrs: ', corrs)
            print('covs: ', covs)
            print('covs_abs: ', covs_abs)
            print(index, 'error: ', top_error, 'threshold', model_dia[cls] * 0.1)

            plt.show()

            #norm_thres = np.asarray(model_dia[true_cat] * 0.1) * (1 / max(np.asarray(errors)))
            #errors_norm = np.asarray(errors) * (1 / np.nanmax(np.asarray(errors)))
            #retvals_norm = np.asarray(retvals) * (1 / np.nanmax(np.asarray(retvals)))

            #sorting = np.argsort(errors_norm)
            #filtered_errors = errors_norm[sorting]
            #filtered_retvals = retvals_norm[sorting]
            #x_axis = range(len(errors_norm))
            #plt.plot(x_axis, filtered_errors, 'bo:'bgm.weight_concentration_[0], linewidth=2, markersize=3, label="errors per hypothesis")
            #plt.plot(x_axis, filtered_retvals, 'rv-', linewidth=2, markersize=3, label="retvals")
            #plt.show()

            ###################################
            # other experiments
            ###############################
            # weighted mean
            ##############################
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

            # result = [int(BOP_scene_id), int(BOP_im_id), int(BOP_obj_id), float(BOP_score), BOP_R[0], BOP_R[1], BOP_R[2], BOP_R[3], BOP_R[4], BOP_R[5], BOP_R[6], BOP_R[7], BOP_R[8], BOP_t[0], BOP_t[1], BOP_t[2]]
            # result = [int(BOP_scene_id), int(BOP_im_id), int(BOP_obj_id), float(BOP_score), BOP_R, BOP_t]
            # results_image.append(result)

            tDbox = R_gt.dot(ori_points.T).T
            tDbox = tDbox + np.repeat(t_gt[:, np.newaxis], 8, axis=1).T
            box3D = toPix_array(tDbox)
            tDbox = np.reshape(box3D, (16))
            tDbox = tDbox.astype(np.uint16)

            #eDbox = R_est.dot(ori_points.T).T
            ##eDbox = eDbox + np.repeat(t_est[:, np.newaxis], 8, axis=1).T
            #eDbox = eDbox + np.repeat(t_est, 8, axis=0)
            #est3D = toPix_array(eDbox)
            #eDbox = np.reshape(est3D, (16))
            #pose = eDbox.astype(np.uint16)

            #print('gt: ', tDbox)
            #print('est: ', pose)

            colGT = (255, 0, 0)
            #colEst = (0, 215, 255)

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
            '''            

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
            '''

            '''
            hyp_mask = np.zeros((640, 480), dtype=np.float32)
            for idx in range(k_hyp):
                hyp_mask[int(est_points[idx, 0, 0]), int(est_points[idx, 0, 1])] += 1

            hyp_mask = np.transpose(hyp_mask)
            hyp_mask = (hyp_mask * (255.0 / np.nanmax(hyp_mask))).astype(np.uint8)

            image_raw[:, :, 0] = np.where(hyp_mask > 0, 0, image_raw[:, :, 0])
            image_raw[:, :, 1] = np.where(hyp_mask > 0, 0, image_raw[:, :, 1])
            image_raw[:, :, 2] = np.where(hyp_mask > 0, hyp_mask, image_raw[:, :, 2])
            '''

            '''
            #image = image_raw
            image_raw = np.zeros((480, 640, 3))
            print(est_points.shape)
            idx = 0
            for i in range(k_hyp):
                image_raw = cv2.circle(image_raw, (est_points[idx, 0, 0], est_points[idx, 0, 1]), 2, (13, 243, 207), -2)
                image_raw = cv2.circle(image_raw, (est_points[idx+1, 0, 0], est_points[idx+1, 0, 1]), 2, (251, 194, 213), -2)
                image_raw = cv2.circle(image_raw, (est_points[idx+2, 0, 0], est_points[idx+2, 0, 1]), 2, (222, 243, 41), -2)
                image_raw = cv2.circle(image_raw, (est_points[idx+3, 0, 0], est_points[idx+3, 0, 1]), 2, (209, 31, 201), -2)
                image_raw = cv2.circle(image_raw, (est_points[idx+4, 0, 0], est_points[idx+4, 0, 1]), 2, (8, 62, 53), -2)
                image_raw = cv2.circle(image_raw, (est_points[idx+5, 0, 0], est_points[idx+5, 0, 1]), 2, (13, 243, 207), -2)
                image_raw = cv2.circle(image_raw, (est_points[idx+6, 0, 0], est_points[idx+6, 0, 1]), 2, (215, 41, 29), -2)
                image_raw = cv2.circle(image_raw, (est_points[idx+7, 0, 0], est_points[idx+7, 0, 1]), 2, (78, 213, 16), -2)
                #image_raw[int(est_points[idx, 0, 1]), int(est_points[idx, 0, 0]), :] += 1
                #image_raw[int(est_points[idx+1, 0, 1]), int(est_points[idx+1, 0, 0]), :] += 1
                #image_raw[int(est_points[idx+2, 0, 1]), int(est_points[idx+2, 0, 0]), :] += 1
                #image_raw[int(est_points[idx+3, 0, 1]), int(est_points[idx+3, 0, 0]), :] += 1
                #image_raw[int(est_points[idx+4, 0, 1]), int(est_points[idx+4, 0, 0]), :] += 1
                #image_raw[int(est_points[idx+5, 0, 1]), int(est_points[idx+5, 0, 0]), :] += 1
                #image_raw[int(est_points[idx+6, 0, 1]), int(est_points[idx+6, 0, 0]), :] += 1
                #image_raw[int(est_points[idx+7, 0, 1]), int(est_points[idx+7, 0, 0]), :] += 1
                idx = idx+8



            #y_min = np.nanmin(pose_votes[0, :, 1])
            #y_max = np.nanmax(pose_votes[0, :, 1])
            #x_min = np.nanmin(pose_votes[0, :, 0])
            #x_max = np.nanmax(pose_votes[0, :, 0])
            y_min = np.nanmin(est_points[:, 0, 1])
            y_max = np.nanmax(est_points[:, 0, 1])
            x_min = np.nanmin(est_points[:, 0, 0])
            x_max = np.nanmax(est_points[:, 0, 0])
            print(y_min, y_max, x_min, x_max)
            image_raw = (image_raw * 255/np.nanmax(image_raw)).astype(np.uint8)
            #image[tDbox[1], tDbox[0]] = (255, 215, 0)
            #image[tDbox[3], tDbox[2]] = (255, 215, 0)
            #image[tDbox[5], tDbox[4]] = (255, 215, 0)
            #image[tDbox[7], tDbox[6]] = (255, 215, 0)
            #image[tDbox[9], tDbox[8]] = (255, 215, 0)
            #image[tDbox[11], tDbox[10]] = (255, 215, 0)
            #image[tDbox[13], tDbox[12]] = (255, 215, 0)
            #image[tDbox[15], tDbox[14]] = (255, 215, 0)

            colGT = (255, 0, 0)
            colEst = (0, 215, 255)

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


            image_raw = cv2.line(image_raw, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), colEst, 1)
            image_raw = cv2.line(image_raw, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), colEst, 1)
            image_raw = cv2.line(image_raw, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), colEst, 1)
            image_raw = cv2.line(image_raw, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), colEst, 1)
            image_raw = cv2.line(image_raw, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), colEst, 1)
            image_raw = cv2.line(image_raw, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), colEst, 1)
            image_raw = cv2.line(image_raw, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), colEst, 1)
            image_raw = cv2.line(image_raw, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), colEst, 1)
            image_raw = cv2.line(image_raw, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()), colEst,
                             1)
            image_raw = cv2.line(image_raw, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), colEst,
                             1)
            image_raw = cv2.line(image_raw, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), colEst,
                             1)
            image_raw = cv2.line(image_raw, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), colEst,
                             1)
            image_raw = image_raw[int(y_min-20):int(y_max+20), int(x_min-20):int(x_max+20)]
            plt.imshow(image_raw)
            plt.show()
            '''

            '''
            # visualize covariances 
            max_x = int(np.max(tDbox[0]) + 20)
            min_x = int(np.min(tDbox[0]) - 20)
            max_y = int(np.max(tDbox[1]) + 20)
            min_y = int(np.min(tDbox[1]) - 20)

            image_raw = plot_ellipses(image_raw, means, covars, weights)
            image_crop = image_raw[min_y:max_y, min_x:max_x, :]
            image_crop = cv2.resize(image_crop, None, fx=10, fy=10)

            name = '/home/stefan/PyraPose_viz/detection_LM.jpg'
            cv2.imwrite(name, image_crop)
            print('break')
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
