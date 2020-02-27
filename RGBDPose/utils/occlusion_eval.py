
#from pycocotools.cocoeval import COCOeval

import keras
import numpy as np
import json
import pyquaternion
import math
import transforms3d as tf3d
import geometry
import os
import copy
import cv2
import open3d
from ..utils import ply_loader
from .pose_error import reproj, add, adi, re, te, vsd
import yaml

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


def toPix_array(translation):

    xpix = ((translation[:, 0] * fxkin) / translation[:, 2]) + cxkin
    ypix = ((translation[:, 1] * fykin) / translation[:, 2]) + cykin
    #zpix = translation[2] * fxkin

    return np.stack((xpix, ypix), axis=1) #, zpix]


def load_pcd(cat):
    # load meshes
    mesh_path = "/home/sthalham/data/LINEMOD/models/"
    #mesh_path = "/home/stefan/data/val_linemod_cc_rgb/models_ply/"
    ply_path = mesh_path + 'obj_' + cat + '.ply'
    model_vsd = ply_loader.load_ply(ply_path)
    pcd_model = open3d.PointCloud()
    pcd_model.points = open3d.Vector3dVector(model_vsd['pts'])
    open3d.estimate_normals(pcd_model, search_param=open3d.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    # open3d.draw_geometries([pcd_model])
    model_vsd_mm = copy.deepcopy(model_vsd)
    model_vsd_mm['pts'] = model_vsd_mm['pts'] * 1000.0
    pcd_model = open3d.read_point_cloud(ply_path)

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


def evaluate_occlusion(generator, model, threshold=0.05):
    threshold = 0.5

    #mesh_info = '/RGBDPose/linemod_13/models_info.yml'
    #mesh_info = '/home/stefan/data/Meshes/linemod_13/models_info.yml'
    mesh_info = '/home/sthalham/data/Meshes/linemod_13/models_info.yml'

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

    # start collecting results
    results = []
    image_ids = []
    image_indices = []
    idx = 0

    tp = np.zeros((16), dtype=np.uint32)
    fp = np.zeros((16), dtype=np.uint32)
    fn = np.zeros((16), dtype=np.uint32)

    # interlude
    tp05 = np.zeros((16), dtype=np.uint32)
    fp05 = np.zeros((16), dtype=np.uint32)
    fn05 = np.zeros((16), dtype=np.uint32)

    tp10 = np.zeros((16), dtype=np.uint32)
    fp10 = np.zeros((16), dtype=np.uint32)
    fn10 = np.zeros((16), dtype=np.uint32)

    tp15 = np.zeros((16), dtype=np.uint32)
    fp15 = np.zeros((16), dtype=np.uint32)
    fn15 = np.zeros((16), dtype=np.uint32)

    tp20 = np.zeros((16), dtype=np.uint32)
    fp20 = np.zeros((16), dtype=np.uint32)
    fn20 = np.zeros((16), dtype=np.uint32)

    tp25 = np.zeros((16), dtype=np.uint32)
    fp25 = np.zeros((16), dtype=np.uint32)
    fn25 = np.zeros((16), dtype=np.uint32)

    tp30 = np.zeros((16), dtype=np.uint32)
    fp30 = np.zeros((16), dtype=np.uint32)
    fn30 = np.zeros((16), dtype=np.uint32)

    tp35 = np.zeros((16), dtype=np.uint32)
    fp35 = np.zeros((16), dtype=np.uint32)
    fn35 = np.zeros((16), dtype=np.uint32)

    tp40 = np.zeros((16), dtype=np.uint32)
    fp40 = np.zeros((16), dtype=np.uint32)
    fn40 = np.zeros((16), dtype=np.uint32)

    tp45 = np.zeros((16), dtype=np.uint32)
    fp45 = np.zeros((16), dtype=np.uint32)
    fn45 = np.zeros((16), dtype=np.uint32)

    tp55 = np.zeros((16), dtype=np.uint32)
    fp55 = np.zeros((16), dtype=np.uint32)
    fn55 = np.zeros((16), dtype=np.uint32)

    tp6 = np.zeros((16), dtype=np.uint32)
    fp6 = np.zeros((16), dtype=np.uint32)
    fn6 = np.zeros((16), dtype=np.uint32)

    tp65 = np.zeros((16), dtype=np.uint32)
    fp65 = np.zeros((16), dtype=np.uint32)
    fn65 = np.zeros((16), dtype=np.uint32)

    tp7 = np.zeros((16), dtype=np.uint32)
    fp7 = np.zeros((16), dtype=np.uint32)
    fn7 = np.zeros((16), dtype=np.uint32)

    tp75 = np.zeros((16), dtype=np.uint32)
    fp75 = np.zeros((16), dtype=np.uint32)
    fn75 = np.zeros((16), dtype=np.uint32)

    tp8 = np.zeros((16), dtype=np.uint32)
    fp8 = np.zeros((16), dtype=np.uint32)
    fn8 = np.zeros((16), dtype=np.uint32)

    tp85 = np.zeros((16), dtype=np.uint32)
    fp85 = np.zeros((16), dtype=np.uint32)
    fn85 = np.zeros((16), dtype=np.uint32)

    tp9 = np.zeros((16), dtype=np.uint32)
    fp9 = np.zeros((16), dtype=np.uint32)
    fn9 = np.zeros((16), dtype=np.uint32)

    tp925 = np.zeros((16), dtype=np.uint32)
    fp925 = np.zeros((16), dtype=np.uint32)
    fn925 = np.zeros((16), dtype=np.uint32)

    tp95 = np.zeros((16), dtype=np.uint32)
    fp95 = np.zeros((16), dtype=np.uint32)
    fn95 = np.zeros((16), dtype=np.uint32)

    tp975 = np.zeros((16), dtype=np.uint32)
    fp975 = np.zeros((16), dtype=np.uint32)
    fn975 = np.zeros((16), dtype=np.uint32)
    # interlude end

    tp_add = np.zeros((16), dtype=np.uint32)
    fp_add = np.zeros((16), dtype=np.uint32)
    fn_add = np.zeros((16), dtype=np.uint32)

    rotD = np.zeros((16), dtype=np.uint32)
    less5 = np.zeros((16), dtype=np.uint32)
    rep_e = np.zeros((16), dtype=np.uint32)
    rep_less5 = np.zeros((16), dtype=np.uint32)
    add_e = np.zeros((16), dtype=np.uint32)
    add_less_d = np.zeros((16), dtype=np.uint32)
    vsd_e = np.zeros((16), dtype=np.uint32)
    vsd_less_t = np.zeros((16), dtype=np.uint32)

    add_less_d005 = np.zeros((16), dtype=np.uint32)
    add_less_d015 = np.zeros((16), dtype=np.uint32)
    add_less_d02 = np.zeros((16), dtype=np.uint32)
    add_less_d025 = np.zeros((16), dtype=np.uint32)
    add_less_d03 = np.zeros((16), dtype=np.uint32)
    add_less_d035 = np.zeros((16), dtype=np.uint32)
    add_less_d04 = np.zeros((16), dtype=np.uint32)
    add_less_d045 = np.zeros((16), dtype=np.uint32)
    add_less_d05 = np.zeros((16), dtype=np.uint32)
    add_less_d055 = np.zeros((16), dtype=np.uint32)
    add_less_d06 = np.zeros((16), dtype=np.uint32)
    add_less_d065 = np.zeros((16), dtype=np.uint32)
    add_less_d07 = np.zeros((16), dtype=np.uint32)
    add_less_d075 = np.zeros((16), dtype=np.uint32)
    add_less_d08 = np.zeros((16), dtype=np.uint32)
    add_less_d085 = np.zeros((16), dtype=np.uint32)
    add_less_d09 = np.zeros((16), dtype=np.uint32)
    add_less_d095 = np.zeros((16), dtype=np.uint32)
    add_less_d1 = np.zeros((16), dtype=np.uint32)
    
    # target annotation
    pc1, mv1, mv1_mm = load_pcd('01')
    pc5, mv5, mv5_mm = load_pcd('05')
    pc6, mv6, mv6_mm = load_pcd('06')
    pc8, mv8, mv8_mm = load_pcd('08')
    pc9, mv9, mv9_mm = load_pcd('09')
    pc10, mv10, mv10_mm = load_pcd('10')
    pc11, mv11, mv11_mm = load_pcd('11')
    pc12, mv12, mv12_mm = load_pcd('12')

    for index in progressbar.progressbar(range(generator.size()), prefix='LineMOD evaluation: '):
        image_raw = generator.load_image(index)
        image = generator.preprocess_image(image_raw)
        image, scale = generator.resize_image(image)

        image_raw_dep = generator.load_image_dep(index)
        image_dep = generator.preprocess_image(image_raw_dep)
        image_dep, scale = generator.resize_image(image_dep)

        raw_dep_path = generator.load_image_dep_raw(index)

        load_viz_path = raw_dep_path[:-17] + raw_dep_path[-17:-12] + '_rgb.jpg'
        image_viz = cv2.imread(load_viz_path)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        anno = generator.load_annotations(index)

        #print(anno['labels'])
        t_cat = anno['labels'].astype(np.int8) + 1
        obj_name = []
        for idx, obj_temp in enumerate(t_cat):
            if obj_temp < 10:
                obj_name.append('0' + str(obj_temp))
            else:
                obj_name.append(str(obj_temp))
        t_bbox = np.asarray(anno['bboxes'], dtype=np.float32)
        gt_poses = anno['poses']

        # run network
        boxes, boxes3D, scores, labels = model.predict_on_batch([np.expand_dims(image, axis=0), np.expand_dims(image_dep, axis=0)])

        #print(scores)
        #print(labels)

        # correct boxes for image scale
        boxes /= scale

        # change to (x, y, w, h) (MS COCO standard)
        boxes[:, :, 2] -= boxes[:, :, 0]
        boxes[:, :, 3] -= boxes[:, :, 1]

        rotD[t_cat] += 1
        rep_e[t_cat] += 1
        add_e[t_cat] += 1
        vsd_e[t_cat] += 1
        fn05[t_cat] += 1
        fn10[t_cat] += 1
        fn15[t_cat] += 1
        fn20[t_cat] += 1
        fn25[t_cat] += 1
        fn30[t_cat] += 1
        fn35[t_cat] += 1
        fn40[t_cat] += 1
        fn45[t_cat] += 1
        fn[t_cat] += 1
        fn55[t_cat] += 1
        fn6[t_cat] += 1
        fn65[t_cat] += 1
        fn7[t_cat] += 1
        fn75[t_cat] += 1
        fn8[t_cat] += 1
        fn85[t_cat] += 1
        fn9[t_cat] += 1
        fn925[t_cat] += 1
        fn95[t_cat] += 1
        fn975[t_cat] += 1

        # end interlude
        fn_add[t_cat] += 1
        fnit = np.ones((16), dtype=np.bool)

        # compute predicted labels and scores
        for box, box3D, score, label in zip(boxes[0], boxes3D[0], scores[0], labels[0]):
            # scores are sorted, so we can break
            if score < threshold:
                continue

            if label < 0:
                continue

            if label == 1:
                continue



            cls = generator.label_to_inv_label(label)
            if cls > 5:
                cls += 2
            elif cls > 2:
                cls += 1
            else: pass
            #cls = 1
            #control_points = box3D[(cls - 1), :]
            control_points = box3D
            #control_points = box3D[0, :]

            # append detection for each positively labeled class
            image_result = {
                'image_id'    : generator.image_ids[index],
                'category_id' : generator.label_to_inv_label(label),
                'score'       : float(score),
                'bbox'        : box.tolist(),
                'pose'        : control_points.tolist()
            }

            # append detection to results
            results.append(image_result)

            if cls in t_cat:
                b1 = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]])
                odx = np.where(t_cat==cls)
                b2 = np.array([t_bbox[odx[0]][0][0], t_bbox[odx[0]][0][1], t_bbox[odx[0]][0][2], t_bbox[odx[0]][0][3]])

                IoU = boxoverlap(b1, b2)
                # occurences of 2 or more instances not possible in LINEMOD
                #print(cls, IoU)
                if IoU > 0.5:
                    if fnit[cls] == True:
                        # interlude
                        '''
                        if IoU > 0.55:
                            tp55[cls] += 1
                            fn55[cls] -= 1
                        else:
                            fp55[cls] += 1
                        if IoU > 0.6:
                            tp6[cls] += 1
                            fn6[cls] -= 1
                        else:
                            fp6[cls] += 1
                        if IoU > 0.65:
                            tp65[cls] += 1
                            fn65[cls] -= 1
                        else:
                            fp65[cls] += 1
                        if IoU > 0.7:
                            tp7[cls] += 1
                            fn7[cls] -= 1
                        else:
                            fp7[cls] += 1
                        if IoU > 0.75:
                            tp75[cls] += 1
                            fn75[cls] -= 1
                        else:
                            fp75[cls] += 1
                        if IoU > 0.8:
                            tp8[cls] += 1
                            fn8[cls] -= 1
                        else:
                            fp8[cls] += 1
                        if IoU > 0.85:
                            tp85[cls] += 1
                            fn85[cls] -= 1
                        else:
                            fp85[cls] += 1
                        if IoU > 0.9:
                            tp9[cls] += 1
                            fn9[cls] -= 1
                        else:
                            fp9[cls] += 1
                        if IoU > 0.925:
                            tp925[cls] += 1
                            fn925[cls] -= 1
                        else:
                            fp925[cls] += 1
                        if IoU > 0.95:
                            tp95[cls] += 1
                            fn95[cls] -= 1
                        else:
                            fp95[cls] += 1
                        if IoU > 0.975:
                            tp975[cls] += 1
                            fn975[cls] -= 1
                        else:
                            fp975[cls] += 1

                        # interlude end

                        tp[cls] += 1
                        fn[cls] -= 1
                        '''
                        if IoU > 0.05:
                            tp05[t_cat[odx[0]]] += 1
                            fn05[t_cat[odx[0]]] -= 1
                        else:
                            fp05[t_cat[odx[0]]] += 1
                        if IoU > 0.1:
                            tp10[t_cat[odx[0]]] += 1
                            fn10[t_cat[odx[0]]] -= 1
                        else:
                            fp10[t_cat[odx[0]]] += 1
                        if IoU > 0.15:
                            tp15[t_cat[odx[0]]] += 1
                            fn15[t_cat[odx[0]]] -= 1
                        else:
                            fp15[t_cat[odx[0]]] += 1
                        if IoU > 0.2:
                            tp20[t_cat[odx[0]]] += 1
                            fn20[t_cat[odx[0]]] -= 1
                        else:
                            fp20[t_cat[odx[0]]] += 1
                        if IoU > 0.25:
                            tp25[t_cat[odx[0]]] += 1
                            fn25[t_cat[odx[0]]] -= 1
                        else:
                            fp25[t_cat[odx[0]]] += 1
                        if IoU > 0.3:
                            tp30[t_cat[odx[0]]] += 1
                            fn30[t_cat[odx[0]]] -= 1
                        else:
                            fp30[t_cat[odx[0]]] += 1
                        if IoU > 0.35:
                            tp35[t_cat[odx[0]]] += 1
                            fn35[t_cat[odx[0]]] -= 1
                        else:
                            fp35[t_cat[odx[0]]] += 1
                        if IoU > 0.4:
                            tp40[t_cat[odx[0]]] += 1
                            fn40[t_cat[odx[0]]] -= 1
                        else:
                            fp40[t_cat[odx[0]]] += 1
                        if IoU > 0.45:
                            tp45[t_cat[odx[0]]] += 1
                            fn45[t_cat[odx[0]]] -= 1
                        else:
                            fp45[t_cat[odx[0]]] += 1
                        if IoU > 0.55:
                            tp55[t_cat[odx[0]]] += 1
                            fn55[t_cat[odx[0]]] -= 1
                        else:
                            fp55[t_cat[odx[0]]] += 1
                        if IoU > 0.6:
                            tp6[t_cat[odx[0]]] += 1
                            fn6[t_cat[odx[0]]] -= 1
                        else:
                            fp6[t_cat[odx[0]]] += 1
                        if IoU > 0.65:
                            tp65[t_cat[odx[0]]] += 1
                            fn65[t_cat[odx[0]]] -= 1
                        else:
                            fp65[t_cat[odx[0]]] += 1
                        if IoU > 0.7:
                            tp7[t_cat[odx[0]]] += 1
                            fn7[t_cat[odx[0]]] -= 1
                        else:
                            fp7[t_cat[odx[0]]] += 1
                        if IoU > 0.75:
                            tp75[t_cat[odx[0]]] += 1
                            fn75[t_cat[odx[0]]] -= 1
                        else:
                            fp75[t_cat[odx[0]]] += 1
                        if IoU > 0.8:
                            tp8[t_cat[odx[0]]] += 1
                            fn8[t_cat[odx[0]]] -= 1
                        else:
                            fp8[t_cat[odx[0]]] += 1
                        if IoU > 0.85:
                            tp85[t_cat[odx[0]]] += 1
                            fn85[t_cat[odx[0]]] -= 1
                        else:
                            fp85[t_cat[odx[0]]] += 1
                        if IoU > 0.9:
                            tp9[t_cat[odx[0]]] += 1
                            fn9[t_cat[odx[0]]] -= 1
                        else:
                            fp9[t_cat[odx[0]]] += 1
                        if IoU > 0.925:
                            tp925[t_cat[odx[0]]] += 1
                            fn925[t_cat[odx[0]]] -= 1
                        else:
                            fp925[t_cat[odx[0]]] += 1
                        if IoU > 0.95:
                            tp95[t_cat[odx[0]]] += 1
                            fn95[t_cat[odx[0]]] -= 1
                        else:
                            fp95[t_cat[odx[0]]] += 1
                        if IoU > 0.975:
                            tp975[t_cat[odx[0]]] += 1
                            fn975[t_cat[odx[0]]] -= 1
                        else:
                            fp975[t_cat[odx[0]]] += 1

                        # interlude end

                        tp[t_cat[odx[0]]] += 1
                        fn[t_cat[odx[0]]] -= 1
                        fnit[cls] = False

                        obj_points = np.ascontiguousarray(threeD_boxes[cls, :, :], dtype=np.float32) #.reshape((8, 1, 3))
                        est_points = np.ascontiguousarray(control_points.T, dtype=np.float32).reshape((8, 1, 2))

                        K = np.float32([fxkin, 0., cxkin, 0., fykin, cykin, 0., 0., 1.]).reshape(3, 3)

                        #retval, orvec, otvec = cv2.solvePnP(obj_points, est_points, K, None, None, None, False, cv2.SOLVEPNP_ITERATIVE)
                        retval, orvec, otvec, inliers = cv2.solvePnPRansac(objectPoints=obj_points,
                                                                           imagePoints=est_points, cameraMatrix=K,
                                                                           distCoeffs=None, rvec=None, tvec=None,
                                                                           useExtrinsicGuess=False, iterationsCount=100,
                                                                           reprojectionError=5.0, confidence=0.99,
                                                                           flags=cv2.SOLVEPNP_ITERATIVE)

                        R_est, _ = cv2.Rodrigues(orvec)
                        t_est = otvec
                        #print(t_est)

                        cur_pose = gt_poses[odx[0]]
                        t_rot = cur_pose[0][3:]
                        t_tra = cur_pose[0][:3]

                        t_rot = tf3d.euler.euler2mat(t_rot[0], t_rot[1], t_rot[2])
                        R_gt = np.array(t_rot, dtype=np.float32).reshape(3, 3)
                        t_gt = np.array(t_tra, dtype=np.float32) * 0.001

                        rd = re(R_gt, R_est)
                        xyz = te(t_gt, t_est.T)
                        #print(control_points)


                        '''
                        font = cv2.FONT_HERSHEY_COMPLEX
                        bottomLeftCornerOfText = (int(bb[0]) + 5, int(bb[1]) + int(bb[3]) - 5)
                        fontScale = 0.5
                        fontColor = (25, 215, 250)
                        fontthickness = 2
                        lineType = 2
                        if detCats[i] == 1:
                            cate = 'Ape'
                        elif detCats[i] == 2:
                            cate = 'Benchvise'
                        elif detCats[i] == 3:
                            cate = 'Bowl'
                        elif detCats[i] == 4:
                            cate = 'Camera'
                        elif detCats[i] == 5:
                            cate = 'Can'
                        elif detCats[i] == 6:
                            cate = 'Cat'
                        elif detCats[i] == 7:
                            cate = 'Cup'
                        elif detCats[i] == 8:
                            cate = 'Driller'
                        elif detCats[i] == 9:
                            cate = 'Duck'
                        elif detCats[i] == 10:
                            cate = 'Eggbox'
                        elif detCats[i] == 11:
                            cate = 'Glue'
                        elif detCats[i] == 12:
                            cate = 'Holepuncher'
                        elif detCats[i] == 13:
                            cate = 'Iron'
                        elif detCats[i] == 14:
                            cate = 'Lamp'
                        elif detCats[i] == 15:
                            cate = 'Phone'
                        gtText = cate
                        # gtText = cate + " / " + str(detSco[i])
                        fontColor2 = (0, 0, 0)
                        fontthickness2 = 4
                        cv2.putText(img, gtText,
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor2,
                            fontthickness2,
                            lineType)
                        cv2.putText(img, gtText,
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            fontthickness,
                            lineType)
                        name = '/home/sthalham/visTests/detected.jpg'
                        img_con = np.concatenate((img, img_gt), axis=1)
                        cv2.imwrite(name, img_con)
                        '''

                        if cls == 1:
                            model_vsd = mv1
                            point_cloud = pc1
                        elif cls == 5:
                            model_vsd = mv5
                            point_cloud = pc5
                        elif cls == 6:
                            model_vsd = mv6
                            point_cloud = pc6
                        elif cls == 8:
                            model_vsd = mv8
                            point_cloud = pc8
                        elif cls == 9:
                            model_vsd = mv9
                            point_cloud = pc9
                        elif cls == 10:
                            model_vsd = mv10
                            point_cloud = pc10
                        elif cls == 11:
                            model_vsd = mv11
                            point_cloud = pc11
                        elif cls == 12:
                            model_vsd = mv12
                            point_cloud = pc12

                        #print('--------------------- ICP refinement -------------------')
                        '''
                        image_dep = cv2.imread(raw_dep_path, cv2.IMREAD_UNCHANGED)
                        # image_icp = np.multiply(image_dep, 0.1)
                        image_icp = image_dep
                        # print(np.nanmax(image_icp))

                        pcd_img = create_point_cloud(image_icp, fxkin, fykin, cxkin, cykin, 1.0)
                        pcd_img = pcd_img.reshape((480, 640, 3))[int(b1[1]):int(b1[3]), int(b1[0]):int(b1[2]), :]
                        pcd_img = pcd_img.reshape((pcd_img.shape[0] * pcd_img.shape[1], 3))
                        todel = []

                        for i in range(len(pcd_img[:, 2])):
                            if pcd_img[i, 2] < 300:
                                todel.append(i)
                        pcd_img = np.delete(pcd_img, todel, axis=0)
                        # print(pcd_img)

                        pcd_crop = open3d.PointCloud()
                        pcd_crop.points = open3d.Vector3dVector(pcd_img)
                        # open3d.estimate_normals(pcd_crop, search_param=open3d.KDTreeSearchParamHybrid(
                        #     radius=2.0, max_nn=30))

                        # pcd_crop.paint_uniform_color(np.array([0.99, 0.0, 0.00]))
                        # open3d.draw_geometries([pcd_crop])

                        guess = np.zeros((4, 4), dtype=np.float32)
                        guess[:3, :3] = R_est
                        guess[:3, 3] = t_est.T * 1000.0
                        guess[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32).T

                        pcd_model = open3d.geometry.voxel_down_sample(point_cloud, voxel_size=5.0)
                        pcd_crop = open3d.geometry.voxel_down_sample(pcd_crop, voxel_size=5.0)

                        # open3d.draw_geometries([pcd_crop, pcd_model])
                        reg_p2p, _, _, _ = get_evaluation_kiru(pcd_model, pcd_crop, 50, guess, 5,
                                                               model_dia[cls] * 1000.0)
                        R_est = reg_p2p[:3, :3]
                        t_est = reg_p2p[:3, 3] * 0.001
                        '''

                        # ----- Visualization
                        '''

                        tDbox = R_gt.dot(obj_points.T).T
                        tDbox = tDbox + np.repeat(t_gt[np.newaxis, :], 8, axis=0)
                        box3D = toPix_array(tDbox)
                        tDbox = np.reshape(box3D, (16))
                        tDbox = tDbox.astype(np.uint16)

                        eDbox = R_est.dot(obj_points.T).T
                        eDbox = eDbox + np.repeat(t_est[np.newaxis, :], 8, axis=0)
                        ebox3D = toPix_array(eDbox)
                        eDbox = np.reshape(ebox3D, (16))
                        eDbox = eDbox.astype(np.uint16)

                        colGT = (0, 223, 255)
                        colEst = (255, 112, 132)

                        #cv2.rectangle(image, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])),
                        #              (255, 255, 255), 2)


                        image_viz = cv2.line(image_viz, tuple(tDbox[0:2].ravel()), tuple(tDbox[2:4].ravel()), colGT, 2)
                        image_viz = cv2.line(image_viz, tuple(tDbox[2:4].ravel()), tuple(tDbox[4:6].ravel()), colGT, 2)
                        image_viz = cv2.line(image_viz, tuple(tDbox[4:6].ravel()), tuple(tDbox[6:8].ravel()), colGT,
                                         2)
                        image_viz = cv2.line(image_viz, tuple(tDbox[6:8].ravel()), tuple(tDbox[0:2].ravel()), colGT,
                                         2)
                        image_viz = cv2.line(image_viz, tuple(tDbox[0:2].ravel()), tuple(tDbox[8:10].ravel()), colGT,
                                         2)
                        image_viz = cv2.line(image_viz, tuple(tDbox[2:4].ravel()), tuple(tDbox[10:12].ravel()), colGT,
                                         2)
                        image_viz = cv2.line(image_viz, tuple(tDbox[4:6].ravel()), tuple(tDbox[12:14].ravel()), colGT,
                                         2)
                        image_viz = cv2.line(image_viz, tuple(tDbox[6:8].ravel()), tuple(tDbox[14:16].ravel()), colGT,
                                         2)
                        image_viz = cv2.line(image_viz, tuple(tDbox[8:10].ravel()), tuple(tDbox[10:12].ravel()),
                                         colGT,
                                         2)
                        image_viz = cv2.line(image_viz, tuple(tDbox[10:12].ravel()), tuple(tDbox[12:14].ravel()),
                                         colGT,
                                         2)
                        image_viz = cv2.line(image_viz, tuple(tDbox[12:14].ravel()), tuple(tDbox[14:16].ravel()),
                                         colGT,
                                         2)
                        image_viz = cv2.line(image_viz, tuple(tDbox[14:16].ravel()), tuple(tDbox[8:10].ravel()),
                                         colGT,
                                         2)

                        image_viz = cv2.line(image_viz, tuple(eDbox[0:2].ravel()), tuple(eDbox[2:4].ravel()), colEst, 2)
                        image_viz = cv2.line(image_viz, tuple(eDbox[2:4].ravel()), tuple(eDbox[4:6].ravel()), colEst, 2)
                        image_viz = cv2.line(image_viz, tuple(eDbox[4:6].ravel()), tuple(eDbox[6:8].ravel()), colEst, 2)
                        image_viz = cv2.line(image_viz, tuple(eDbox[6:8].ravel()), tuple(eDbox[0:2].ravel()), colEst, 2)
                        image_viz = cv2.line(image_viz, tuple(eDbox[0:2].ravel()), tuple(eDbox[8:10].ravel()), colEst, 2)
                        image_viz = cv2.line(image_viz, tuple(eDbox[2:4].ravel()), tuple(eDbox[10:12].ravel()), colEst, 2)
                        image_viz = cv2.line(image_viz, tuple(eDbox[4:6].ravel()), tuple(eDbox[12:14].ravel()), colEst, 2)
                        image_viz = cv2.line(image_viz, tuple(eDbox[6:8].ravel()), tuple(eDbox[14:16].ravel()), colEst, 2)
                        image_viz = cv2.line(image_viz, tuple(eDbox[8:10].ravel()), tuple(eDbox[10:12].ravel()), colEst,
                                         2)
                        image_viz = cv2.line(image_viz, tuple(eDbox[10:12].ravel()), tuple(eDbox[12:14].ravel()), colEst,
                                         2)
                        image_viz = cv2.line(image_viz, tuple(eDbox[12:14].ravel()), tuple(eDbox[14:16].ravel()), colEst,
                                         2)
                        image_viz = cv2.line(image_viz, tuple(eDbox[14:16].ravel()), tuple(eDbox[8:10].ravel()), colEst,
                                         2)

                        '''
                        if not math.isnan(rd):
                            if rd < 5.0 and xyz < 0.05:
                                less5[cls - 1] += 1

                        err_repr = reproj(K, R_est, t_est, R_gt, t_gt, model_vsd["pts"])

                        if not math.isnan(err_repr):
                            if err_repr < 5.0:
                                rep_less5[cls - 1] += 1

                        if cls == 3 or cls == 7 or cls == 10 or cls == 11:
                            err_add = adi(R_est, t_est, R_gt, t_gt, model_vsd["pts"])
                        else:
                            err_add = add(R_est, t_est, R_gt, t_gt, model_vsd["pts"])

                        print(' ')
                        print('error: ', err_add, 'threshold', model_dia[cls] * 0.1)

                        if not math.isnan(err_add):
                            if err_add < (model_dia[cls - 1] * 0.05):
                                add_less_d005[cls - 1] += 1
                            if err_add < (model_dia[cls - 1] * 0.1):
                                add_less_d[cls - 1] += 1
                            if err_add < (model_dia[cls - 1] * 0.15):
                                add_less_d015[cls - 1] += 1
                            if err_add < (model_dia[cls - 1] * 0.2):
                                add_less_d02[cls - 1] += 1
                            if err_add < (model_dia[cls - 1] * 0.25):
                                add_less_d025[cls - 1] += 1
                            if err_add < (model_dia[cls - 1] * 0.3):
                                add_less_d03[cls - 1] += 1
                            if err_add < (model_dia[cls - 1] * 0.35):
                                add_less_d035[cls - 1] += 1
                            if err_add < (model_dia[cls - 1] * 0.4):
                                add_less_d04[cls - 1] += 1
                            if err_add < (model_dia[cls - 1] * 0.45):
                                add_less_d045[cls - 1] += 1
                            if err_add < (model_dia[cls - 1] * 0.5):
                                add_less_d05[cls - 1] += 1

                        if not math.isnan(err_add):
                            if err_add < (model_dia[cls - 1] * 0.15):
                                tp_add[cls - 1] += 1
                                fn_add[cls - 1] -= 1
                    else:
                        fp[cls] += 1
                        fp_add[cls] += 1

                        fp05[cls] += 1
                        fp10[cls] += 1
                        fp15[cls] += 1
                        fp20[cls] += 1
                        fp25[cls] += 1
                        fp30[cls] += 1
                        fp35[cls] += 1
                        fp40[cls] += 1
                        fp45[cls] += 1
                        fp55[cls] += 1
                        fp6[cls] += 1
                        fp65[cls] += 1
                        fp7[cls] += 1
                        fp75[cls] += 1
                        fp8[cls] += 1
                        fp85[cls] += 1
                        fp9[cls] += 1
                        fp925[cls] += 1
                        fp95[cls] += 1
                        fp975[cls] += 1
                else:
                    fp[cls] += 1
                    fp_add[cls] += 1

                    fp05[cls] += 1
                    fp10[cls] += 1
                    fp15[cls] += 1
                    fp20[cls] += 1
                    fp25[cls] += 1
                    fp30[cls] += 1
                    fp35[cls] += 1
                    fp40[cls] += 1
                    fp45[cls] += 1
                    fp55[cls] += 1
                    fp6[cls] += 1
                    fp65[cls] += 1
                    fp7[cls] += 1
                    fp75[cls] += 1
                    fp8[cls] += 1
                    fp85[cls] += 1
                    fp9[cls] += 1
                    fp925[cls] += 1
                    fp95[cls] += 1
                    fp975[cls] += 1

        #print(raw_dep_path)
        vis_name='/home/sthalham/occlusion_viz/' + raw_dep_path[-17:-12] + '.jpg'
        name_est = '/home/sthalham/visTests/detected_est.jpg'
        cv2.imwrite(vis_name, image_viz)
        print('Stop')

        # append image to list of processed images
        image_ids.append(generator.image_ids[index])
        image_indices.append(index)
        idx += 1

    print(len(image_ids))

    if not len(results):
        return

    # write output
    json.dump(results, open('{}_bbox_results.json'.format(generator.set_name), 'w'), indent=4)
    #json.dump(image_ids, open('{}_processed_image_ids.json'.format(generator.set_name), 'w'), indent=4)

    detPre = [0.0] * 16
    detRec = [0.0] * 16
    detPre_add = [0.0] * 16
    detRec_add = [0.0] * 16
    F1_add = [0.0] * 16
    less_55 = [0.0] * 16
    less_repr_5 = [0.0] * 16
    less_add_d = [0.0] * 16
    less_vsd_t = [0.0] * 16

    less_add_d005 = [0.0] * 16
    less_add_d015 = [0.0] * 16
    less_add_d02 = [0.0] * 16
    less_add_d025 = [0.0] * 16
    less_add_d03 = [0.0] * 16
    less_add_d035 = [0.0] * 16
    less_add_d04 = [0.0] * 16
    less_add_d045 = [0.0] * 16
    less_add_d05 = [0.0] * 16
    less_add_d055 = [0.0] * 16
    less_add_d06 = [0.0] * 16
    less_add_d065 = [0.0] * 16
    less_add_d07 = [0.0] * 16
    less_add_d075 = [0.0] * 16
    less_add_d08 = [0.0] * 16
    less_add_d085 = [0.0] * 16
    less_add_d09 = [0.0] * 16
    less_add_d095 = [0.0] * 16
    less_add_d1 = [0.0] * 16
    np.set_printoptions(precision=2)
    print('')
    for ind in range(1, 16):
        if ind == 0:
            continue

        if ind == 2 or ind == 3 or ind == 4 or ind == 7 or ind == 13 or ind == 14 or ind == 15:
        #if tp[ind] == 0:
            detPre[ind] = 0.0
            detRec[ind] = 0.0
            detPre_add[ind] = 0.0
            detRec_add[ind] = 0.0
            less_55[ind] = 0.0
            less_repr_5[ind] = 0.0
            less_add_d[ind] = 0.0
            less_vsd_t[ind] = 0.0
        else:
            detRec[ind] = tp[ind] / (tp[ind] + fn[ind]) * 100.0
            detPre[ind] = tp[ind] / (tp[ind] + fp[ind]) * 100.0
            detRec_add[ind] = tp_add[ind] / (tp_add[ind] + fn_add[ind]) * 100.0
            detPre_add[ind] = tp_add[ind] / (tp_add[ind] + fp_add[ind]) * 100.0
            F1_add[ind] = 2 * ((detPre_add[ind] * detRec_add[ind])/(detPre_add[ind] + detRec_add[ind]))
            less_55[ind] = (less5[ind]) / (rotD[ind]) * 100.0
            less_repr_5[ind] = (rep_less5[ind]) / (rep_e[ind]) * 100.0
            less_add_d[ind] = (add_less_d[ind]) / (add_e[ind]) * 100.0
            less_vsd_t[ind] = (vsd_less_t[ind]) / (vsd_e[ind]) * 100.0

            less_add_d005[ind] = (add_less_d005[ind]) / (add_e[ind]) * 100.0
            less_add_d015[ind] = (add_less_d015[ind]) / (add_e[ind]) * 100.0
            less_add_d02[ind] = (add_less_d02[ind]) / (add_e[ind]) * 100.0
            less_add_d025[ind] = (add_less_d025[ind]) / (add_e[ind]) * 100.0
            less_add_d03[ind] = (add_less_d03[ind]) / (add_e[ind]) * 100.0
            less_add_d035[ind] = (add_less_d035[ind]) / (add_e[ind]) * 100.0
            less_add_d04[ind] = (add_less_d04[ind]) / (add_e[ind]) * 100.0
            less_add_d045[ind] = (add_less_d045[ind]) / (add_e[ind]) * 100.0
            less_add_d05[ind] = (add_less_d05[ind]) / (add_e[ind]) * 100.0
            less_add_d055[ind] = (add_less_d055[ind]) / (add_e[ind]) * 100.0
            less_add_d06[ind] = (add_less_d06[ind]) / (add_e[ind]) * 100.0
            less_add_d065[ind] = (add_less_d065[ind]) / (add_e[ind]) * 100.0
            less_add_d07[ind] = (add_less_d07[ind]) / (add_e[ind]) * 100.0
            less_add_d075[ind] = (add_less_d075[ind]) / (add_e[ind]) * 100.0
            less_add_d08[ind] = (add_less_d08[ind]) / (add_e[ind]) * 100.0
            less_add_d085[ind] = (add_less_d085[ind]) / (add_e[ind]) * 100.0
            less_add_d09[ind] = (add_less_d09[ind]) / (add_e[ind]) * 100.0
            less_add_d095[ind] = (add_less_d095[ind]) / (add_e[ind]) * 100.0
            less_add_d1[ind] = (add_less_d1[ind]) / (add_e[ind]) * 100.0

            print('cat', ind)
            print('add < 0.05: ', less_add_d005[ind])
            print('add < 0.1: ', less_add_d[ind])
            print('add < 0.15: ', less_add_d015[ind])
            print('add < 0.2: ', less_add_d02[ind])
            print('add < 0.25: ', less_add_d025[ind])
            print('add < 0.3: ', less_add_d03[ind])
            print('add < 0.35: ', less_add_d035[ind])
            print('add < 0.4: ', less_add_d04[ind])
            print('add < 0.45: ', less_add_d045[ind])
            print('add < 0.5: ', less_add_d05[ind])
            print('add < 0.55: ', less_add_d055[ind])
            print('add < 0.6: ', less_add_d06[ind])
            print('add < 0.65: ', less_add_d065[ind])
            print('add < 0.7: ', less_add_d07[ind])
            print('add < 0.75: ', less_add_d075[ind])
            print('add < 0.8: ', less_add_d08[ind])
            print('add < 0.85: ', less_add_d085[ind])
            print('add < 0.9: ', less_add_d09[ind])
            print('add < 0.95: ', less_add_d095[ind])
            print('add < 1: ', less_add_d1[ind])

        print('cat', ind)
        print('add < 0.05: ', less_add_d005[ind])
        print('add < 0.1: ', less_add_d[ind])
        print('add < 0.15: ', less_add_d015[ind])
        print('add < 0.2: ', less_add_d02[ind])
        print('add < 0.25: ', less_add_d025[ind])
        print('add < 0.3: ', less_add_d03[ind])
        print('add < 0.35: ', less_add_d035[ind])
        print('add < 0.4: ', less_add_d04[ind])
        print('add < 0.45: ', less_add_d045[ind])
        print('add < 0.5: ', less_add_d05[ind])

        print('cat ', ind, ' rec ', detPre[ind], ' pre ', detRec[ind], ' less5 ', less_55[ind], ' repr ',
                  less_repr_5[ind], ' add ', less_add_d[ind], ' vsd ', less_vsd_t[ind], ' F1 add 0.15d ', F1_add[ind])

    dataset_recall = sum(tp) / (sum(tp) + sum(fp)) * 100.0
    dataset_precision = sum(tp) / (sum(tp) + sum(fn)) * 100.0
    dataset_recall_add = sum(tp_add) / (sum(tp_add) + sum(fp_add)) * 100.0
    dataset_precision_add = sum(tp_add) / (sum(tp_add) + sum(fn_add)) * 100.0
    F1_add_all = 2 * ((dataset_precision_add * dataset_recall_add)/(dataset_precision_add + dataset_recall_add))
    less_55 = sum(less5) / sum(rotD) * 100.0
    less_repr_5 = sum(rep_less5) / sum(rep_e) * 100.0
    less_add_d = sum(add_less_d) / sum(add_e) * 100.0
    less_vsd_t = sum(vsd_less_t) / sum(vsd_e) * 100.0

    print('IoU 005: ', sum(tp05) / (sum(tp05) + sum(fp05)) * 100.0, sum(tp05) / (sum(tp05) + sum(fn05)) * 100.0)
    print('IoU 010: ', sum(tp10) / (sum(tp10) + sum(fp10)) * 100.0, sum(tp10) / (sum(tp10) + sum(fn10)) * 100.0)
    print('IoU 015: ', sum(tp15) / (sum(tp15) + sum(fp15)) * 100.0, sum(tp15) / (sum(tp15) + sum(fn15)) * 100.0)
    print('IoU 020: ', sum(tp20) / (sum(tp20) + sum(fp20)) * 100.0, sum(tp20) / (sum(tp20) + sum(fn20)) * 100.0)
    print('IoU 025: ', sum(tp25) / (sum(tp25) + sum(fp25)) * 100.0, sum(tp25) / (sum(tp25) + sum(fn25)) * 100.0)
    print('IoU 030: ', sum(tp30) / (sum(tp30) + sum(fp30)) * 100.0, sum(tp30) / (sum(tp30) + sum(fn30)) * 100.0)
    print('IoU 035: ', sum(tp35) / (sum(tp35) + sum(fp35)) * 100.0, sum(tp35) / (sum(tp35) + sum(fn35)) * 100.0)
    print('IoU 040: ', sum(tp40) / (sum(tp40) + sum(fp40)) * 100.0, sum(tp40) / (sum(tp40) + sum(fn40)) * 100.0)
    print('IoU 045: ', sum(tp45) / (sum(tp45) + sum(fp45)) * 100.0, sum(tp45) / (sum(tp45) + sum(fn45)) * 100.0)
    print('IoU 05: ', sum(tp) / (sum(tp) + sum(fp)) * 100.0, sum(tp) / (sum(tp) + sum(fn)) * 100.0)
    print('IoU 055: ', sum(tp55) / (sum(tp55) + sum(fp55)) * 100.0, sum(tp55) / (sum(tp55) + sum(fn55)) * 100.0)
    print('IoU 06: ', sum(tp6) / (sum(tp6) + sum(fp6)) * 100.0, sum(tp6) / (sum(tp6) + sum(fn6)) * 100.0)
    print('IoU 065: ', sum(tp65) / (sum(tp65) + sum(fp65)) * 100.0, sum(tp65) / (sum(tp65) + sum(fn65)) * 100.0)
    print('IoU 07: ', sum(tp7) / (sum(tp7) + sum(fp7)) * 100.0, sum(tp7) / (sum(tp7) + sum(fn7)) * 100.0)
    print('IoU 075: ', sum(tp75) / (sum(tp75) + sum(fp75)) * 100.0, sum(tp75) / (sum(tp75) + sum(fn75)) * 100.0)
    print('IoU 08: ', sum(tp8) / (sum(tp8) + sum(fp8)) * 100.0, sum(tp8) / (sum(tp8) + sum(fn8)) * 100.0)
    print('IoU 085: ', sum(tp85) / (sum(tp85) + sum(fp85)) * 100.0, sum(tp85) / (sum(tp85) + sum(fn85)) * 100.0)
    print('IoU 09: ', sum(tp9) / (sum(tp9) + sum(fp9)) * 100.0, sum(tp9) / (sum(tp9) + sum(fn9)) * 100.0)
    print('IoU 0975: ', sum(tp925) / (sum(tp925) + sum(fp925)) * 100.0, sum(tp925) / (sum(tp925) + sum(fn925)) * 100.0)
    print('IoU 095: ', sum(tp95) / (sum(tp95) + sum(fp95)) * 100.0, sum(tp95) / (sum(tp95) + sum(fn95)) * 100.0)
    print('IoU 0975: ', sum(tp975) / (sum(tp975) + sum(fp975)) * 100.0, sum(tp975) / (sum(tp975) + sum(fn975)) * 100.0)

    return dataset_recall, dataset_precision, less_55, less_vsd_t, less_repr_5, less_add_d, F1_add_all
