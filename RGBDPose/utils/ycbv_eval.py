

import keras
import numpy as np
import json
import math
import transforms3d as tf3d
import geometry
import os
import copy
import cv2
import open3d
from ..utils import ply_loader
from .pose_error import reproj, add, adi, re, te, vsd
import time

import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."


fxkin = 1066.778
fykin = 1067.487
cxkin = 312.9869
cykin = 241.3109


def get_evaluation_kiru(pcd_temp_,pcd_scene_,inlier_thres,tf,final_th, model_dia):#queue
    tf_pcd =np.eye(4)
    pcd_temp_.transform(tf)

    mean_temp = np.mean(np.array(pcd_temp_.points)[:, 2])
    mean_scene = np.median(np.array(pcd_scene_.points)[:, 2])
    pcd_diff = mean_scene - mean_temp

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
    open3d.estimate_normals(pcd_scene_, search_param=open3d.KDTreeSearchParamHybrid(
        radius=5.0, max_nn=10))

    reg_p2p = open3d.registration.registration_icp(pcd_temp_,pcd_scene_ , inlier_thres, np.eye(4),
                                                   open3d.registration.TransformationEstimationPointToPoint(),
                                                   open3d.registration.ICPConvergenceCriteria(max_iteration = 5)) #5?
    tf = np.matmul(reg_p2p.transformation,tf)
    tf_pcd = np.matmul(reg_p2p.transformation,tf_pcd)
    pcd_temp_.transform(reg_p2p.transformation)

    #open3d.estimate_normals(pcd_temp_, search_param=open3d.KDTreeSearchParamHybrid(
    #    radius=2.0, max_nn=30))
    points_unfiltered = np.asarray(pcd_temp_.points)
    last_pcd_temp = []
    for i, normal in enumerate(pcd_temp_.normals):
        if normal[2] < 0:
            last_pcd_temp.append(points_unfiltered[i, :])

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
    #mesh_path = "/home/sthalham/data/LINEMOD/models/"
    mesh_path = "/home/stefan/data/Meshes/ycb_video_st/models/"
    template = '000000'
    lencat = len(cat)
    cat = template[:-lencat] + cat
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


def evaluate_ycbv(generator, model, threshold=0.05):
    threshold = 0.5

    mesh_info = '/home/stefan/data/Meshes/ycb_video_st/models/models_info.json'

    threeD_boxes = np.ndarray((22, 8, 3), dtype=np.float32)
    sym_cont = np.ndarray((22, 3), dtype=np.float32)
    sym_disc = np.ndarray((28, 4, 4), dtype=np.float32)
    model_dia = np.zeros((22), dtype=np.float32)

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
        threeD_boxes[int(key), :, :] = three_box_solo
        model_dia[int(key)] = value['diameter'] * fac

    # start collecting results
    results = []
    image_ids = []
    image_indices = []
    idx = 0

    tp = np.zeros((22), dtype=np.uint32)
    fp = np.zeros((22), dtype=np.uint32)
    fn = np.zeros((22), dtype=np.uint32)

    # interlude end

    tp_add = np.zeros((22), dtype=np.uint32)
    fp_add = np.zeros((22), dtype=np.uint32)
    fn_add = np.zeros((22), dtype=np.uint32)

    rotD = np.zeros((22), dtype=np.uint32)
    less5 = np.zeros((22), dtype=np.uint32)
    rep_e = np.zeros((22), dtype=np.uint32)
    rep_less5 = np.zeros((22), dtype=np.uint32)
    add_e = np.zeros((22), dtype=np.uint32)
    add_less_d = np.zeros((22), dtype=np.uint32)
    vsd_e = np.zeros((22), dtype=np.uint32)
    vsd_less_t = np.zeros((22), dtype=np.uint32)
    
    # target annotation
    pc1, mv1, mv1_mm = load_pcd('01')
    pc2, mv2, mv2_mm = load_pcd('02')
    pc3, mv3, mv3_mm = load_pcd('03')
    pc4, mv4, mv4_mm = load_pcd('04')
    pc5, mv5, mv5_mm = load_pcd('05')
    pc6, mv6, mv6_mm = load_pcd('06')
    pc7, mv7, mv7_mm = load_pcd('07')
    pc8, mv8, mv8_mm = load_pcd('08')
    pc9, mv9, mv9_mm = load_pcd('09')
    pc10, mv10, mv10_mm = load_pcd('10')
    pc11, mv11, mv11_mm = load_pcd('11')
    pc12, mv12, mv12_mm = load_pcd('12')
    pc13, mv13, mv13_mm = load_pcd('13')
    pc14, mv14, mv14_mm = load_pcd('14')
    pc15, mv15, mv15_mm = load_pcd('15')
    pc16, mv16, mv16_mm = load_pcd('16')
    pc17, mv17, mv17_mm = load_pcd('17')
    pc18, mv18, mv18_mm = load_pcd('18')
    pc19, mv19, mv19_mm = load_pcd('19')
    pc20, mv20, mv20_mm = load_pcd('20')
    pc21, mv21, mv21_mm = load_pcd('21')

    for index in progressbar.progressbar(range(generator.size()), prefix='LineMOD evaluation: '):
        image_raw = generator.load_image(index)
        image = generator.preprocess_image(image_raw)
        image, scale = generator.resize_image(image)

        image_raw_dep = generator.load_image_dep(index)
        image_dep = generator.preprocess_image(image_raw_dep)
        image_dep, scale = generator.resize_image(image_dep)

        raw_dep_path = generator.load_image_dep_raw(index)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        anno = generator.load_annotations(index)

        print(anno['labels'])
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

        # correct boxes for image scale
        boxes /= scale

        # change to (x, y, w, h) (MS COCO standard)
        boxes[:, :, 2] -= boxes[:, :, 0]
        boxes[:, :, 3] -= boxes[:, :, 1]

        rotD[t_cat] += 1
        rep_e[t_cat] += 1
        add_e[t_cat] += 1
        vsd_e[t_cat] += 1
        fn[t_cat] += 1

        # end interlude
        fn_add[t_cat] += 1
        fnit = np.ones((22), dtype=np.bool)

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
            #cls = 1
            #control_points = box3D[(cls - 1), :]
            control_points = box3D

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
                if IoU > 0.5:
                    if fnit[cls] == True:
                        # interlude

                        tp[cls] += 1
                        fn[cls] -= 1

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


                        cur_pose = gt_poses[odx[0]]
                        t_rot = cur_pose[0][3:]
                        t_tra = cur_pose[0][:3]

                        t_rot = tf3d.euler.euler2mat(t_rot[0], t_rot[1], t_rot[2])
                        R_gt = np.array(t_rot, dtype=np.float32).reshape(3, 3)
                        t_gt = np.array(t_tra, dtype=np.float32) * 0.001

                        rd = re(R_gt, R_est)
                        xyz = te(t_gt, t_est.T)
                        #print(control_points)

                        #tDbox = R_gt.dot(obj_points.T).T
                        #tDbox = tDbox + np.repeat(t_gt[np.newaxis, :], 8, axis=0)
                        #box3D = toPix_array(tDbox)
                        #tDbox = np.reshape(box3D, (16))
                        #print(tDbox)

                        if cls == 1:
                            model_vsd = mv1
                            pcd_model = pc1
                        elif cls == 2:
                            model_vsd = mv2
                            pcd_model = pc2
                        elif cls == 3:
                            model_vsd = mv3
                            pcd_model = pc3
                        elif cls == 4:
                            model_vsd = mv4
                            pcd_model = pc4
                        elif cls == 5:
                            model_vsd = mv5
                            pcd_model = pc5
                        elif cls == 6:
                            model_vsd = mv6
                            pcd_model = pc6
                        elif cls == 7:
                            model_vsd = mv7
                            pcd_model = pc7
                        elif cls == 8:
                            model_vsd = mv8
                            pcd_model = pc8
                        elif cls == 9:
                            model_vsd = mv9
                            pcd_model = pc9
                        elif cls == 10:
                            model_vsd = mv10
                            pcd_model = pc10
                        elif cls == 11:
                            model_vsd = mv11
                            pcd_model = pc11
                        elif cls == 12:
                            model_vsd = mv12
                            pcd_model = pc12
                        elif cls == 13:
                            model_vsd = mv13
                            pcd_model = pc13
                        elif cls == 14:
                            model_vsd = mv14
                            pcd_model = pc14
                        elif cls == 15:
                            model_vsd = mv15
                            pcd_model = pc15
                        elif cls == 16:
                            model_vsd = mv16
                            pcd_model = pc16
                        elif cls == 17:
                            model_vsd = mv17
                            pcd_model = pc17
                        elif cls == 18:
                            model_vsd = mv18
                            pcd_model = pc18
                        elif cls == 19:
                            model_vsd = mv19
                            pcd_model = pc19
                        elif cls == 20:
                            model_vsd = mv20
                            pcd_model = pc20
                        elif cls == 21:
                            model_vsd = mv21
                            pcd_model = pc21

                        print('--------------------- ICP refinement -------------------')
                        #print(raw_dep_path)
                        image_dep = cv2.imread(raw_dep_path, cv2.IMREAD_UNCHANGED)
                        image_icp = np.multiply(image_dep, 0.1)

                        pcd_img = create_point_cloud(image_icp, fxkin, fykin, cxkin, cykin, 1.0)
                        pcd_img = pcd_img.reshape((480, 640, 3))[int(b1[1]):int(b1[3]), int(b1[0]):int(b1[2]), :]
                        pcd_img = pcd_img.reshape((pcd_img.shape[0] * pcd_img.shape[1], 3))
                        pcd_crop = open3d.PointCloud()
                        pcd_crop.points = open3d.Vector3dVector(pcd_img)
                        open3d.estimate_normals(pcd_crop, search_param=open3d.KDTreeSearchParamHybrid(
                            radius=2.0, max_nn=30))

                        # pcd_crop.paint_uniform_color(np.array([0.99, 0.0, 0.00]))
                        # open3d.draw_geometries([pcd_crop])

                        guess = np.zeros((4, 4), dtype=np.float32)
                        guess[:3, :3] = R_est
                        guess[:3, 3] = t_est.T * 1000.0
                        guess[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32).T

                        pcd_model = open3d.geometry.voxel_down_sample(pcd_model, voxel_size=5.0)
                        pcd_crop = open3d.geometry.voxel_down_sample(pcd_crop, voxel_size=5.0)

                        #open3d.draw_geometries([pcd_crop, pcd_model])
                        reg_p2p, _, _, _ =get_evaluation_kiru(pcd_model, pcd_crop, 50, guess, 5, model_dia[cls]*0.001)
                        R_est = reg_p2p[:3, :3]
                        t_est = reg_p2p[:3, 3]

                        '''
                                                pose = est_points.reshape((16)).astype(np.int16)
                                                bb = b1

                                                tDbox = R_gt.dot(obj_points.T).T
                                                tDbox = tDbox + np.repeat(t_gt[np.newaxis, :], 8, axis=0)
                                                box3D = toPix_array(tDbox)
                                                tDbox = np.reshape(box3D, (16))
                                                tDbox = tDbox.astype(np.uint16)

                                                colGT = (0, 128, 0)
                                                colEst = (255, 0, 0)

                                                cv2.rectangle(image, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])),
                                                              (255, 255, 255), 2)

                                                image = cv2.line(image, tuple(tDbox[0:2].ravel()), tuple(tDbox[2:4].ravel()), colGT, 3)
                                                image = cv2.line(image, tuple(tDbox[2:4].ravel()), tuple(tDbox[4:6].ravel()), colGT, 3)
                                                image = cv2.line(image, tuple(tDbox[4:6].ravel()), tuple(tDbox[6:8].ravel()), colGT,
                                                                 3)
                                                image = cv2.line(image, tuple(tDbox[6:8].ravel()), tuple(tDbox[0:2].ravel()), colGT,
                                                                 3)
                                                image = cv2.line(image, tuple(tDbox[0:2].ravel()), tuple(tDbox[8:10].ravel()), colGT,
                                                                 3)
                                                image = cv2.line(image, tuple(tDbox[2:4].ravel()), tuple(tDbox[10:12].ravel()), colGT,
                                                                 3)
                                                image = cv2.line(image, tuple(tDbox[4:6].ravel()), tuple(tDbox[12:14].ravel()), colGT,
                                                                 3)
                                                image = cv2.line(image, tuple(tDbox[6:8].ravel()), tuple(tDbox[14:16].ravel()), colGT,
                                                                 3)
                                                image = cv2.line(image, tuple(tDbox[8:10].ravel()), tuple(tDbox[10:12].ravel()),
                                                                 colGT,
                                                                 3)
                                                image = cv2.line(image, tuple(tDbox[10:12].ravel()), tuple(tDbox[12:14].ravel()),
                                                                 colGT,
                                                                 3)
                                                image = cv2.line(image, tuple(tDbox[12:14].ravel()), tuple(tDbox[14:16].ravel()),
                                                                 colGT,
                                                                 3)
                                                image = cv2.line(image, tuple(tDbox[14:16].ravel()), tuple(tDbox[8:10].ravel()),
                                                                 colGT,
                                                                 3)

                                                image = cv2.line(image, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), colEst, 5)
                                                image = cv2.line(image, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), colEst, 5)
                                                image = cv2.line(image, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), colEst, 5)
                                                image = cv2.line(image, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), colEst, 5)
                                                image = cv2.line(image, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), colEst, 5)
                                                image = cv2.line(image, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), colEst, 5)
                                                image = cv2.line(image, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), colEst, 5)
                                                image = cv2.line(image, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), colEst, 5)
                                                image = cv2.line(image, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()), colEst,
                                                                 5)
                                                image = cv2.line(image, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), colEst,
                                                                 5)
                                                image = cv2.line(image, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), colEst,
                                                                 5)
                                                image = cv2.line(image, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), colEst,
                                                                 5)

                                                name = '/home/stefan/detection_ycbv.jpg'
                                                cv2.imwrite(name, image)

                                                print('break')
                                                '''

                        if not math.isnan(rd):
                            if rd < 5.0 and xyz < 0.05:
                                less5[cls - 1] += 1

                        err_repr = reproj(K, R_est, t_est, R_gt, t_gt, model_vsd["pts"])

                        if not math.isnan(err_repr):
                            if err_repr < 5.0:
                                rep_less5[cls - 1] += 1

                        print(np.nanmax(model_vsd["pts"]), t_est, t_gt*1000.0)

                        if cls in [1, 13, 16, 18, 19, 20, 21]:
                            err_add = adi(R_est, t_est, R_gt, t_gt*1000.0, model_vsd["pts"])
                        else:
                            err_add = add(R_est, t_est, R_gt, t_gt*1000.0, model_vsd["pts"])

                        print(' ')
                        print('error: ', err_add, 'threshold', model_dia[cls] * 100.0) # 0.1 * 1000.0

                        if not math.isnan(err_add):

                            if err_add < (model_dia[cls] ):
                                add_less_d1[cls] += 1

                        if not math.isnan(err_add):
                            if err_add < (model_dia[cls] * 0.15):
                                tp_add[cls] += 1
                                fn_add[cls] -= 1

                        if not math.isnan(err_add):
                            if err_add < (model_dia[cls - 1] * 0.1):
                                add_less_d[cls - 1] += 1

                        if not math.isnan(err_add):
                            if err_add < (model_dia[cls - 1] * 0.15):
                                tp_add[cls - 1] += 1
                                fn_add[cls - 1] -= 1
                    else:
                        fp[cls] += 1
                        fp_add[cls] += 1

                else:
                    fp[cls] += 1
                    fp_add[cls] += 1

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

    np.set_printoptions(precision=2)
    print('')
    for ind in range(1, 16):
        if ind == 0:
            continue

        if tp[ind] == 0:
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

            print('cat', ind)
            print('add < 0.1: ', less_add_d[ind])

        print('cat', ind)
        print('add < 0.1: ', less_add_d[ind])

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

    print('IoU 05: ', sum(tp) / (sum(tp) + sum(fp)) * 100.0, sum(tp) / (sum(tp) + sum(fn)) * 100.0)

    return dataset_recall, dataset_precision, less_55, less_vsd_t, less_repr_5, less_add_d, F1_add_all
