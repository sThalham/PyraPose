
#from pycocotools.cocoeval import COCOeval

import numpy as np
import transforms3d as tf3d
import copy
import cv2
import open3d
from ..utils import ply_loader
from .pose_error import reproj, add, adi, re, te, vsd
import yaml
import os
import time
from ..utils.anchors import anchors_for_shape

from PIL import Image

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
    #a = np.array([a[0], a[1], a[0] + a[2], a[1] + a[3]])
    #b = np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]])
    a = np.array([a[0], a[1], a[2], a[3]])
    b = np.array([b[0], b[1], b[2], b[3]])

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


def evaluate_custom(generator, model, threshold=0.5):

    test_path = generator
    mesh_info = '/home/stefan/data/Meshes/metal_Markus/models_info.yml'
    mesh_path = '/home/stefan/data/Meshes/metal_Markus/plate_final.ply'
    results_path = '/home/stefan/data/metal_Markus/occaug_results_26012021_m68'

    '''
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

    threeD_boxes = np.ndarray((2, 8, 3), dtype=np.float32)
    threeD_boxes[1, :, :] = np.array([[0.015, 0.105, 0.035],  # Metal [30, 210, 70] links
                                      [0.015, 0.105, -0.035],
                                      [0.015, -0.105, -0.035],
                                      [0.015, -0.105, 0.035],
                                      [-0.015, 0.105, 0.035],
                                      [-0.015, 0.105, -0.035],
                                      [-0.015, -0.105, -0.035],
                                      [-0.015, -0.105, 0.035]])

    model_vsd = ply_loader.load_ply(mesh_path)
    pcd_model = open3d.geometry.PointCloud()
    #pcd_model = open3d.io.read_point_cloud(mesh_path)
    pcd_model.points = open3d.utility.Vector3dVector(model_vsd['pts'])
    print('max model: ', np.nanmax(model_vsd['pts']))
    print('min model: ', np.nanmin(model_vsd['pts']))

    anchor_params = anchors_for_shape((480, 640))
    print(anchor_params.shape)

    for img_idx, img_name in enumerate(os.listdir(test_path)):
        img_path = os.path.join(test_path, img_name)

        print('------------------------------------')
        print('processing image: ', img_path)

        image_raw = cv2.imread(img_path, 1)
        image_mask = copy.deepcopy(image_raw)
        image_mask = cv2.resize(image_mask, (640, 480))
        image_pose = copy.deepcopy(image_mask)
        image_pose_rep = copy.deepcopy(image_mask)
        image_ori = copy.deepcopy(image_mask)

        image = image_raw.astype(np.float32)
        image[..., 0] -= 103.939
        image[..., 1] -= 116.779
        image[..., 2] -= 123.68
        image = cv2.resize(image, (640, 480))

        boxes3D, scores, mask = model.predict_on_batch(np.expand_dims(image, axis=0))

        clust_t = time.time()
        for inv_cls in range(scores.shape[2]):
            cls = inv_cls + 1
            cls_mask = scores[0, :, inv_cls]
            obj_mask = mask[0, :, inv_cls]

            cls_indices = np.where(cls_mask > threshold)
            if len(cls_indices[0]) < 1:
                continue

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
                #print('ind_anchors: ', ind_anchors)
                same_obj = True
                while same_obj == True:
                    # update matrices based on iou
                    same_obj = False
                    indcs2rm = []
                    for adx in range(pos_anchors.shape[0]):
                        # loop through anchors
                        box_b = pos_anchors[adx, :]
                        if not np.all((box_b > 0)): # need x_max or y_max here? maybe irrelevant due to positivity
                            indcs2rm.append(adx)
                            continue
                        for qdx in range(len(obj_ancs)):
                            # loop through anchors belonging to instance
                            iou = boxoverlap(obj_ancs[qdx], box_b)
                            if iou > 0.4:
                                #print('anc_anchors: ', pos_anchors)
                                #print('ind_anchors: ', ind_anchors)
                                #print('adx: ', adx)
                                obj_ancs.append(box_b)
                                obj_inds.append(ind_anchors[adx])
                                indcs2rm.append(adx)
                                same_obj = True
                                break
                        if same_obj == True:
                            break

                    #print('pos_anchors: ', pos_anchors.shape)
                    #print('ind_anchors: ', len(ind_anchors))
                    #print('indcs2rm: ', indcs2rm)
                    pos_anchors = np.delete(pos_anchors, indcs2rm, axis=0)
                    ind_anchors = np.delete(ind_anchors, indcs2rm, axis=0)

                print('obj_inds per instance: ', obj_inds)
                per_obj_hyps.append(obj_inds)

                #obj_col = (np.random.random() * 255, np.random.random() * 255, np.random.random() * 255)
                #for bb in obj_ancs:
                #    cv2.rectangle(image_ori, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])),
                #          (int(obj_col[0]), int(obj_col[1]), int(obj_col[2])), 3)
            #print('separate objects time: ', time.time() - clust_t)

            #print('per_obj_hyps: ', len(per_obj_hyps))

            cls_img = np.where(obj_mask > 0.5, 1, 0)
            cls_img = cls_img.reshape((60, 80)).astype(np.uint8)
            cls_img = np.asarray(Image.fromarray(cls_img).resize((640, 480), Image.NEAREST))
            cls_img = np.repeat(cls_img[:, :, np.newaxis], 3, 2)
            cls_img = cls_img.astype(np.uint8)
            cls_img[:, :, 0] *= 0
            cls_img[:, :, 1] *= 215
            cls_img[:, :, 2] *= 255
            image_mask = np.where(cls_img > 0, cls_img, image_mask)

            for per_ins_indices in per_obj_hyps:

                print('per_ins: ', per_ins_indices)
                print('len per_ins', len(per_ins_indices))
                k_hyp = len(per_ins_indices)
                ori_points = np.ascontiguousarray(threeD_boxes[cls, :, :], dtype=np.float32)  # .reshape((8, 1, 3))
                K = np.float32([fxkin, 0., cxkin, 0., fykin, cykin, 0., 0., 1.]).reshape(3, 3)

                ##############################
                # pnp
                pose_votes = boxes3D[0, per_ins_indices, :]
                print('pose votes: ', pose_votes.shape)
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

                eDbox = R_est.dot(ori_points.T).T
                # eDbox = eDbox + np.repeat(t_est[:, np.newaxis], 8, axis=1).T
                eDbox = eDbox + np.repeat(t_est, 8, axis=1).T
                est3D = toPix_array(eDbox)
                eDbox = np.reshape(est3D, (16))
                pose = eDbox.astype(np.uint16)

                colEst = (255, 0, 0)

                image_pose = cv2.line(image_pose, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), colEst, 3)
                image_pose = cv2.line(image_pose, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), colEst, 3)
                image_pose = cv2.line(image_pose, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), colEst, 3)
                image_pose = cv2.line(image_pose, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), colEst, 3)
                image_pose = cv2.line(image_pose, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), colEst, 3)
                image_pose = cv2.line(image_pose, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), colEst, 3)
                image_pose = cv2.line(image_pose, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), colEst, 3)
                image_pose = cv2.line(image_pose, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), colEst, 3)
                image_pose = cv2.line(image_pose, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()), colEst,
                                 3)
                image_pose = cv2.line(image_pose, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), colEst,
                                 3)
                image_pose = cv2.line(image_pose, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), colEst,
                                 3)
                image_pose = cv2.line(image_pose, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), colEst,
                                 3)

                #pose_path = os.path.join(results_path, 'pose_' + str(idx) + '.png')
                #cv2.imwrite(pose_path, image_pose)

                guess = np.zeros((4, 4), dtype=np.float32)
                guess[:3, :3] = R_est
                guess[:3, 3] = t_est.T #* 1000.0
                guess[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32).T

                pcd_now = copy.deepcopy(pcd_model)
                pcd_now.transform(guess)
                cloud_points = np.asarray(pcd_now.points)
                model_image = toPix_array(cloud_points)
                for idx in range(model_image.shape[0]):
                    if int(model_image[idx, 1]) > 479 or int(model_image[idx, 0]) > 639 or int(
                            model_image[idx, 1]) < 0 or int(model_image[idx, 0]) < 0:
                        continue
                    image_pose_rep[int(model_image[idx, 1]), int(model_image[idx, 0]), :] = (75, 46, 254)

        ori_mask = np.concatenate([image_ori, image_mask], axis=1)
        box_rep = np.concatenate([image_pose, image_pose_rep], axis=1)
        image_out = np.concatenate([ori_mask, box_rep], axis=0)
        out_path = os.path.join(results_path, 'sample_' + str(img_idx) + '.png')
        cv2.imwrite(out_path, image_out)

    print('Look at what you did... are you proud of yourself?')
