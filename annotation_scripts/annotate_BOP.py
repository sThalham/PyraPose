#!/usr/bin/env python

import sys
import os
import subprocess
import yaml
import cv2
import numpy as np
import json
from scipy import ndimage
import math
import datetime
import copy
import transforms3d as tf3d
import time
from pathlib import Path
import geometry
from shutil import copyfile

depSca = 1.0
resX = 640
resY = 480
fxkin = 572.41140
fykin = 573.57043
cxkin = 325.26110
cykin = 242.04899
depthCut = 2000.0


def draw_axis(img, cam_R, cam_T):
    # unit is mm
    points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)

    rotMat = tf3d.quaternions.quat2mat(cam_R)
    rot, _ = cv2.Rodrigues(rotMat)

    tra = cam_T

    K = np.float32([fxkin, 0., cxkin, 0., fykin, cykin, 0., 0., 1.]).reshape(3,3)

    axisPoints, _ = cv2.projectPoints(points, rot, tra, K, (0, 0, 0, 0))
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255,0,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0,0,255), 3)
    return img


def toPix(translation):

    xpix = ((translation[0] * fxkin) / translation[2]) + cxkin
    ypix = ((translation[1] * fykin) / translation[2]) + cykin
    #zpix = translation[2] * 0.001 * fxkin

    return [xpix, ypix]


def toPix_array(translation, fx=None, fy=None, cx=None, cy=None):

    xpix = ((translation[:, 0] * fx) / translation[:, 2]) + cx
    ypix = ((translation[:, 1] * fy) / translation[:, 2]) + cy
    #zpix = translation[2] * fxkin

    return np.stack((xpix, ypix), axis=1) #, zpix]


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

    return cloud_final


def get_normal(depth_refine, fx=-1, fy=-1, cx=-1, cy=-1, for_vis=True):
    res_y = depth_refine.shape[0]
    res_x = depth_refine.shape[1]

    # inpainting
    scaleOri = np.amax(depth_refine)

    inPaiMa = np.where(depth_refine == 0.0, 255, 0)
    inPaiMa = inPaiMa.astype(np.uint8)
    inPaiDia = 5.0
    depth_refine = depth_refine.astype(np.float32)
    depPaint = cv2.inpaint(depth_refine, inPaiMa, inPaiDia, cv2.INPAINT_NS)

    depNorm = depPaint - np.amin(depPaint)
    rangeD = np.amax(depNorm)
    depNorm = np.divide(depNorm, rangeD)
    depth_refine = np.multiply(depNorm, scaleOri)

    depth_inp = copy.deepcopy(depth_refine)

    centerX = cx
    centerY = cy

    constant = 1 / fx
    uv_table = np.zeros((res_y, res_x, 2), dtype=np.int16)
    column = np.arange(0, res_y)

    uv_table[:, :, 1] = np.arange(0, res_x) - centerX  # x-c_x (u)
    uv_table[:, :, 0] = column[:, np.newaxis] - centerY  # y-c_y (v)
    uv_table_sign = np.copy(uv_table)
    uv_table = np.abs(uv_table)

    # kernel = np.ones((5, 5), np.uint8)
    # depth_refine = cv2.dilate(depth_refine, kernel, iterations=1)
    # depth_refine = cv2.medianBlur(depth_refine, 5 )
    depth_refine = ndimage.gaussian_filter(depth_refine, 2)  # sigma=3)
    # depth_refine = ndimage.uniform_filter(depth_refine, size=11)

    # very_blurred = ndimage.gaussian_filter(face, sigma=5)
    v_x = np.zeros((res_y, res_x, 3))
    v_y = np.zeros((res_y, res_x, 3))
    normals = np.zeros((res_y, res_x, 3))

    dig = np.gradient(depth_refine, 2, edge_order=2)
    v_y[:, :, 0] = uv_table_sign[:, :, 1] * constant * dig[0]
    v_y[:, :, 1] = depth_refine * constant + (uv_table_sign[:, :, 0] * constant) * dig[0]
    v_y[:, :, 2] = dig[0]

    v_x[:, :, 0] = depth_refine * constant + uv_table_sign[:, :, 1] * constant * dig[1]
    v_x[:, :, 1] = uv_table_sign[:, :, 0] * constant * dig[1]
    v_x[:, :, 2] = dig[1]

    cross = np.cross(v_x.reshape(-1, 3), v_y.reshape(-1, 3))
    norm = np.expand_dims(np.linalg.norm(cross, axis=1), axis=1)
    # norm[norm == 0] = 1

    cross = cross / norm
    cross = cross.reshape(res_y, res_x, 3)
    cross = np.abs(cross)
    cross = np.nan_to_num(cross)

    #cross[depth_refine <= 200] = 0  # 0 and near range cut
    cross[depth_refine > depthCut] = 0  # far range cut
    if not for_vis:
        scaDep = 1.0 / np.nanmax(depth_refine)
        depth_refine = np.multiply(depth_refine, scaDep)
        cross[:, :, 0] = cross[:, :, 0] * (1 - (depth_refine - 0.5))  # nearer has higher intensity
        cross[:, :, 1] = cross[:, :, 1] * (1 - (depth_refine - 0.5))
        cross[:, :, 2] = cross[:, :, 2] * (1 - (depth_refine - 0.5))
        scaCro = 255.0 / np.nanmax(cross)
        cross = np.multiply(cross, scaCro)
        cross = cross.astype(np.uint8)

    return cross, depth_refine, depth_inp


def create_BB(rgb):

    imgray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    mask = imgray > 25

    oneA = np.ones(imgray.shape)
    masked = np.where(mask, oneA, 0)

    kernel = np.ones((9, 9), np.uint8)
    mask_dil = cv2.dilate(masked, kernel, iterations=1)

    im2, contours, hier = cv2.findContours(np.uint8(mask_dil), 1, 2)

    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    area = cv2.contourArea(box)

    # cv2.drawContours(rgb, [box], -1, (170, 160, 0), 2)
    # cv2.rectangle(rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
    bb = [int(x),int(y),int(w),int(h)]

    return cnt, bb, area, mask_dil


if __name__ == "__main__":

    dataset = 'linemod'
    root = "/home/stefan/data/datasets/YCBV_BOP_train/"  # path to train samples, depth + rgb
    target = '/home/stefan/data/train_data/ycbv_PBR_BOP/'
    mesh_info = '/home/stefan/data/Meshes/ycb_video/models/models_info.json'
    # print(root)
    visu = False

    threeD_boxes = np.ndarray((31, 8, 3), dtype=np.float32)

    for key, value in json.load(open(mesh_info)).items():
        fac = 0.001
        x_minus = value['min_x']
        y_minus = value['min_y']
        z_minus = value['min_z']
        x_plus = value['size_x'] + x_minus
        y_plus = value['size_y'] + y_minus
        z_plus = value['size_z'] + z_minus
        three_box_solo = np.array([[x_plus, y_plus, z_plus],
                                   [x_plus, y_plus, z_minus],
                                   [x_plus, y_minus, z_minus],
                                   [x_plus, y_minus, z_plus],
                                   [x_minus, y_plus, z_plus],
                                   [x_minus, y_plus, z_minus],
                                   [x_minus, y_minus, z_minus],
                                   [x_minus, y_minus, z_plus]])
        threeD_boxes[int(key), :, :] = three_box_solo

    sub = os.listdir(root)

    now = datetime.datetime.now()
    dateT = str(now)

    dict = {"info": {
                "description": "tless",
                "url": "cmp.felk.cvut.cz/t-less/",
                "version": "1.0",
                "year": 2018,
                "contributor": "Stefan Thalhammer",
                "date_created": dateT
                    },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": []
            }

    annoID = 0

    count = 0
    syns = os.listdir(root)

    for set in syns:
        set_root = os.path.join(root, set)

        rgbPath = set_root + "/rgb/"
        depPath = set_root + "/depth/"
        masPath = set_root + "/mask/"
        visPath = set_root + "/mask_visib/"
        camPath = set_root + "/scene_camera.json"
        gtPath = set_root + "/scene_gt.json"
        infoPath = set_root + "/scene_gt_info.json"

        with open(camPath, 'r') as streamCAM:
            camjson = json.load(streamCAM)

        with open(gtPath, 'r') as streamGT:
            scenejson = json.load(streamGT)

        with open(infoPath, 'r') as streamINFO:
            gtjson = json.load(streamINFO)

        for samp in os.listdir(rgbPath):

            imgname = samp
            rgbImgPath = rgbPath + samp
            depImgPath = depPath + samp[:-4] + '.png'
            visImgPath = visPath + samp[:-4] + '.png'

            if samp.startswith('00000'):
                samp = samp[5:]
            elif samp.startswith('0000'):
                samp = samp[4:]
            elif samp.startswith('000'):
                samp = samp[3:]
            elif samp.startswith('00'):
                samp = samp[2:]
            elif samp.startswith('0'):
                samp = samp[1:]
            samp = samp[:-4]

            calib = camjson.get(str(samp))
            K = calib["cam_K"]
            depSca = calib["depth_scale"]
            fxca = K[0]
            fyca = K[4]
            cxca = K[2]
            cyca = K[5]

            #########################
            # Prepare the stuff
            #########################

            # read images and create mask
            # read images and create mask
            rgbImg = cv2.imread(rgbImgPath)
            depImg = cv2.imread(depImgPath, cv2.IMREAD_UNCHANGED)
            rows, cols = depImg.shape
            depImg = np.multiply(depImg, depSca)

            # create image number and name
            template_samp = '00000'
            imgNum = set + template_samp[:-len(samp)] + samp
            img_id = int(imgNum)
            imgNam = imgNum + '.png'
            iname = str(imgNam)

            gtImg = gtjson.get(str(samp))

            bbox_vis = []
            cat_vis = []
            camR_vis = []
            camT_vis = []
            # if rnd == 1:

            fileName = target + 'images/train/' + imgNam[:-4] + '_dep.png'
            myFile = Path(fileName)
            if myFile.exists():
                print('File exists, skip encoding, ', fileName)
            else:
                imgI = depImg.astype(np.uint16)

                rgb_name = fileName[:-8] + '_rgb.png'
                cv2.imwrite(rgb_name, rgbImg)
                cv2.imwrite(fileName, imgI)
                print("storing image in : ", fileName)

            mask_ind = 0
            mask_img = np.zeros((480, 640), dtype=np.uint8)
            bbvis = []
            cnt = 0
            # bbsca = 720.0 / 640.0
            for i in range(len(gtImg)):
                mask_name = '000000'[:-len(samp)] + samp + '_000000'[:-len(str(mask_ind))] + str(mask_ind) + '.png'
                mask_path = os.path.join(visPath, mask_name)
                obj_mask = cv2.imread(mask_path)[:, :, 0]
                mask_id = mask_ind + 1
                mask_img = np.where(obj_mask > 0, mask_id, mask_img)
                mask_ind = mask_ind + 1

                curlist = gtImg[i]
                obj_bb = curlist["bbox_obj"]
                bbox_vis.append(obj_bb)

                gtPose = scenejson.get(str(samp))
                obj_id = gtPose[i]['obj_id']
                if obj_id == 7 or obj_id == 3:
                    continue

                R = gtPose[i]["cam_R_m2c"]
                T = gtPose[i]["cam_t_m2c"]
                cat_vis.append(obj_id)

                # pose [x, y, z, roll, pitch, yaw]
                R = np.asarray(R, dtype=np.float32)
                rot = tf3d.quaternions.mat2quat(R.reshape(3, 3))
                rot = np.asarray(rot, dtype=np.float32)
                tra = np.asarray(T, dtype=np.float32)
                pose = [np.asscalar(tra[0]), np.asscalar(tra[1]), np.asscalar(tra[2]),
                        np.asscalar(rot[0]), np.asscalar(rot[1]), np.asscalar(rot[2]), np.asscalar(rot[3])]
                camR_vis.append([np.asscalar(rot[0]), np.asscalar(rot[1]), np.asscalar(rot[2])])
                camT_vis.append(tra)

                visib_fract = float(curlist["visib_fract"])

                # if tra[2] > max_obj_dist:
                #    max_obj_dist = tra[2]
                # if tra[2] < min_obj_dist:
                #    min_obj_dist = tra[2]

                area = obj_bb[2] * obj_bb[3]

                trans = np.asarray(T)
                tDbox = R.reshape(3, 3).dot(threeD_boxes[obj_id, :, :].T).T
                tDbox = tDbox + np.repeat(trans[np.newaxis, :], 8, axis=0)
                box3D = toPix_array(tDbox, fx=fxca, fy=fyca, cx=cxca, cy=cyca)
                box3D = np.reshape(box3D, (16))
                box3D = box3D.tolist()

                # tDfea = rot[:3, :3].dot(threeD_feats[cls, :, :].T).T
                # tDfea = tDfea + np.repeat(poses[i, np.newaxis, 0:3], 8, axis=0)
                # fea3D = toPix_array(tDfea, fx=fxkin, fy=fykin, cx=cxkin, cy=cykin)
                # fea3D = np.reshape(fea3D, (16))
                # fea3D = fea3D.tolist()

                bbvis.append(box3D)

                nx1 = obj_bb[0]
                ny1 = obj_bb[1]
                nx2 = nx1 + obj_bb[2]
                ny2 = ny1 + obj_bb[3]
                npseg = np.array([nx1, ny1, nx2, ny1, nx2, ny2, nx1, ny2])
                cont = npseg.tolist()

                annoID = annoID + 1
                tempTA = {
                    "id": annoID,
                    "image_id": img_id,
                    "category_id": obj_id,
                    "bbox": obj_bb,
                    "pose": pose,
                    "segmentation": box3D,
                    "mask_id": mask_id,
                    "area": area,
                    "iscrowd": 0,
                    "feature_visibility": visib_fract
                }

                dict["annotations"].append(tempTA)
                count = count + 1

            tempTL = {
                "url": "cmp.felk.cvut.cz/t-less/",
                "id": img_id,
                "name": iname,
            }
            dict["licenses"].append(tempTL)

            # mask_img = cv2.resize(mask_img, None, fx=1 / 4, fy=1 / 4, interpolation=cv2.INTER_NEAREST)
            mask_safe_path = fileName[:-8] + '_mask.png'
            cv2.imwrite(mask_safe_path, mask_img)

            tempTV = {
                "license": 2,
                "url": "cmp.felk.cvut.cz/t-less/",
                "file_name": iname,
                "height": resY,
                "width": resX,
                "fx": fxca,
                "fy": fyca,
                "cx": cxca,
                "cy": cyca,
                "date_captured": dateT,
                "id": img_id,
            }
            dict["images"].append(tempTV)

            if visu is True:
                img = rgbImg
                for i, bb in enumerate(bbvis):

                    bb = np.array(bb)

                    phler = True
                    if phler:
                        pose = np.asarray(bbvis[i], dtype=np.float32)

                        colR = 250
                        colG = 25
                        colB = 175

                        img = cv2.line(img, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), (130, 245, 13), 2)
                        img = cv2.line(img, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), (50, 112, 220), 2)
                        img = cv2.line(img, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), (50, 112, 220), 2)
                        img = cv2.line(img, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), (50, 112, 220), 2)
                        img = cv2.line(img, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), (colR, colG, colB),
                                       2)
                        img = cv2.line(img, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()),
                                       (colR, colG, colB), 2)
                        img = cv2.line(img, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()),
                                       (colR, colG, colB), 2)
                        img = cv2.line(img, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()),
                                       (colR, colG, colB), 2)
                        img = cv2.line(img, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()),
                                       (colR, colG, colB), 2)
                        img = cv2.line(img, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()),
                                       (colR, colG, colB), 2)
                        img = cv2.line(img, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()),
                                       (colR, colG, colB), 2)
                        img = cv2.line(img, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()),
                                       (colR, colG, colB), 2)

                cv2.imwrite(rgb_name, img)

                print('STOP')

    catsInt = range(1, 22)

    for s in catsInt:
        objName = str(s)
        tempC = {
            "id": s,
            "name": objName,
            "supercategory": "object"
        }
        dict["categories"].append(tempC)

    valAnno = target + 'annotations/instances_train.json'

    with open(valAnno, 'w') as fpT:
        json.dump(dict, fpT)

    print('everythings done')


