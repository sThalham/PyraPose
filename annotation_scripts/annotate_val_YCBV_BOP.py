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

depthCut = 2000.0


def draw_axis(img, cam_R, cam_T, fxkin=None, fykin=None, cxkin=None, cykin=None):
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


def encodeImage(depth):
    img = np.zeros((resY, resX, 3), dtype=np.uint8)

    normImg, depImg = get_normal(depth, fxkin, fykin, cxkin, cykin, for_vis=True)
    img[:, :, 0] = compute_disparity(depImg)
    img[:, :, 1] = encode_area(depImg)
    img[:, :, 2] = compute_angle2gravity(normImg, depImg)

    return img


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
    print(scaleOri)

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
    root = "/home/stefan/data/datasets/YCBV_BOP_val/"  # path to train samples, depth + rgb
    target = '/home/stefan/data/train_data/val_YCBV_BOP_RGBD/'
    # print(root)
    visu = False

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

    dictVal = copy.deepcopy(dict)

    annoID = 0

    gloCo = 0
    allCo = 18723
    if dataset == 'tless':
        allCo = 10080
    times = []

    min_obj_dist = 1000.0
    max_obj_dist = 0.0

    for s in sub:

        rgbPath = root + s + "/rgb/"
        depPath = root + s + "/depth/"
        camPath = root + s + "/scene_camera.json"
        gtPath = root + s + "/scene_gt.json"
        infoPath = root + s + "/scene_gt_info.json"

        with open(camPath, 'r') as streamCAM:
            camjson = json.load(streamCAM)

        with open(gtPath, 'r') as streamGT:
            scenejson = json.load(streamGT)

        with open(infoPath, 'r') as streamINFO:
            gtjson = json.load(streamINFO)

        subsub = os.listdir(rgbPath)

        counter = 0
        for ss in subsub:

            start_time = time.time()
            gloCo = gloCo + 1

            imgname = ss
            rgbImgPath = rgbPath + ss
            depImgPath = depPath + ss
            #print(rgbImgPath)

            if ss.startswith('00000'):
                ss = ss[5:]
            elif ss.startswith('0000'):
                ss = ss[4:]
            elif ss.startswith('000'):
                ss = ss[3:]
            elif ss.startswith('00'):
                ss = ss[2:]
            elif ss.startswith('0'):
                ss = ss[1:]
            ss = ss[:-4]

            calib = camjson.get(str(ss))
            K = calib["cam_K"]
            depSca = calib["depth_scale"]
            fxca = K[0]
            fyca = K[4]
            cxca = K[2]
            cyca = K[5]
            #cam_R = calib["cam_R_w2c"]
            #cam_T = calib["cam_t_w2c"]

            #########################
            # Prepare the stuff
            #########################

            # read images and create mask
            rgbImg = cv2.imread(rgbImgPath)
            depImg = cv2.imread(depImgPath, cv2.IMREAD_UNCHANGED)
            #depImg = cv2.resize(depImg, None, fx=1 / 2, fy=1 / 2)
            rows, cols = depImg.shape
            depImg = np.multiply(depImg, 0.1)
            print(np.amax(depImg))

            # create image number and name
            template = '00000'
            s = int(s)
            ssm = int(ss) + 1
            pre = (s-1) * 1296
            img_id = pre + ssm
            tempSS = template[:-len(str(img_id))]

            imgNum = str(img_id)
            imgNam = tempSS + imgNum + '.jpg'
            iname = str(imgNam)

            gtImg = gtjson.get(str(ss))

            bbox_vis = []
            cat_vis = []
            camR_vis = []
            camT_vis = []
            #if rnd == 1:
            rnd = True
            #print(rnd)
            if rnd == True:

                fileName = target + 'images/val/' + imgNam[:-4] + '_dep.jpg'

                myFile = Path(fileName)
                if myFile.exists():
                    print('File exists, skip encoding, ', fileName)
                else:
                    # imgI = encodeImage(depImg)
                    #imgI, depth_refine, depth_inp = get_normal(depImg, fx=fxca, fy=fyca, cx=cxca, cy=cyca, for_vis=False)
                    #depName = target + 'images/val/' + tempSS + imgNum + '_dep.png'
                    #copyfile(depImgPath, depName)

                    depImg[depImg > depthCut] = 0
                    scaCro = 255/depthCut
                    cross = np.multiply(depImg, scaCro)
                    dep_sca = cross.astype(np.uint8)
                    imgI = np.repeat(dep_sca[:, :, np.newaxis], 3, 2)

                    rgb_name = fileName[:-8] + '_rgb.jpg'
                    cv2.imwrite(rgb_name, rgbImg)
                    cv2.imwrite(fileName, imgI)
                    dep_raw_name = fileName[:-8] + '_dep_raw.png'
                    copyfile(depImgPath, dep_raw_name)
                    print("storing image in : ", fileName)

                bbvis = []
                bb3vis = []
                cats = []
                posvis = []
                postra = []
                #bbsca = 720.0 / 640.0
                for i in range(len(gtImg)):

                    curlist = gtImg[i]
                    #if obj_id > 6:
                    #    obj_id = obj_id - 2
                    #elif obj_id > 2:
                    #    obj_id = obj_id - 1
                    obj_bb = curlist["bbox_obj"]
                    bbvis.append(obj_bb)

                    gtPose = scenejson.get(str(ss))
                    obj_id = gtPose[i]['obj_id']
                    R = gtPose[i]["cam_R_m2c"]
                    T = gtPose[i]["cam_t_m2c"]
                    #cats.append(objID)

                    # pose [x, y, z, roll, pitch, yaw]
                    R = np.asarray(R, dtype=np.float32)
                    rot = tf3d.euler.mat2euler(R.reshape(3, 3))
                    rot = np.asarray(rot, dtype=np.float32)
                    tra = np.asarray(T, dtype=np.float32)
                    pose = [np.asscalar(tra[0]), np.asscalar(tra[1]), np.asscalar(tra[2]),
                            np.asscalar(rot[0]), np.asscalar(rot[1]), np.asscalar(rot[2])]
                    camR_vis.append([np.asscalar(rot[0]), np.asscalar(rot[1]), np.asscalar(rot[2])])
                    #camT_vis.append(tra)

                    if tra[2] > max_obj_dist:
                        max_obj_dist = tra[2]
                    if tra[2] < min_obj_dist:
                        min_obj_dist = tra[2]

                    area = obj_bb[2] * obj_bb[3]

                    # placeholder
                    box3D = np.zeros((16), dtype=np.float32).tolist()

                    nx1 = obj_bb[0]
                    ny1 = obj_bb[1]
                    nx2 = nx1 + obj_bb[2]
                    ny2 = ny1 + obj_bb[3]
                    npseg = np.array([nx1, ny1, nx2, ny1, nx2, ny2, nx1, ny2])
                    cont = npseg.tolist()

                    annoID = annoID + 1
                    tempVa = {
                        "id": annoID,
                        "image_id": img_id,
                        "category_id": obj_id,
                        "bbox": obj_bb,
                        "pose": pose,
                        "segmentation": box3D,
                        "area": area,
                        "iscrowd": 0
                    }
                    dictVal["annotations"].append(tempVa)

                # create dictionaries for json
                tempVl = {
                    "url": "cmp.felk.cvut.cz/t-less/",
                    "id": img_id,
                    "name": iname
                }
                dictVal["licenses"].append(tempVl)

                tempVi = {
                    "license": 2,
                    "url": "cmp.felk.cvut.cz/t-less/",
                    "file_name": iname,
                    "height": rows,
                    "width": cols,
                    "date_captured": dateT,
                    "id": img_id
                }
                dictVal["images"].append(tempVi)

            else:
                continue

            elapsed_time = time.time() - start_time
            times.append(elapsed_time)
            meantime = sum(times) / len(times)
            eta = ((allCo - gloCo) * meantime) / 60
            if gloCo % 100 == 0:
                print('eta: ', eta, ' min')
                times = []

            if visu is True:

                img = rgbImg
                for i, bb in enumerate(bbvis):

                    # if cats[i] not in [19, 20, 23]:
                    #    continue

                    bb = np.array(bb)

                    cv2.rectangle(img, (int(bb[1]), int(bb[0])), (int(bb[3]), int(bb[2])),
                                  (255, 255, 255), 2)
                    cv2.rectangle(img, (int(bb[1]), int(bb[0])), (int(bb[3]), int(bb[2])),
                                  (0, 0, 0), 1)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (int(bb[1]), int(bb[0]))
                    fontScale = 1
                    fontColor = (0, 0, 0)
                    fontthickness = 1
                    lineType = 2
                    gtText = str(cats[i])
                    # print(cats[i])

                    fontColor2 = (255, 255, 255)
                    fontthickness2 = 3
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

                    # print(posvis[i])
                    if i is not poses.shape[0]:
                        pose = np.asarray(bb3vis[i], dtype=np.float32)

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

                cv2.imwrite(fileName, img)

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
        dictVal["categories"].append(tempC)

    valAnno = target + 'annotations/instances_val.json'
    trainAnno = target + 'annotations/instances_test.json'

    with open(valAnno, 'w') as fpV:
        json.dump(dictVal, fpV)

    with open(trainAnno, 'w') as fpT:
        json.dump(dict, fpT)

    print('min_obj_dist: ', min_obj_dist)
    print('max_obj_dist: ', max_obj_dist)

    print('everythings done')


