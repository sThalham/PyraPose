import os
import yaml
import cv2
import numpy as np
import datetime
import copy
import transforms3d as tf3d
import time
import random
import json
import math

import OpenEXR, Imath
from pathlib import Path

from misc import manipulate_RGB, toPix_array, toPix
from Augmentations import augmentDepth, maskDepth, augmentRGB, augmentAAEext, augmentRGB_V2, augmentRGB_V3, get_normal


def get_cont_sympose(rot_pose, sym):

    cam_in_obj = np.dot(np.linalg.inv(rot_pose), (0, 0, 0, 1))
    alpha = math.atan2(cam_in_obj[1], cam_in_obj[0])
    rot_pose[:3, :3] = np.dot(rot_pose[:3, :3], tf3d.euler.euler2mat(0.0, 0.0, alpha, 'sxyz'))

    return rot_pose


def get_disc_sympose(rot_pose, sym, oid):

    if len(sym) > 3:
        sym = np.array(sym, dtype=np.float32)
        if sym[0, 0] == 1:
            c_alpha = np.dot([0, 1, 0], np.dot(rot_pose[0:3, 0:3], [0, 1, 0]))
            if c_alpha < 0:
                rot_pose_new = np.dot(rot_pose, sym)
            else:
                rot_pose_new = rot_pose
        if sym[1, 1] == 1:
            c_alpha = np.dot([1, 0, 0], np.dot(rot_pose[0:3, 0:3], [1, 0, 0]))
            if c_alpha < 0:
                rot_pose_new = np.dot(rot_pose, sym)
            else:
                rot_pose_new = rot_pose
        if sym[2, 2] == 1:
            c_alpha = np.dot([1, 0, 0], np.dot(rot_pose[0:3, 0:3], [1, 0, 0]))
            if c_alpha < 0:
                rot_pose_new = np.dot(rot_pose, sym)
            else:
                rot_pose_new = rot_pose
        else:
            rot_pose_new = rot_pose

    else:
        rot_pose1 = np.dot(rot_pose, sym[0])
        rot_pose2 = np.dot(rot_pose, sym[1])
        rot_pose3 = np.dot(rot_pose, sym[2])
        alpha_0 = np.dot([1, 0, 0], np.dot(rot_pose[0:3, 0:3], [1, 0, 0]))
        alpha_1 = np.dot([1, 0, 0], np.dot(rot_pose1[0:3, 0:3], [1, 0, 0]))
        alpha_2 = np.dot([1, 0, 0], np.dot(rot_pose2[0:3, 0:3], [1, 0, 0]))
        alpha_3 = np.dot([1, 0, 0], np.dot(rot_pose3[0:3, 0:3], [1, 0, 0]))
        if alpha_1 < alpha_0 and alpha_1 < alpha_2 and alpha_1 < alpha_3:
            rot_pose_new = rot_pose1
        elif alpha_2 < alpha_0 and alpha_2 < alpha_1 and alpha_2 < alpha_3:
            rot_pose_new = rot_pose2
        elif alpha_3 < alpha_0 and alpha_3 < alpha_1 and alpha_3 < alpha_1:
            rot_pose_new = rot_pose3
        else:
            rot_pose_new = rot_pose

    return rot_pose_new


if __name__ == "__main__":

<<<<<<< HEAD
    root = '/home/stefan/data/rendered_data/ycbv_rgbd/patches'
    root2 = '/home/stefan/data/rendered_data/ycbv_rgbd_2/patches'
    target = '/home/stefan/data/train_data/ycbv_RGBD_V2/'
    mesh_info = '/home/stefan/data/Meshes/ycb_video_st/models/models_info.json'
=======
    root = '/home/sthalham/ycb_test/patches'
    target = '/home/sthalham/data/prepro/ycbv_RGBD/'
    mesh_info = '/home/sthalham/data/Meshes/ycbv_st/models/models_info.json'
>>>>>>> 177484e6aa32844a6e9ebe9a55dc81406dd72afc

    visu = False
    resX = 640
    resY = 480
    fxkin = 579.68  # blender calculated
    fykin = 542.31  # blender calculated
    cxkin = 320
    cykin = 240
    depthCut = 2000

    threeD_boxes = np.ndarray((22, 8, 3), dtype=np.float32)
    sym_cont = np.ndarray((22, 3), dtype=np.float32)
    sym_disc = np.ndarray((28, 4, 4), dtype=np.float32)

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

        if "symmetries_continuous" in value:
            sym_cont[int(key), :] = np.asarray(value['symmetries_continuous'][0]['axis'], dtype=np.float32)
        elif "symmetries_discrete" in value:
            syms = value['symmetries_discrete']
            #Obj 16
            if len(syms) > 1:
                sym_disc[int(key), :, :] = np.asarray(syms[0], dtype=np.float32).reshape((4, 4))
                sym_disc[22, :, :] = np.asarray(syms[1], dtype=np.float32).reshape((4, 4))
                sym_disc[23, :, :] = np.asarray(syms[2], dtype=np.float32).reshape((4, 4))
                sym_disc[24, :, :] = np.asarray(syms[3], dtype=np.float32).reshape((4, 4))
                sym_disc[25, :, :] = np.asarray(syms[4], dtype=np.float32).reshape((4, 4))
                sym_disc[26, :, :] = np.asarray(syms[5], dtype=np.float32).reshape((4, 4))
                sym_disc[27, :, :] = np.asarray(syms[6], dtype=np.float32).reshape((4, 4))
            else:
                sym_disc[int(key), :, :] = np.asarray(syms[0], dtype=np.float32).reshape((4,4))
        else:
            pass

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
    times = []

    trainN = 1
    testN = 1
    valN = 1

    depPath = root + "/depth/"
    partPath = root + "/part/"
    gtPath = root
    maskPath = root + "/mask/"
    rgbPath = root + "/rgb/"
    excludedImgs = []
    boxWidths = []
    boxHeights = []
    meanRGBD = np.zeros((6), np.float64)

    syns = os.listdir(root)
    syns2 = os.listdir(root2)
    all = len(syns) + len(syns2)

    for fileInd in syns:
        if fileInd.endswith(".yaml"):

            start_time = time.time()
            gloCo = gloCo + 1

            redname = fileInd[:-8]

            gtfile = gtPath + '/' + fileInd
            depfile = depPath + redname + "_depth.exr"
            partfile = partPath + redname + "_part.png"
            maskfile = maskPath + redname + "_mask.npy"
            rgbfile = rgbPath + redname + "_rgb.png"

            depth_refine, rgb_refine, mask, bboxes, poses, mask_ids, visibilities = manipulate_RGB(gtfile, depfile, partfile, rgbfile)
            try:
                obj_mask = np.load(maskfile)
            except Exception:
                continue
            obj_mask = obj_mask.astype(np.int8)

            if bboxes is None:
                excludedImgs.append(int(redname))
                continue

            depth_refine = np.multiply(depth_refine, 1000.0)  # to millimeters
            rows, cols = depth_refine.shape

            for k in range(0, 1):

                newredname = redname[1:] + str(k)

                fileName = target + "images/train/" + newredname + '_rgb.jpg'
                myFile = Path(fileName)
                print(myFile)

                if myFile.exists():
                    print('File exists, skip encoding and safing.')

                else:

                    depthAug = maskDepth(depth_refine, obj_mask, mask)
                    rgbAug = rgb_refine

                    depthAug[depthAug > depthCut] = 0
                    aug_dep = depthAug.astype(np.uint16)

                    meanRGBD[0] += np.nanmean(rgbAug[:, :, 0])
                    meanRGBD[1] += np.nanmean(rgbAug[:, :, 1])
                    meanRGBD[2] += np.nanmean(rgbAug[:, :, 2])
                    meanRGBD[3] += np.nanmean(aug_dep[:, :])
                    meanRGBD[4] += np.nanmean(aug_dep[:, :])
                    meanRGBD[5] += np.nanmean(aug_dep[:, :])

                    cv2.imwrite(fileName, rgbAug)
                    cv2.imwrite(fileName[:-8] + '_dep.png', aug_dep)

                imgID = int(newredname)
                imgName = newredname + '.jpg'
                # print(imgName)

                # bb scaling because of image scaling
                bbvis = []
                bb3vis = []
                cats = []
                posvis = []
                postra = []
                # for i, bbox in enumerate(bboxes[:-1]):
                for i, bbox in enumerate(bboxes[:-1]):

                    if visibilities[i] < 0.5:
                        # print('visivility: ', visibilities[i], ' skip!')
                        continue
                    #print(visibilities[i])
                    #if (np.asscalar(bbox[0]) + 1) > 13:
                    #    continue

                    bbvis.append(bbox.astype(int))
                    objID = np.asscalar(bbox[0]) + 1
                    #objID = np.asscalar(bboxes[i+1][0]) + 1
                    cats.append(objID)

                    bbox = (bbox).astype(int)

                    #rot = tf3d.quaternions.quat2mat(poses[i, 3:])
                    #rot = np.asarray(rot, dtype=np.float32)

                    rot = tf3d.quaternions.quat2mat(poses[i, 3:])
                    tra = poses[i, 0:3]
                    pose = np.zeros((4, 4), dtype=np.float32)
                    pose[:3, :3] = rot
                    pose[:3, 3] = tra
                    pose[3, 3] = 1

                    if objID in [13, 18]:
                        rot = get_cont_sympose(pose, sym_cont[objID, :])

                    elif objID in [1, 19, 20, 21]:
                        rot = get_disc_sympose(pose, sym_disc[objID, :, :], objID)

                    #elif objID == 16:
                    #    rot = get_disc_sympose(pose, [sym_disc[16, :, :], sym_disc[22, :, :], sym_disc[23, :, :], sym_disc[24, :, :], sym_disc[25, :, :], sym_disc[26, :, :], sym_disc[27, :, :]],
                    #                           objID)

                    rot = np.asarray(rot, dtype=np.float32)

<<<<<<< HEAD
                    tDbox = rot[:3, :3].dot(threeD_boxes[objID, :, :].T).T
                    tDbox = tDbox + np.repeat(poses[i, np.newaxis, 0:3], 8, axis=0)

                    # if objID == 10 or objID == 11:
                    #    print(tf3d.euler.quat2euler(poses[i, 3:]))

                    box3D = toPix_array(tDbox, fx=fxkin, fy=fykin, cx=cxkin, cy=cykin)
                    box3D = np.reshape(box3D, (16))
                    box3D = box3D.tolist()
                    bb3vis.append(box3D)

                    bbox = bbox.astype(int)
                    x1 = np.asscalar(bbox[2])
                    y1 = np.asscalar(bbox[1])
                    x2 = np.asscalar(bbox[4])
                    y2 = np.asscalar(bbox[3])
                    nx1 = bbox[2]
                    ny1 = bbox[1]
                    nx2 = bbox[4]
                    ny2 = bbox[3]
                    w = (x2 - x1)
                    h = (y2 - y1)
                    boxWidths.append(w)
                    boxHeights.append(h)
                    bb = [x1, y1, w, h]
                    area = w * h
                    npseg = np.array([nx1, ny1, nx2, ny1, nx2, ny2, nx1, ny2])
                    seg = npseg.tolist()

                    pose = [np.asscalar(poses[i, 0]), np.asscalar(poses[i, 1]), np.asscalar(poses[i, 2]),
                            np.asscalar(poses[i, 3]), np.asscalar(poses[i, 4]), np.asscalar(poses[i, 5]),
                            np.asscalar(poses[i, 6])]
                    if i != len(bboxes):
                        pose[0:2] = toPix(pose[0:3], fx=fxkin, fy=fykin, cx=cxkin, cy=cykin)

                    posvis.append(pose)
                    tra = np.asarray(poses[i, :3], dtype=np.float32)
                    postra.append(tra)

                    annoID = annoID + 1
                    tempTA = {
                        "id": annoID,
                        "image_id": imgID,
                        "category_id": objID,
                        "bbox": bb,
                        "pose": pose,
                        "segmentation": box3D,
                        "area": area,
                        "iscrowd": 0,
                        # "feature_visibility": feat_vis
                    }
                    # print('norm q: ', np.linalg.norm(pose[3:]))

                    dict["annotations"].append(tempTA)

                tempTL = {
                    "url": "cmp.felk.cvut.cz/t-less/",
                    "id": imgID,
                    "name": imgName
                }
                dict["licenses"].append(tempTL)

                tempTV = {
                    "license": 2,
                    "url": "cmp.felk.cvut.cz/t-less/",
                    "file_name": imgName,
                    "height": resY,
                    "width": resX,
                    "date_captured": dateT,
                    "id": imgID
                }
                dict["images"].append(tempTV)

                gloCo += 1

                elapsed_time = time.time() - start_time
                times.append(elapsed_time)
                meantime = sum(times) / len(times)
                eta = ((all - gloCo) * meantime) / 60
                if gloCo % 100 == 0:
                    print('eta: ', eta, ' min')
                    times = []

                if visu is True:
                    img = rgbAug
                    for i, bb in enumerate(bbvis):

                        # if cats[i] not in [19, 20, 23]:
                        #    continue

                        bb = np.array(bb)

                        cv2.rectangle(img, (int(bb[2]), int(bb[1])), (int(bb[4]), int(bb[3])),
                                      (255, 255, 255), 2)
                        cv2.rectangle(img, (int(bb[2]), int(bb[1])), (int(bb[4]), int(bb[3])),
                                      (0, 0, 0), 1)

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        bottomLeftCornerOfText = (int(bb[2]), int(bb[1]))
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
                            print(pose)

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

    for fileInd in syns2:
        if fileInd.endswith(".yaml"):

            start_time = time.time()
            gloCo = gloCo + 1

            redname = fileInd[:-8]

            gtfile = gtPath + '/' + fileInd
            depfile = depPath + redname + "_depth.exr"
            partfile = partPath + redname + "_part.png"
            maskfile = maskPath + redname + "_mask.npy"
            rgbfile = rgbPath + redname + "_rgb.png"

            depth_refine, rgb_refine, mask, bboxes, poses, mask_ids, visibilities = manipulate_RGB(gtfile, depfile, partfile, rgbfile)
            try:
                obj_mask = np.load(maskfile)
            except Exception:
                continue
            obj_mask = obj_mask.astype(np.int8)

            if bboxes is None:
                excludedImgs.append(int(redname))
                continue

            depth_refine = np.multiply(depth_refine, 1000.0)  # to millimeters
            rows, cols = depth_refine.shape

            for k in range(0, 1):

                newredname = str(2) + redname[1:] + str(k)

                fileName = target + "images/train/" + newredname + '_rgb.jpg'
                myFile = Path(fileName)
                print(myFile)

                if myFile.exists():
                    print('File exists, skip encoding and safing.')

                else:
                    depthAug = maskDepth(depth_refine, obj_mask, mask)
                    rgbAug = rgb_refine

                    depthAug[depthAug > depthCut] = 0
                    aug_dep = depthAug.astype(np.uint16)

                    meanRGBD[0] += np.nanmean(rgbAug[:, :, 0])
                    meanRGBD[1] += np.nanmean(rgbAug[:, :, 1])
                    meanRGBD[2] += np.nanmean(rgbAug[:, :, 2])
                    meanRGBD[3] += np.nanmean(aug_dep[:, :])
                    meanRGBD[4] += np.nanmean(aug_dep[:, :])
                    meanRGBD[5] += np.nanmean(aug_dep[:, :])

                    cv2.imwrite(fileName, rgbAug)
                    cv2.imwrite(fileName[:-8] + '_dep.png', aug_dep)

                imgID = int(newredname)
                imgName = newredname + '.jpg'
                # print(imgName)

                # bb scaling because of image scaling
                bbvis = []
                bb3vis = []
                cats = []
                posvis = []
                postra = []
                # for i, bbox in enumerate(bboxes[:-1]):
                for i, bbox in enumerate(bboxes[:-1]):

                    if visibilities[i] < 0.5:
                        # print('visivility: ', visibilities[i], ' skip!')
                        continue
                    #print(visibilities[i])
                    #if (np.asscalar(bbox[0]) + 1) > 13:
                    #    continue

                    bbvis.append(bbox.astype(int))
                    objID = np.asscalar(bbox[0]) + 1
                    #objID = np.asscalar(bboxes[i+1][0]) + 1
                    cats.append(objID)

                    bbox = (bbox).astype(int)

                    #rot = tf3d.quaternions.quat2mat(poses[i, 3:])
                    #rot = np.asarray(rot, dtype=np.float32)

                    rot = tf3d.quaternions.quat2mat(poses[i, 3:])
                    tra = poses[i, 0:3]
                    pose = np.zeros((4, 4), dtype=np.float32)
                    pose[:3, :3] = rot
                    pose[:3, 3] = tra
                    pose[3, 3] = 1

                    if objID in [13, 18]:
                        rot = get_cont_sympose(pose, sym_cont[objID, :])

                    elif objID in [1, 19, 20, 21]:
                        rot = get_disc_sympose(pose, sym_disc[objID, :, :], objID)

                    # elif objID == 16:
                    #    rot = get_disc_sympose(pose, [sym_disc[16, :, :], sym_disc[22, :, :], sym_disc[23, :, :], sym_disc[24, :, :], sym_disc[25, :, :], sym_disc[26, :, :], sym_disc[27, :, :]],
                    #                           objID)

                    rot = np.asarray(rot, dtype=np.float32)
=======
                    cls = objID
>>>>>>> 177484e6aa32844a6e9ebe9a55dc81406dd72afc

                    tDbox = rot[:3, :3].dot(threeD_boxes[objID, :, :].T).T
                    tDbox = tDbox + np.repeat(poses[i, np.newaxis, 0:3], 8, axis=0)

                    # if objID == 10 or objID == 11:
                    #    print(tf3d.euler.quat2euler(poses[i, 3:]))

                    box3D = toPix_array(tDbox, fx=fxkin, fy=fykin, cx=cxkin, cy=cykin)
                    box3D = np.reshape(box3D, (16))
                    box3D = box3D.tolist()
                    bb3vis.append(box3D)

                    bbox = bbox.astype(int)
                    x1 = np.asscalar(bbox[2])
                    y1 = np.asscalar(bbox[1])
                    x2 = np.asscalar(bbox[4])
                    y2 = np.asscalar(bbox[3])
                    nx1 = bbox[2]
                    ny1 = bbox[1]
                    nx2 = bbox[4]
                    ny2 = bbox[3]
                    w = (x2 - x1)
                    h = (y2 - y1)
                    boxWidths.append(w)
                    boxHeights.append(h)
                    bb = [x1, y1, w, h]
                    area = w * h
                    npseg = np.array([nx1, ny1, nx2, ny1, nx2, ny2, nx1, ny2])
                    seg = npseg.tolist()

                    pose = [np.asscalar(poses[i, 0]), np.asscalar(poses[i, 1]), np.asscalar(poses[i, 2]),
                            np.asscalar(poses[i, 3]), np.asscalar(poses[i, 4]), np.asscalar(poses[i, 5]),
                            np.asscalar(poses[i, 6])]
                    if i != len(bboxes):
                        pose[0:2] = toPix(pose[0:3], fx=fxkin, fy=fykin, cx=cxkin, cy=cykin)

                    posvis.append(pose)
                    tra = np.asarray(poses[i, :3], dtype=np.float32)
                    postra.append(tra)

                    annoID = annoID + 1
                    tempTA = {
                        "id": annoID,
                        "image_id": imgID,
                        "category_id": objID,
                        "bbox": bb,
                        "pose": pose,
                        "segmentation": box3D,
                        "area": area,
                        "iscrowd": 0,
                        # "feature_visibility": feat_vis
                    }
                    # print('norm q: ', np.linalg.norm(pose[3:]))

                    dict["annotations"].append(tempTA)

                tempTL = {
                    "url": "cmp.felk.cvut.cz/t-less/",
                    "id": imgID,
                    "name": imgName
                }
                dict["licenses"].append(tempTL)

                tempTV = {
                    "license": 2,
                    "url": "cmp.felk.cvut.cz/t-less/",
                    "file_name": imgName,
                    "height": resY,
                    "width": resX,
                    "date_captured": dateT,
                    "id": imgID
                }
                dict["images"].append(tempTV)

                gloCo += 1

                elapsed_time = time.time() - start_time
                times.append(elapsed_time)
                meantime = sum(times) / len(times)
                eta = ((all - gloCo) * meantime) / 60
                if gloCo % 100 == 0:
                    print('eta: ', eta, ' min')
                    times = []

                if visu is True:
                    img = rgbAug
                    for i, bb in enumerate(bbvis):

                        # if cats[i] not in [19, 20, 23]:
                        #    continue

                        bb = np.array(bb)

                        cv2.rectangle(img, (int(bb[2]), int(bb[1])), (int(bb[4]), int(bb[3])),
                                      (255, 255, 255), 2)
                        cv2.rectangle(img, (int(bb[2]), int(bb[1])), (int(bb[4]), int(bb[3])),
                                      (0, 0, 0), 1)

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        bottomLeftCornerOfText = (int(bb[2]), int(bb[1]))
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
                            print(pose)

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

    traAnno = target + "annotations/instances_train.json"

    with open(traAnno, 'w') as fpT:
        json.dump(dict, fpT)

    excludedImgs.sort()
    print('excluded images: ')
    for ex in excludedImgs:
        print(ex)

    all_rendered = len(os.listdir(target + "images/train/")) * 0.5
    means = meanRGBD / all_rendered
    print('means: ', means)

    print('Chill for once in your life... everything\'s done')
