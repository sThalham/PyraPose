import os
import sys
import yaml
import cv2
import numpy as np
import datetime
import copy
import transforms3d as tf3d
import time
import random
import json

import OpenEXR, Imath
from pathlib import Path

from misc import manipulate_RGB, toPix_array, toPix, calculate_feature_visibility
from Augmentations import augmentDepth, augmentRGB, augmentAAEext, augmentRGB_V2, get_normal

# Import bop_renderer and bop_toolkit.
# ------------------------------------------------------------------------------
# Import bop_renderer and bop_toolkit.
# ------------------------------------------------------------------------------
bop_renderer_path = '/home/stefan/bop_renderer/build'
sys.path.append(bop_renderer_path)
import bop_renderer


if __name__ == "__main__":

    mesh_path = sys.argv[1]
    background = '/home/stefan/data/dataset/cocoval2017/'
    set_path = '/home/stefan/data/train_data/fronius_train/'

    visu = False
    resX = 640
    resY = 480
    fx = 572.41140
    fy = 573.57043
    cx = 325.26110
    cy = 242.04899

    ren = bop_renderer.Renderer()
    ren.init(resX, resY)
    mesh_id = 0
    for mesh_now in os.listdir(mesh_path):
        print(mesh_now)
        ren.add_object(mesh_id, mesh_now)
        mesh_id += 1

    threeD_boxes = np.ndarray((4, 8, 3), dtype=np.float32)
    threeD_boxes[2, :, :] = np.array([[0.060, 0.1, 0.03],  # Seite-AC [120, 198, 45] links
                                      [0.060, 0.1, -0.03],
                                      [0.060, -0.1, -0.03],
                                      [0.060, -0.1, 0.03],
                                      [-0.060, 0.1, 0.03],
                                      [-0.060, 0.1, -0.03],
                                      [-0.060, -0.1, -0.03],
                                      [-0.060, -0.1, 0.03]])
    threeD_boxes[0, :, :] = np.array([[0.05, 0.04, 0.03],  # AC-Abdeckung [81, 68, 25] expand last dim
                                      [0.05, 0.04, -0.03],
                                      [0.05, -0.04, -0.03],
                                      [0.05, -0.04, 0.03],
                                      [-0.05, 0.04, 0.03],
                                      [-0.05, 0.04, -0.03],
                                      [-0.05, -0.04, -0.03],
                                      [-0.05, -0.04, 0.03]])
    threeD_boxes[1, :, :] = np.array([[0.05, 0.04, 0.03],  # DC [81, 72, 38]
                                      [0.05, 0.04, -0.03],
                                      [0.05, -0.04, -0.03],
                                      [0.05, -0.04, 0.03],
                                      [-0.05, 0.04, 0.03],
                                      [-0.05, 0.04, -0.03],
                                      [-0.05, -0.04, -0.03],
                                      [-0.05, -0.04, 0.03]])
    threeD_boxes[3, :, :] = np.array([[0.060, 0.1, 0.03],  # Seite-DC [120, 206, 56] rechts
                                      [0.060, 0.1, -0.03],
                                      [0.060, -0.1, -0.03],
                                      [0.060, -0.1, 0.03],
                                      [-0.060, 0.1, 0.03],
                                      [-0.060, 0.1, -0.03],
                                      [-0.060, -0.1, -0.03],
                                      [-0.060, -0.1, 0.03]])


    dict = {"info": {
        "description": "tless",
        "url": "cmp.felk.cvut.cz/t-less/",
        "version": "1.0",
        "year": 2020,
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

    syns = os.listdir(background)
    for o_idx in range(10):
        for bg_img_path in syns:

            bg_img = cv2.imread(bg_img_path)
            bg_x, bg_y, _ = bg_img.shape()

            if bg_y > bg_x:
                bg_img = np.swapaxes(bg_img, 0, 1)

            bg_img = cv2.resize(bg_img, ())

            for objID in threeD_boxes.shape[0]:
                R = np.eye(3)
                t = np.array([[0.0, 0.0, 150.0]]).T

                R_list = R.flatten().tolist()
                t_list = t.flatten().tolist()
                ren.render_object(obj_id, R_list, t_list, fx, fy, cx, cy)
                rgb = ren.get_color_image(obj_id)

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
                    depthAug = augmentDepth(depth_refine, obj_mask, mask)
                    rgbAug = augmentRGB(rgb_refine)
                    #rgbAug = augmentAAEext(rgb_refine)

                    #aug_xyz, depth_refine_aug, depth_imp = get_normal(depthAug, fx=fxkin, fy=fykin, cx=cxkin, cy=cykin,
                    #                                                  for_vis=False)

                    depthAug[depthAug > depthCut] = 0
                    scaCro = 255.0 / np.nanmax(depthAug)
                    cross = np.multiply(depthAug, scaCro)
                    aug_dep = cross.astype(np.uint8)
                    #aug_dep = np.repeat(aug_dep[:, :, np.newaxis], 3, 2)

                    meanRGBD[0] += np.nanmean(rgbAug[:, :, 0])
                    meanRGBD[1] += np.nanmean(rgbAug[:, :, 1])
                    meanRGBD[2] += np.nanmean(rgbAug[:, :, 2])
                    meanRGBD[3] += np.nanmean(aug_dep[:, :])
                    meanRGBD[4] += np.nanmean(aug_dep[:, :])
                    meanRGBD[5] += np.nanmean(aug_dep[:, :])
                    # meanRGBD[3] = np.nanmean(depthAug[:, :, 0])
                    # meanRGBD[4] = np.nanmean(depthAug[:, :, 1])
                    # meanRGBD[5] = np.nanmean(depthAug[:, :, 2])

                    cv2.imwrite(fileName, rgbAug)
                    cv2.imwrite(fileName[:-8] + '_dep.jpg', aug_dep)
                    #img_rgbd = np.concatenate((rgbAug, aug_dep[:, :, np.newaxis]), axis=2)
                    #cv2.imwrite(fileName, img_rgbd)
                    #np.save(fileName, img_rgbd)

                imgID = int(newredname)
                imgName = newredname + '.jpg'
                # print(imgName)

                # bb scaling because of image scaling
                bbvis = []
                bb3vis = []
                cats = []
                posvis = []
                postra = []
                feat_visualization = []
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

                    rot = tf3d.quaternions.quat2mat(poses[i, 3:])
                    rot = np.asarray(rot, dtype=np.float32)

                    if objID > 5:
                        cls = objID + 2
                    elif objID > 2:
                        cls = objID + 1
                    else:
                        cls = objID

                    tDbox = rot[:3, :3].dot(threeD_boxes[cls, :, :].T).T
                    tDbox = tDbox + np.repeat(poses[i, np.newaxis, 0:3], 8, axis=0)

                    # if objID == 10 or objID == 11:
                    #    print(tf3d.euler.quat2euler(poses[i, 3:]))

                    box3D = toPix_array(tDbox, fx=fxkin, fy=fykin, cx=cxkin, cy=cykin)
                    box3D = np.reshape(box3D, (16))
                    box3D = box3D.tolist()
                    bb3vis.append(box3D)

                    feature_visibilities = calculate_feature_visibility(depth_refine, box3D, tDbox)
                    feat_visualization.append(feature_visibilities)
                    #print(feature_visibilities)

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
                        "feature_visibility": feature_visibilities
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

                        if cats[i] not in [4]:
                            continue

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

                            #colR = np.random.uniform(0, 255)
                            #colG = np.random.uniform(0, 255)
                            #colB = np.random.uniform(0, 255)

                            #colRinv = np.random.uniform(0, 255)
                            #colGinv = np.random.uniform(0, 255)
                            #colBinv = np.random.uniform(0, 255)

                            colR = 217
                            colG = 17
                            colB = 112

                            colRinv = 97
                            colGinv = 241
                            colBinv = 43

                            if feat_visualization[i][0] == 1:
                                colr = colR
                                colg = colG
                                colb = colB
                            else:
                                colr = colRinv
                                colg = colGinv
                                colb = colBinv
                            cv2.circle(img, (int(pose[0]), int(pose[1])), 4, (colr, colg, colb), 3)
                            if feat_visualization[i][1] == 1:
                                colr = colR
                                colg = colG
                                colb = colB
                            else:
                                colr = colRinv
                                colg = colGinv
                                colb = colBinv
                            cv2.circle(img, (int(pose[2]), int(pose[3])), 4, (colr, colg, colb), 3)
                            if feat_visualization[i][2] == 1:
                                colr = colR
                                colg = colG
                                colb = colB
                            else:
                                colr = colRinv
                                colg = colGinv
                                colb = colBinv
                            cv2.circle(img, (int(pose[4]), int(pose[5])), 4, (colr, colg, colb), 3)
                            if feat_visualization[i][3] == 1:
                                colr = colR
                                colg = colG
                                colb = colB
                            else:
                                colr = colRinv
                                colg = colGinv
                                colb = colBinv
                            cv2.circle(img, (int(pose[6]), int(pose[7])), 4, (colr, colg, colb), 3)
                            if feat_visualization[i][4] == 1:
                                colr = colR
                                colg = colG
                                colb = colB
                            else:
                                colr = colRinv
                                colg = colGinv
                                colb = colBinv
                            cv2.circle(img, (int(pose[8]), int(pose[9])), 4, (colr, colg, colb), 3)
                            if feat_visualization[i][5] == 1:
                                colr = colR
                                colg = colG
                                colb = colB
                            else:
                                colr = colRinv
                                colg = colGinv
                                colb = colBinv
                            cv2.circle(img, (int(pose[10]), int(pose[11])), 4, (colr, colg, colb), 3)
                            if feat_visualization[i][6] == 1:
                                colr = colR
                                colg = colG
                                colb = colB
                            else:
                                colr = colRinv
                                colg = colGinv
                                colb = colBinv
                            cv2.circle(img, (int(pose[12]), int(pose[13])), 4, (colr, colg, colb), 3)
                            if feat_visualization[i][7] == 1:
                                colr = colR
                                colg = colG
                                colb = colB
                            else:
                                colr = colRinv
                                colg = colGinv
                                colb = colBinv
                            cv2.circle(img, (int(pose[14]), int(pose[15])), 4, (colr, colg, colb), 3)

                            font = cv2.FONT_HERSHEY_COMPLEX
                            bottomLeftCornerOfText = (5, 10)
                            bottomLeftCornerOfText2 = (5, 20)
                            fontScale = 0.5
                            fontColor = (colR, colG, colB)
                            fontthickness = 2
                            lineType = 2
                            gtText_vis = 'visible'
                            gtText_invis = 'invisible'
                            fontColor2 = (colRinv, colGinv, colBinv)
                            fontthickness2 = 4
                            cv2.putText(img, gtText_vis,
                                        bottomLeftCornerOfText,
                                        font,
                                        fontScale,
                                        fontColor,
                                        fontthickness,
                                        lineType)
                            cv2.putText(img, gtText_invis,
                                        bottomLeftCornerOfText2,
                                        font,
                                        fontScale,
                                        fontColor2,
                                        fontthickness,
                                        lineType)

                            #img = cv2.line(img, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), (130, 245, 13), 2)
                            #img = cv2.line(img, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), (50, 112, 220), 2)
                            #img = cv2.line(img, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), (50, 112, 220), 2)
                            #img = cv2.line(img, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), (50, 112, 220), 2)
                            #img = cv2.line(img, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), (colR, colG, colB),
                            #               2)
                            #img = cv2.line(img, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()),
                            #               (colR, colG, colB), 2)
                            #img = cv2.line(img, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()),
                            #               (colR, colG, colB), 2)
                            #img = cv2.line(img, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()),
                            #               (colR, colG, colB), 2)
                            #img = cv2.line(img, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()),
                            #               (colR, colG, colB), 2)
                            #img = cv2.line(img, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()),
                            #               (colR, colG, colB), 2)
                            #img = cv2.line(img, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()),
                            #               (colR, colG, colB), 2)
                            #img = cv2.line(img, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()),
                            #               (colR, colG, colB), 2)

                    cv2.imwrite(fileName, img)

                    print('STOP')

    catsInt = range(1, 14)

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
