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

import OpenEXR, Imath
from pathlib import Path

from annotation_scripts.misc import manipulate_RGB, toPix_array, toPix
from annotation_scripts.Augmentations import augmentDepth, augmentRGB, get_normal


if __name__ == "__main__":

    root = '/home/stefan/data/rendered_data/linemod_rgbd/patches'
    target = '/home/stefan/data/train_data/linemod_RGBD/'
    mesh_info = '/home/stefan/data/Meshes/linemod_13/models_info.yml'

    visu = False
    resX = 640
    resY = 480
    fxkin = 579.68  # blender calculated
    fykin = 542.31  # blender calculated
    cxkin = 320
    cykin = 240
    depthCut = 2000

    threeD_boxes = np.ndarray((31, 8, 3), dtype=np.float32)

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

    syns = os.listdir(root)
    all = len(syns)
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

                if myFile.exists():
                    print('File exists, skip encoding and safing.')

                else:
                    depthAug = augmentDepth(depth_refine, obj_mask, mask)
                    rgbAug = augmentRGB(rgb_refine)

                    #aug_xyz, depth_refine_aug, depth_imp = get_normal(depthAug, fx=fxkin, fy=fykin, cx=cxkin, cy=cykin,
                    #                                                  for_vis=False)

                    depthAug[depthAug > depthCut] = 0
                    scaCro = 255.0 / np.nanmax(depthAug)
                    cross = np.multiply(depthAug, scaCro)
                    aug_dep = cross.astype(np.uint8)
                    #aug_dep = np.repeat(aug_dep[:, :, np.newaxis], 3, 2)

                    cv2.imwrite(fileName, rgbAug)
                    cv2.imwrite(fileName[:-8] + '_dep.jpg', aug_dep)
                    #img_rgbd = np.concatenate((rgbAug, aug_dep[:, :, np.newaxis]), axis=2)
                    #cv2.imwrite(fileName, img_rgbd)
                    #np.save(fileName, img_rgbd)

                imgID = int(newredname)
                imgName = newredname + '_rgb.jpg'
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

                    bbvis.append(bbox.astype(int))
                    objID = np.asscalar(bbox[0]) + 1
                    cats.append(objID)

                    bbox = (bbox).astype(int)

                    rot = tf3d.quaternions.quat2mat(poses[i, 3:])
                    rot = np.asarray(rot, dtype=np.float32)

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

    print('Chill for once in your life... everything\'s done')
