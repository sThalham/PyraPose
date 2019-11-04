import sys
import os
import subprocess
import yaml
import cv2
import numpy as np
import json
from scipy import ndimage, signal
import math
import datetime
import copy
import transforms3d as tf3d
import time
import itertools
import random
import pyfastnoisesimd as fns
import geometry

import OpenEXR, Imath
from pathlib import Path

dataset = 'linemod'
sensor_method = 'authentic'
resX = 640
resY = 480
if dataset is 'tless':
    resX = 720
    resY = 540
# fov = 1.0088002681732178
fov = 57.8
fxkin = 579.68  # blender calculated
fykin = 542.31  # blender calculated
cxkin = 320
cykin = 240
depthCut = 2000



#np.set_printoptions(threshold=np.nan)

threeD_boxes = np.ndarray((15, 8, 3), dtype=np.float32)
threeD_boxes[0, :, :] = np.array([[0.038, 0.039, 0.046],  # ape [76, 78, 92]
                                     [0.038, 0.039, -0.046],
                                     [0.038, -0.039, -0.046],
                                     [0.038, -0.039, 0.046],
                                     [-0.038, 0.039, 0.046],
                                     [-0.038, 0.039, -0.046],
                                     [-0.038, -0.039, -0.046],
                                     [-0.038, -0.039, 0.046]])
threeD_boxes[1, :, :] = np.array([[0.108, 0.061, 0.1095],  # benchvise [216, 122, 219]
                                     [0.108, 0.061, -0.1095],
                                     [0.108, -0.061, -0.1095],
                                     [0.108, -0.061, 0.1095],
                                     [-0.108, 0.061, 0.1095],
                                     [-0.108, 0.061, -0.1095],
                                     [-0.108, -0.061, -0.1095],
                                     [-0.108, -0.061, 0.1095]])
threeD_boxes[2, :, :] = np.array([[0.083, 0.0825, 0.037],  # bowl [166, 165, 74]
                                     [0.083, 0.0825, -0.037],
                                     [0.083, -0.0825, -0.037],
                                     [0.083, -0.0825, 0.037],
                                     [-0.083, 0.0825, 0.037],
                                     [-0.083, 0.0825, -0.037],
                                     [-0.083, -0.0825, -0.037],
                                     [-0.083, -0.0825, 0.037]])
threeD_boxes[3, :, :] = np.array([[0.0685, 0.0715, 0.05],  # camera [137, 143, 100]
                                     [0.0685, 0.0715, -0.05],
                                     [0.0685, -0.0715, -0.05],
                                     [0.0685, -0.0715, 0.05],
                                     [-0.0685, 0.0715, 0.05],
                                     [-0.0685, 0.0715, -0.05],
                                     [-0.0685, -0.0715, -0.05],
                                     [-0.0685, -0.0715, 0.05]])
threeD_boxes[4, :, :] = np.array([[0.0505, 0.091, 0.097],  # can [101, 182, 194]
                                     [0.0505, 0.091, -0.097],
                                     [0.0505, -0.091, -0.097],
                                     [0.0505, -0.091, 0.097],
                                     [-0.0505, 0.091, 0.097],
                                     [-0.0505, 0.091, -0.097],
                                     [-0.0505, -0.091, -0.097],
                                     [-0.0505, -0.091, 0.097]])
threeD_boxes[5, :, :] = np.array([[0.0335, 0.064, 0.0585],  # cat [67, 128, 117]
                                     [0.0335, 0.064, -0.0585],
                                     [0.0335, -0.064, -0.0585],
                                     [0.0335, -0.064, 0.0585],
                                     [-0.0335, 0.064, 0.0585],
                                     [-0.0335, 0.064, -0.0585],
                                     [-0.0335, -0.064, -0.0585],
                                     [-0.0335, -0.064, 0.0585]])
threeD_boxes[6, :, :] = np.array([[0.059, 0.046, 0.0475],  # mug [118, 92, 95]
                                     [0.059, 0.046, -0.0475],
                                     [0.059, -0.046, -0.0475],
                                     [0.059, -0.046, 0.0475],
                                    [-0.059, 0.046, 0.0475],
                                    [-0.059, 0.046, -0.0475],
                                     [-0.059, -0.046, -0.0475],
                                     [-0.059, -0.046, 0.0475]])
threeD_boxes[7, :, :] = np.array([[0.115, 0.038, 0.104],  # drill [230, 76, 208]
                                     [0.115, 0.038, -0.104],
                                     [0.115, -0.038, -0.104],
                                     [0.115, -0.038, 0.104],
                                     [-0.115, 0.038, 0.104],
                                     [-0.115, 0.038, -0.104],
                                     [-0.115, -0.038, -0.104],
                                     [-0.115, -0.038, 0.104]])
threeD_boxes[8, :, :] = np.array([[0.052, 0.0385, 0.043],  # duck [104, 77, 86]
                                     [0.052, 0.0385, -0.043],
                                     [0.052, -0.0385, -0.043],
                                     [0.052, -0.0385, 0.043],
                                     [-0.052, 0.0385, 0.043],
                                     [-0.052, 0.0385, -0.043],
                                     [-0.052, -0.0385, -0.043],
                                     [-0.052, -0.0385, 0.043]])
threeD_boxes[9, :, :] = np.array([[0.075, 0.0535, 0.0345],  # eggbox [150, 107, 69]
                                     [0.075, 0.0535, -0.0345],
                                     [0.075, -0.0535, -0.0345],
                                     [0.075, -0.0535, 0.0345],
                                     [-0.075, 0.0535, 0.0345],
                                     [-0.075, 0.0535, -0.0345],
                                     [-0.075, -0.0535, -0.0345],
                                     [-0.075, -0.0535, 0.0345]])
threeD_boxes[10, :, :] = np.array([[0.0185, 0.039, 0.0865],  # glue [37, 78, 173]
                                     [0.0185, 0.039, -0.0865],
                                     [0.0185, -0.039, -0.0865],
                                     [0.0185, -0.039, 0.0865],
                                     [-0.0185, 0.039, 0.0865],
                                     [-0.0185, 0.039, -0.0865],
                                     [-0.0185, -0.039, -0.0865],
                                     [-0.0185, -0.039, 0.0865]])
threeD_boxes[11, :, :] = np.array([[0.0505, 0.054, 0.04505],  # holepuncher [101, 108, 91]
                                     [0.0505, 0.054, -0.04505],
                                     [0.0505, -0.054, -0.04505],
                                     [0.0505, -0.054, 0.04505],
                                     [-0.0505, 0.054, 0.04505],
                                     [-0.0505, 0.054, -0.04505],
                                     [-0.0505, -0.054, -0.04505],
                                     [-0.0505, -0.054, 0.04505]])
threeD_boxes[12, :, :] = np.array([[0.115, 0.038, 0.104],  # drill [230, 76, 208]
                                     [0.115, 0.038, -0.104],
                                     [0.115, -0.038, -0.104],
                                     [0.115, -0.038, 0.104],
                                     [-0.115, 0.038, 0.104],
                                     [-0.115, 0.038, -0.104],
                                     [-0.115, -0.038, -0.104],
                                     [-0.115, -0.038, 0.104]])
threeD_boxes[13, :, :] = np.array([[0.129, 0.059, 0.0705],  # iron [258, 118, 141]
                                     [0.129, 0.059, -0.0705],
                                     [0.129, -0.059, -0.0705],
                                     [0.129, -0.059, 0.0705],
                                     [-0.129, 0.059, 0.0705],
                                     [-0.129, 0.059, -0.0705],
                                     [-0.129, -0.059, -0.0705],
                                     [-0.129, -0.059, 0.0705]])
threeD_boxes[14, :, :] = np.array([[0.047, 0.0735, 0.0925],  # phone [94, 147, 185]
                                     [0.047, 0.0735, -0.0925],
                                     [0.047, -0.0735, -0.0925],
                                     [0.047, -0.0735, 0.0925],
                                     [-0.047, 0.0735, 0.0925],
                                     [-0.047, 0.0735, -0.0925],
                                     [-0.047, -0.0735, -0.0925],
                                     [-0.047, -0.0735, 0.0925]])


def draw_axis(img, poses):
    # unit is mm
    points = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)

    rotMat = tf3d.quaternions.quat2mat(poses[3:7])
    rot, _ = cv2.Rodrigues(rotMat)
    tra = poses[0:3] * 1000.0
    K = np.float32([fxkin, 0., cxkin, 0., fykin, cykin, 0., 0., 1.]).reshape(3,3)
    axisPoints, _ = cv2.projectPoints(points, rot, tra, K, (0, 0, 0, 0))

    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255,0,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0,0,255), 3)
    return img


def toPix(translation):

    xpix = ((translation[0] * fxkin) / translation[2]) + cxkin
    ypix = ((translation[1] * fykin) / translation[2]) + cykin
    #zpix = translation[2] * fxkin

    return [xpix, ypix] #, zpix]


def toPix_array(translation):

    xpix = ((translation[:, 0] * fxkin) / translation[:, 2]) + cxkin
    ypix = ((translation[:, 1] * fykin) / translation[:, 2]) + cykin
    #zpix = translation[2] * fxkin

    return np.stack((xpix, ypix), axis=1) #, zpix]


def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def manipulate_depth(fn_gt, fn_depth, fn_part):

    with open(fn_gt, 'r') as stream:
        query = yaml.load(stream)
        if query is None:
            print('Whatever is wrong there.... ¯\_(ツ)_/¯')
            return None, None, None, None, None, None
        bboxes = np.zeros((len(query), 5), np.int)
        poses = np.zeros((len(query), 7), np.float32)
        mask_ids = np.zeros((len(query)), np.int)
        visibilities = np.zeros((len(query)), np.float32)
        for j in range(len(query)-1): # skip cam pose
            qr = query[j]
            class_id = qr['class_id']
            bbox = qr['bbox']
            mask_ids[j] = int(qr['mask_id'])
            visibilities[j] = float(qr['visibility'])
            pose = np.array(qr['pose']).reshape(4, 4)
            bboxes[j, 0] = class_id
            bboxes[j, 1:5] = np.array(bbox)
            q_pose = tf3d.quaternions.mat2quat(pose[:3, :3])
            poses[j, 3:7] = np.array(q_pose)
            poses[j, 0:3] = np.array([pose[0, 3], pose[1, 3], pose[2, 3]])

    if bboxes.shape[0] < 2:
        print('invalid train image, no bboxes in fov')
        return None, None, None, None, None, None

    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    golden = OpenEXR.InputFile(fn_depth)
    dw = golden.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    redstr = golden.channel('R', pt)
    depth = np.fromstring(redstr, dtype=np.float32)
    depth.shape = (size[1], size[0])

    centerX = depth.shape[1] / 2.0
    centerY = depth.shape[0] / 2.0

    uv_table = np.zeros((depth.shape[0], depth.shape[1], 2), dtype=np.int16)
    column = np.arange(0, depth.shape[0])
    uv_table[:, :, 1] = np.arange(0, depth.shape[1]) - centerX
    uv_table[:, :, 0] = column[:, np.newaxis] - centerY
    uv_table = np.abs(uv_table)

    depth = depth * np.cos(np.radians(fov / depth.shape[1] * np.abs(uv_table[:, :, 1]))) * np.cos(
        np.radians(fov / depth.shape[1] * uv_table[:, :, 0]))

    #print('depth: ', np.nanmean(depth))
    if np.nanmean(depth) < 0.5 or np.nanmean(depth) > 4.0:
        print('invalid train image; range is wrong')
        return None, None, None, None, None, None

    partmask = cv2.imread(fn_part, 0)

    #print('partmask: ', np.nanmean(partmask))
    if np.nanmean(partmask) < 150.0:
        print('invalid visibility mask!')
        return None, None, None, None, None, None

    return depth, partmask, bboxes, poses, mask_ids, visibilities


def augmentDepth(depth, obj_mask, mask_ori, shadowClK, shadowMK, blurK, blurS, depthNoise, method):

    sensor = True
    simplex = True

    if method == 0:
        pass
    elif method == 1:
        sensor = True
        simplex = False
    elif method == 2:
        sensor = False
        simplex = True

    # erode and blur mask to get more realistic appearance
    partmask = mask_ori
    partmask = partmask.astype(np.float32)
    #mask = partmask > (np.median(partmask) * 0.4)
    partmask = np.where(partmask > 0.0, 255.0, 0.0)

    cv2.imwrite('/home/sthalham/partmask.png', partmask)

    # apply shadow
    kernel = np.ones((shadowClK, shadowClK))
    partmask = cv2.morphologyEx(partmask, cv2.MORPH_OPEN, kernel)
    partmask = signal.medfilt2d(partmask, kernel_size=shadowMK)
    partmask = partmask.astype(np.uint8)
    mask = partmask > 20
    depth = np.where(mask, depth, 0.0)

    if sensor is True:
        depthFinal = cv2.resize(depth, None, fx=1 / 2, fy=1 / 2)
        res = (((depthFinal / 1000.0) * 1.41421356) ** 2)
        depthFinal = cv2.GaussianBlur(depthFinal, (blurK, blurK), blurS, blurS)
        # quantify to depth resolution and apply gaussian
        dNonVar = np.divide(depthFinal, res, out=np.zeros_like(depthFinal), where=res != 0)
        dNonVar = np.round(dNonVar)
        dNonVar = np.multiply(dNonVar, res)
        noise = np.multiply(dNonVar, depthNoise)
        depthFinal = np.random.normal(loc=dNonVar, scale=noise, size=dNonVar.shape)

        depth = cv2.resize(depthFinal, (resX, resY))

    if simplex is True:
        # fast perlin noise
        seed = np.random.randint(2 ** 31)
        N_threads = 4
        perlin = fns.Noise(seed=seed, numWorkers=N_threads)
        drawFreq = random.uniform(0.05, 0.2)  # 0.05 - 0.2
        # drawFreq = 0.5
        perlin.frequency = drawFreq
        perlin.noiseType = fns.NoiseType.SimplexFractal
        perlin.fractal.fractalType = fns.FractalType.FBM
        drawOct = [4, 8]
        freqOct = np.bincount(drawOct)
        rndOct = np.random.choice(np.arange(len(freqOct)), 1, p=freqOct / len(drawOct), replace=False)
        perlin.fractal.octaves = rndOct
        perlin.fractal.lacunarity = 2.1
        perlin.fractal.gain = 0.45
        perlin.perturb.perturbType = fns.PerturbType.NoPerturb

        # linemod
        if not sensor:
            # noise according to keep it unreal
            #noiseX = np.random.uniform(0.0001, 0.1, resX * resY) # 0.0001 - 0.1
            #noiseY = np.random.uniform(0.0001, 0.1, resX * resY) # 0.0001 - 0.1
            #noiseZ = np.random.uniform(0.01, 0.1, resX * resY) # 0.01 - 0.1
            #Wxy = np.random.randint(0, 10) # 0 - 10
            #Wz = np.random.uniform(0.0, 0.005) #0 - 0.005
            noiseX = np.random.uniform(0.001, 0.01, resX * resY)  # 0.0001 - 0.1
            noiseY = np.random.uniform(0.001, 0.01, resX * resY)  # 0.0001 - 0.1
            noiseZ = np.random.uniform(0.01, 0.1, resX * resY)  # 0.01 - 0.1
            Wxy = np.random.randint(2, 5)  # 0 - 10
            Wz = np.random.uniform(0.0001, 0.004)  # 0 - 0.005
        else:
            noiseX = np.random.uniform(0.001, 0.01, resX * resY) # 0.0001 - 0.1
            noiseY = np.random.uniform(0.001, 0.01, resX * resY) # 0.0001 - 0.1
            noiseZ = np.random.uniform(0.01, 0.1, resX * resY) # 0.01 - 0.1
            Wxy = np.random.randint(1, 5) # 1 - 5
            Wz = np.random.uniform(0.0001, 0.004) #0.0001 - 0.004
        # tless
        #noiseX = np.random.uniform(0.001, 0.1, resX * resY)  # 0.0001 - 0.1
        #noiseY = np.random.uniform(0.001, 0.1, resX * resY)  # 0.0001 - 0.1
        #noiseZ = np.random.uniform(0.01, 0.1, resX * resY)  # 0.01 - 0.1
        #Wxy = np.random.randint(2, 8)  # 0 - 10
        #Wz = np.random.uniform(0.0, 0.005)


        X, Y = np.meshgrid(np.arange(resX), np.arange(resY))
        coords0 = fns.empty_coords(resX * resY)
        coords1 = fns.empty_coords(resX * resY)
        coords2 = fns.empty_coords(resX * resY)

        coords0[0, :] = noiseX.ravel()
        coords0[1, :] = Y.ravel()
        coords0[2, :] = X.ravel()
        VecF0 = perlin.genFromCoords(coords0)
        VecF0 = VecF0.reshape((resY, resX))

        coords1[0, :] = noiseY.ravel()
        coords1[1, :] = Y.ravel()
        coords1[2, :] = X.ravel()
        VecF1 = perlin.genFromCoords(coords1)
        VecF1 = VecF1.reshape((resY, resX))

        coords2[0, :] = noiseZ.ravel()
        coords2[1, :] = Y.ravel()
        coords2[2, :] = X.ravel()
        VecF2 = perlin.genFromCoords(coords2)
        VecF2 = VecF2.reshape((resY, resX))

        x = np.arange(resX, dtype=np.uint16)
        x = x[np.newaxis, :].repeat(resY, axis=0)
        y = np.arange(resY, dtype=np.uint16)
        y = y[:, np.newaxis].repeat(resX, axis=1)

        # vanilla
        #fx = x + Wxy * VecF0
        #fy = y + Wxy * VecF1
        #fx = np.where(fx < 0, 0, fx)
        #fx = np.where(fx >= resX, resX - 1, fx)
        #fy = np.where(fy < 0, 0, fy)
        #fy = np.where(fy >= resY, resY - 1, fy)
        #fx = fx.astype(dtype=np.uint16)
        #fy = fy.astype(dtype=np.uint16)
        #Dis = depth[fy, fx] + Wz * VecF2
        #depth = np.where(Dis > 0, Dis, 0.0)

        #print(x.shape)
        #print(np.amax(depth))
        #print(np.amin(depth))
        Wxy_scaled = depth * 0.001 * Wxy
        Wz_scaled = depth * 0.001 * Wz
        # scale with depth
        fx = x + Wxy_scaled * VecF0
        fy = y + Wxy_scaled * VecF1
        fx = np.where(fx < 0, 0, fx)
        fx = np.where(fx >= resX, resX - 1, fx)
        fy = np.where(fy < 0, 0, fy)
        fy = np.where(fy >= resY, resY - 1, fy)
        fx = fx.astype(dtype=np.uint16)
        fy = fy.astype(dtype=np.uint16)
        Dis = depth[fy, fx] + Wz_scaled * VecF2
        depth = np.where(Dis > 0, Dis, 0.0)

    return depth


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

    depth_imp = copy.deepcopy(depth_refine)

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

    return cross, depth_refine, depth_imp


##########################
#         MAIN           #
##########################
if __name__ == "__main__":

    #root = '/home/sthalham/data/renderings/linemod_BG/patches31052018/patches'  # path to train samples

    root = '/home/sthalham/data/renderings/linemod/patches'
    target = '/home/sthalham/data/prepro/linemod_baseline_10kx2/'

    # [depth, normals, sensor, simplex, full]
    method = 'full'
    visu = False
    #n_samples = 22986 # real=1214
    n_samples = 11493
    if dataset is 'tless':
        n_samples = 2524

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
    all = n_samples
    times = []

    trainN = 1
    testN = 1
    valN = 1

    depPath = root + "/depth/"
    partPath = root + "/part/"
    gtPath = root
    maskPath = root + "/mask/"
    excludedImgs = []
    boxWidths = []
    boxHeights = []

    syns = os.listdir(root)
    walkit = random.sample(syns, n_samples)
    for fileInd in walkit:
        #if fileInd.endswith(".yaml") and len(os.listdir(target + 'images/train')) < n_samples:
        if fileInd.endswith(".yaml"):

            start_time = time.time()
            gloCo = gloCo + 1

            redname = fileInd[:-8]

            gtfile = gtPath + '/' + fileInd
            depfile = depPath + redname + "_depth.exr"
            partfile = partPath + redname + "_part.png"
            maskfile = maskPath + redname + "_mask.npy"

            depth_refine, mask, bboxes, poses, mask_ids, visibilities = manipulate_depth(gtfile, depfile, partfile)
            try:
                obj_mask = np.load(maskfile)
            except Exception:
                continue
            obj_mask = obj_mask.astype(np.int8)

            if bboxes is None:
                excludedImgs.append(int(redname))
                continue

            #wanna_cls = 5
            #check_cls = np.where(bboxes[:-1, 0] == (wanna_cls-1))
            #check_vis = visibilities[check_cls].astype(np.float32)
            #print(len(check_cls), check_vis < 0.7, check_vis)
            #if not visibilities[check_cls] or check_vis < 0.7:
            #    print('no cls ', wanna_cls)
            #    continue

            depth_refine = np.multiply(depth_refine, 1000.0)  # to millimeters
            rows, cols = depth_refine.shape

            for k in range(0, 2):

                newredname = redname[1:] + str(k)

                print(newredname)

                fileName = target + "images/train/" + newredname + '.jpg'
                myFile = Path(fileName)
                #print(newredname)
                #print(myFile)

                if myFile.exists():
                    print('File exists, skip encoding and safing.')

                else:
                    if method == 'normals':
                        normals, depth_refine_aug, depth_imp = get_normal(depth_refine, fx=fxkin, fy=fykin, cx=cxkin, cy=cykin,
                                                           for_vis=False)

                    elif method == 'simplex':
                        drawKern = [3, 5, 7]
                        freqKern = np.bincount(drawKern)
                        kShadow = np.random.choice(np.arange(len(freqKern)), 1, p=freqKern / len(drawKern), replace=False)
                        kMed = np.random.choice(np.arange(len(freqKern)), 1, p=freqKern / len(drawKern), replace=False)
                        kBlur = np.random.choice(np.arange(len(freqKern)), 1, p=freqKern / len(drawKern), replace=False)
                        sBlur = random.uniform(0.25, 3.5)
                        sDep = random.uniform(0.002, 0.004)
                        kShadow.astype(int)
                        kMed.astype(int)
                        kBlur.astype(int)
                        kShadow = kShadow[0]
                        kMed = kMed[0]
                        kBlur = kBlur[0]
                        augmentation_var = 2  # [0 = full, 1 = sensor, 2 = simplex]
                        depthAug = augmentDepth(depth_refine, obj_mask, mask, kShadow, kMed, kBlur, sBlur, sDep, augmentation_var)

                        aug_xyz, depth_refine_aug, depth_imp = get_normal(depthAug, fx=fxkin, fy=fykin, cx=cxkin, cy=cykin,
                                                         for_vis=False)

                        #depth_imp[depth_imp > depthCut] = 0
                        #scaCro = 255.0 / np.nanmax(depth_imp)
                        #cross = np.multiply(depth_imp, scaCro)


                        #dep_sca = cross.astype(np.uint8)
                        #cv2.imwrite(fileName, dep_sca)
                        #aug_xyz[:, :, 2] = dep_sca
                        cv2.imwrite(fileName, aug_xyz)

                    elif method == 'full':
                        drawKern = [3, 5, 7]
                        freqKern = np.bincount(drawKern)
                        kShadow = np.random.choice(np.arange(len(freqKern)), 1, p=freqKern / len(drawKern), replace=False)
                        kMed = np.random.choice(np.arange(len(freqKern)), 1, p=freqKern / len(drawKern), replace=False)
                        kBlur = np.random.choice(np.arange(len(freqKern)), 1, p=freqKern / len(drawKern), replace=False)
                        sBlur = random.uniform(0.25, 1.5)
                        sDep = random.uniform(0.002, 0.004)
                        kShadow.astype(int)
                        kMed.astype(int)
                        kBlur.astype(int)
                        kShadow = kShadow[0]
                        kMed = kMed[0]
                        kBlur = kBlur[0]
                        augmentation_var = 0  # [0 = full, 1 = sensor, 2 = simplex]
                        #depthAug = augmentDepth(depth_refine, obj_mask, mask, kShadow, kMed, kBlur, sBlur, sDep, augmentation_var)
                        depthAug = augmentDepth(depth_refine, obj_mask, mask, kShadow, kMed, kBlur, sBlur, sDep,
                                            augmentation_var)

                        depthAug[depthAug > depthCut] = 0

                        # LOGNORM
                        depth_drs = np.log(depthAug) + 2.0
                        scaCro = 255.0 / (np.log(depthCut) + 2.0)
                        cross = np.multiply(depth_drs, scaCro)

                        # MAXNORM
                        #scaCro = 255.0 / np.nanmax(depthAug)
                        #cross = np.multiply(depthAug, scaCro)

                        aug_xyz = cross.astype(np.uint8)
                        aug_xyz = np.repeat(aug_xyz[:, :, np.newaxis], 3, 2)
                        #aug_xyz, depth_refine_aug, depth_imp = get_normal(depthAug, fx=fxkin, fy=fykin, cx=cxkin, cy=cykin,
                        #                                 for_vis=False)

                        cv2.imwrite(fileName, aug_xyz)

                imgID = int(newredname)
                imgName = newredname + '.jpg'
                #print(imgName)

                # bb scaling because of image scaling
                bbvis = []
                #bbvis = (bboxes * bbsca).astype(int)
                #bbvis = bbvis.astype(int)
                bb3vis = []
                cats = []
                posvis = []
                postra = []
                #print(imgName)
                for i, bbox in enumerate(bboxes[:-1]):

                    #if (np.asscalar(bbox[0]) + 1) != wanna_cls:
                    #    continue

                    if visibilities[i] < 0.5:
                        print('visivility: ', visibilities[i], ' skip!')
                        continue

                    bbvis.append(bbox.astype(int))
                    objID = np.asscalar(bbox[0]) + 1
                    cats.append(objID)

                    bbox = (bbox).astype(int)

                    rot = tf3d.quaternions.quat2mat(poses[i, 3:])
                    rot = np.asarray(rot, dtype=np.float32)

                    tDbox = rot.dot(threeD_boxes[bbox[0], :, :].T).T
                    tDbox = tDbox + np.repeat(poses[i, np.newaxis, 0:3], 8, axis=0)

                    #if objID == 10 or objID == 11:
                    #    print(tf3d.euler.quat2euler(poses[i, 3:]))

                    box3D = toPix_array(tDbox)
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
                                np.asscalar(poses[i,3]), np.asscalar(poses[i,4]), np.asscalar(poses[i,5]),
                                np.asscalar(poses[i,6])]
                    if i != len(bboxes):
                        pose[0:2] = toPix(pose[0:3])

                    posvis.append(pose)
                    tra = np.asarray(poses[i, :3], dtype=np.float32)
                    postra.append(tra)

                    #if pose[3] < 0.0:
                    #    raise ValueError('w < 0.0')

                    #print(objID)
                    annoID = annoID + 1
                    tempTA = {
                        "id": annoID,
                        "image_id": imgID,
                        "category_id": objID,
                        "bbox": bb,
                        "pose": pose,
                        "segmentation": box3D,
                        "area": area,
                        "iscrowd": 0
                    }
                    #print('norm q: ', np.linalg.norm(pose[3:]))

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
                meantime = sum(times)/len(times)
                eta = ((all - gloCo) * meantime) / 60
                if gloCo % 100 == 0:
                    print('eta: ', eta, ' min')
                    times = []

                if visu is True:
                    img = aug_xyz
                    #img = cv2.imread(fileName, cv2.IMREAD_UNCHANGED)
                    for i, bb in enumerate(bbvis):

                        cv2.rectangle(aug_xyz, (int(bb[2]), int(bb[1])), (int(bb[4]), int(bb[3])),
                                      (255, 255, 255), 2)
                        cv2.rectangle(img, (int(bb[2]), int(bb[1])), (int(bb[4]), int(bb[3])),
                                      (0, 0, 0), 1)
                        #
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        bottomLeftCornerOfText = (int(bb[2]), int(bb[1]))
                        fontScale = 1
                        fontColor = (0, 0, 0)
                        fontthickness = 1
                        lineType = 2
                        gtText = str(cats[i])

                        fontColor2 = (255, 255, 255)
                        fontthickness2 = 3
                        cv2.putText(aug_xyz, gtText,
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor2,
                                fontthickness2,
                                lineType)

                        cv2.putText(aug_xyz, gtText,
                                bottomLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                fontthickness,
                                lineType)


                        #print(posvis[i])
                        if i is not poses.shape[0]:
                            pose = np.asarray(bb3vis[i], dtype=np.float32)

                            #pose2D = posvis[i]
                            #print(str(cats[i]))
                            
                            colR = random.randint(0, 255)
                            colG = random.randint(0, 255)
                            colB = random.randint(0, 255)
                            '''
                            #rot_lie = [[0.0, pose[3], pose[4]], [-pose[3], 0.0, pose[5]], [-pose[4], -pose[5], 0.0]]
                            #ssm =np.asarray(rot_lie, dtype=np.float32)
                            #map = geometry.rotations.map_hat(ssm)
                            #quat = tf3d.euler.euler2quat(pose2D[3], pose2D[4], pose2D[5])
                            #quat = quat / np.linalg.norm(quat)
                            pose2D = np.concatenate([postra[i], pose2D[3:]])
                            #print('x: ', (pose[0]-bb[2])/(bb[4]-bb[2]))
                            #print('y: ', (pose[1] - bb[1]) / (bb[3] - bb[1]))
                            #print(pose[0:2], bb[1:])

                            #cv2.circle(img, (int(pose[0]), int(pose[1])), 5, (0, 255, 0), 3)
                            draw_axis(aug_xyz, pose2D)
                            '''

                            img = cv2.line(img, tuple(pose[0:2].ravel()), tuple(pose[2:4].ravel()), (colR, colG, colB), 2)
                            img = cv2.line(img, tuple(pose[2:4].ravel()), tuple(pose[4:6].ravel()), (colR, colG, colB), 2)
                            img = cv2.line(img, tuple(pose[4:6].ravel()), tuple(pose[6:8].ravel()), (colR, colG, colB), 2)
                            img = cv2.line(img, tuple(pose[6:8].ravel()), tuple(pose[0:2].ravel()), (colR, colG, colB), 2)
                            img = cv2.line(img, tuple(pose[0:2].ravel()), tuple(pose[8:10].ravel()), (colR, colG, colB), 2)
                            img = cv2.line(img, tuple(pose[2:4].ravel()), tuple(pose[10:12].ravel()), (colR, colG, colB), 2)
                            img = cv2.line(img, tuple(pose[4:6].ravel()), tuple(pose[12:14].ravel()), (colR, colG, colB), 2)
                            img = cv2.line(img, tuple(pose[6:8].ravel()), tuple(pose[14:16].ravel()), (colR, colG, colB), 2)
                            img = cv2.line(img, tuple(pose[8:10].ravel()), tuple(pose[10:12].ravel()), (colR, colG, colB), 2)
                            img = cv2.line(img, tuple(pose[10:12].ravel()), tuple(pose[12:14].ravel()), (colR, colG, colB), 2)
                            img = cv2.line(img, tuple(pose[12:14].ravel()), tuple(pose[14:16].ravel()), (colR, colG, colB), 2)
                            img = cv2.line(img, tuple(pose[14:16].ravel()), tuple(pose[8:10].ravel()), (colR, colG, colB), 2)

                    cv2.imwrite(fileName, img)

                    print('STOP')

    catsInt = range(1,16)
    #catsInt = [wanna_cls]

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
    #boxWidths = np.asarray(boxWidths, np.float32)
    #boxHeigths = np.asarray(boxHeigths, np.float32)
    #print('box widths min and max: ', np.nanmin(boxWidths), np.nanmax(boxWidths))
    #print('box widths min and max: ', np.nanmin(boxHeights), np.nanmax(boxHeigths))
    print('Chill for once in your life... everything\'s done')
