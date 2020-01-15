import yaml
import cv2
import numpy as np
import transforms3d as tf3d
import OpenEXR, Imath


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


def toPix(translation, fx=None, fy=None, cx=None, cy=None):

    xpix = ((translation[0] * fx) / translation[2]) + cx
    ypix = ((translation[1] * fy) / translation[2]) + cy
    #zpix = translation[2] * fxkin

    return [xpix, ypix] #, zpix]


def toPix_array(translation, fx=None, fy=None, cx=None, cy=None):

    xpix = ((translation[:, 0] * fx) / translation[:, 2]) + cx
    ypix = ((translation[:, 1] * fy) / translation[:, 2]) + cy
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


def calculate_feature_visibility(depth, features_2D, features_3D):

    feat_vis = []
    for f_i in range(0, len(features_2D), 2):
        feat_x = int(features_2D[f_i])
        feat_y = int(features_2D[f_i+1])

        if feat_x < 0 or feat_x > depth.shape[1]-1 or feat_y < 0 or feat_y > depth.shape[0]-1:
            feat_vis.append(0)
            continue
        feat_dep = depth[feat_y, feat_x]
        feat_gt = features_3D[int(f_i/2), 2]

        nbh = 3
        feat_nbh = depth[(feat_y-nbh):(feat_y+nbh), (feat_x-nbh):(feat_x+nbh)]
        feat_nbh = feat_nbh.flatten()

        #if feat_dep < (feat_gt * 1000.0)+20.0 and feat_dep > (feat_gt * 1000.0)-20.0:
        #    feat_vis.append(1)
        #else:
        #    feat_vis.append(0)

        # ugly but necessary due to quantization onto image grid sampled feature locations that might be self occluded but visibility of the relevant object part is given
        no_match = True
        for nbh_pixel in feat_nbh:
            if nbh_pixel < (feat_gt*1000.0) + 5.0 and nbh_pixel > (feat_gt*1000.0) - 5.0:
                feat_vis.append(1)
                no_match = False
                break
        if no_match == True:
            feat_vis.append(0)

    return feat_vis




def manipulate_depth(fn_gt, fn_depth, fn_part):

    fov = 57.8

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

    fin_depth = np.where(np.isfinite(depth), depth, np.NAN)
    if np.nanmean(fin_depth) < 0.5 or np.nanmean(fin_depth) > 4.0:
        print('invalid train image; range is wrong')
        return None, None, None, None, None, None

    partmask = cv2.imread(fn_part, 0)

    #print('partmask: ', np.nanmean(partmask))
    if np.nanmean(partmask) < 150.0:
        print('invalid visibility mask!')
        return None, None, None, None, None, None

    return depth, partmask, bboxes, poses, mask_ids, visibilities


def manipulate_RGB(fn_gt, fn_depth, fn_part, fn_rgb):

    fov = 57.8

    with open(fn_gt, 'r') as stream:
        query = yaml.load(stream)
        if query is None:
            print('Whatever is wrong there.... ¯\_(ツ)_/¯')
            return None, None, None, None, None, None, None

        rot = np.asarray(query['camera_rot'])
        rot = tf3d.euler.mat2euler(rot, 'sxyz')

        if rot[0] > -1.508:
            print('ANGLE ERROR.... ¯\_(ツ)_/¯')
            return None, None, None, None, None, None, None

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
        return None, None, None, None, None, None, None

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

    fin_depth = np.where(np.isfinite(depth), depth, np.NAN)
    #print('depth: ', np.nanmean(fin_depth))
    if np.nanmean(fin_depth) < 0.5 or np.nanmean(fin_depth) > 3.0:
        print('invalid train image; range is wrong')
        return None, None, None, None, None, None, None

    partmask = cv2.imread(fn_part, 0)

    # mask rendering failed for first dataset
    #print('partmask: ', np.nanmean(partmask))
    #if np.nanmean(partmask) < 150.0:
    #    print('invalid visibility mask!')
    #    return None, None, None, None, None, None, None

    rgb_img = cv2.imread(fn_rgb, 1)

    return depth, rgb_img, partmask, bboxes, poses, mask_ids, visibilities
