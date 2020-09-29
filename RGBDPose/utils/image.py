"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import division
import numpy as np
import cv2
from PIL import Image
import imgaug.augmenters as iaa
#import pyfastnoisesimd as fns
import random
from scipy import ndimage

from .transform import change_transform_origin


def read_image_bgr(path):
    """ Read an image in BGR format.

    Args
        path: Path to the image.
    """
    image = np.asarray(Image.open(path).convert('RGB'))
    return image[:, :, ::-1].copy()


def read_image_dep(path):
    """ Read an image in BGR format.

    Args
        path: Path to the image.
    """
    image = np.asarray(Image.open(path))
    return image[:, :].copy()


def preprocess_image(x, mode='caffe'):
    """ Preprocess an image by subtracting the ImageNet mean.

    Args
        x: np.array of shape (None, None, 3) or (3, None, None).
        mode: One of "caffe" or "tf".
            - caffe: will zero-center each color channel with
                respect to the ImageNet dataset, without scaling.
            - tf: will scale pixels between -1 and 1, sample-wise.

    Returns
        The input with the ImageNet mean subtracted.
    """
    # mostly identical to "https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py"
    # except for converting RGB -> BGR since we assume BGR already

    # covert always to float32 to keep compatibility with opencv
    x = x.astype(np.float32)

    if mode == 'tf':
        x /= 127.5
        x -= 1.
    elif mode == 'caffe':
        x[..., 0] -= 103.939
        x[..., 1] -= 116.779
        x[..., 2] -= 123.68

    return x


def adjust_transform_for_image(transform, image, relative_translation):
    """ Adjust a transformation for a specific image.

    The translation of the matrix will be scaled with the size of the image.
    The linear part of the transformation will adjusted so that the origin of the transformation will be at the center of the image.
    """
    height, width, channels = image[0].shape

    result = transform

    # Scale the translation with the image size if specified.
    if relative_translation:
        result[0:2, 2] *= [width, height]

    # Move the origin of transformation.
    result = change_transform_origin(transform, (0.5 * width, 0.5 * height))

    return result


def adjust_transform_for_mask(transform, image, relative_translation):
    """ Adjust a transformation for a specific image.

    The translation of the matrix will be scaled with the size of the image.
    The linear part of the transformation will adjusted so that the origin of the transformation will be at the center of the image.
    """
    height, width = image.shape

    result = transform.copy()

    # Scale the translation with the image size if specified.
    if relative_translation:
        result[0:2, 2] *= [width, height]

    # Move the origin of transformation.
    result = change_transform_origin(transform, (0.5 * width, 0.5 * height))

    return result


class TransformParameters:
    """ Struct holding parameters determining how to apply a transformation to an image.

    Args
        fill_mode:             One of: 'constant', 'nearest', 'reflect', 'wrap'
        interpolation:         One of: 'nearest', 'linear', 'cubic', 'area', 'lanczos4'
        cval:                  Fill value to use with fill_mode='constant'
        relative_translation:  If true (the default), interpret translation as a factor of the image size.
                               If false, interpret it as absolute pixels.
    """
    def __init__(
        self,
        fill_mode            = 'nearest',
        interpolation        = 'linear',
        cval                 = 0,
        relative_translation = True,
    ):
        self.fill_mode            = fill_mode
        self.cval                 = cval
        self.interpolation        = interpolation
        self.relative_translation = relative_translation

    def cvBorderMode(self):
        if self.fill_mode == 'constant':
            return cv2.BORDER_CONSTANT
        if self.fill_mode == 'nearest':
            return cv2.BORDER_REPLICATE
        if self.fill_mode == 'reflect':
            return cv2.BORDER_REFLECT_101
        if self.fill_mode == 'wrap':
            return cv2.BORDER_WRAP

    def cvInterpolation(self):
        if self.interpolation == 'nearest':
            return cv2.INTER_NEAREST
        if self.interpolation == 'linear':
            return cv2.INTER_LINEAR
        if self.interpolation == 'cubic':
            return cv2.INTER_CUBIC
        if self.interpolation == 'area':
            return cv2.INTER_AREA
        if self.interpolation == 'lanczos4':
            return cv2.INTER_LANCZOS4


def apply_transform(matrix, image, params, cpara):

    # rgb
    # seq describes an object for rgb image augmentation using aleju/imgaug
    '''
    seq = iaa.Sequential([
        # blur
        iaa.SomeOf((0, 2), [
            iaa.GaussianBlur((0.0, 2.0)),
            iaa.AverageBlur(k=(3, 7)),
            iaa.MedianBlur(k=(3, 7)),
            iaa.BilateralBlur(d=(1, 7)),
            iaa.MotionBlur(k=(3, 7))
        ]),
        # color
        iaa.SomeOf((0, 2), [
            # iaa.WithColorspace(),
            iaa.AddToHueAndSaturation((-15, 15)),
            # iaa.ChangeColorspace(to_colorspace[], alpha=0.5),
            iaa.Grayscale(alpha=(0.0, 0.2))
        ]),
        # brightness
        iaa.OneOf([
            iaa.Sequential([
                iaa.Add((-10, 10), per_channel=0.5),
                iaa.Multiply((0.75, 1.25), per_channel=0.5)
            ]),
            iaa.Add((-10, 10), per_channel=0.5),
            iaa.Multiply((0.75, 1.25), per_channel=0.5),
            iaa.FrequencyNoiseAlpha(
                exponent=(-4, 0),
                first=iaa.Multiply((0.75, 1.25), per_channel=0.5),
                second=iaa.LinearContrast((0.7, 1.3), per_channel=0.5))
        ]),
        # contrast
        iaa.SomeOf((0, 2), [
            iaa.GammaContrast((0.75, 1.25), per_channel=0.5),
            iaa.SigmoidContrast(gain=(0, 10), cutoff=(0.25, 0.75), per_channel=0.5),
            iaa.LogContrast(gain=(0.75, 1), per_channel=0.5),
            iaa.LinearContrast(alpha=(0.7, 1.3), per_channel=0.5)
        ]),
    ], random_order=True)
    '''
    seq = iaa.Sequential([
        # blur
        iaa.SomeOf((0, 2), [
            iaa.GaussianBlur((0.0, 2.0)),
            iaa.AverageBlur(k=(3, 7)),
            iaa.MedianBlur(k=(3, 7)),
            iaa.BilateralBlur(d=(1, 7)),
            iaa.MotionBlur(k=(3, 7))
        ]),
        # color
        iaa.SomeOf((0, 2), [
            # iaa.WithColorspace(),
            iaa.AddToHueAndSaturation((-25, 25)),
            # iaa.ChangeColorspace(to_colorspace[], alpha=0.5),
            iaa.Grayscale(alpha=(0.0, 0.3))
        ]),
        # brightness
        iaa.OneOf([
            iaa.Sequential([
                iaa.Add((-25, 25), per_channel=0.5),
                iaa.Multiply((0.5, 1.5), per_channel=0.5)
            ]),
            iaa.Add((-25, 25), per_channel=0.5),
            iaa.Multiply((0.5, 1.5), per_channel=0.5),
            iaa.FrequencyNoiseAlpha(
                exponent=(-4, 0),
                first=iaa.Multiply((0.5, 1.5), per_channel=0.5),
                second=iaa.LinearContrast((0.5, 1.5), per_channel=0.5))
        ]),
        # contrast
        iaa.SomeOf((0, 2), [
            iaa.GammaContrast((0.5, 1.5), per_channel=0.5),
            iaa.SigmoidContrast(gain=(0, 25), cutoff=(0.25, 0.75), per_channel=0.5),
            iaa.LogContrast(gain=(0.5, 1), per_channel=0.5),
            iaa.LinearContrast(alpha=(0.5, 1.5), per_channel=0.5)
        ]),
    ], random_order=True)
    image0 = seq.augment_image(image[0])
    image0 = cv2.warpAffine(
        image0,
        matrix[:2, :],
        dsize       = (image[0].shape[1], image[0].shape[0]),
        flags       = params.cvInterpolation(),
        borderMode  = params.cvBorderMode(),
        borderValue = params.cval,
    )

    # depth
    image1 = image[1]
    image1 = image1.astype('float32')
    '''
    blurK = np.random.choice([3, 5, 7], 1).astype(int)
    blurS = random.uniform(0.0, 1.5)

    image1 = cv2.resize(image1, None, fx=1 / 2, fy=1 / 2)
    res = (((image1 / 1000.0) * 1.41421356) ** 2)
    image1 = cv2.GaussianBlur(image1, (blurK, blurK), blurS, blurS)
    # quantify to depth resolution and apply gaussian
    dNonVar = np.divide(image1, res, out=np.zeros_like(image1), where=res != 0)
    dNonVar = np.round(dNonVar)
    dNonVar = np.multiply(dNonVar, res)
    noise = np.multiply(dNonVar, random.uniform(0.002, 0.004))  # empirically determined
    image1 = np.random.normal(loc=dNonVar, scale=noise, size=dNonVar.shape)
    image1 = cv2.resize(image1, (image[1].shape[1], image[1].shape[0]))

    # fast perlin noise
    seed = np.random.randint(2 ** 31)
    N_threads = 4
    perlin = fns.Noise(seed=seed, numWorkers=N_threads)
    drawFreq = random.uniform(0.05, 0.5)  # 0.05 - 0.2
    #drawFreq = 0.5
    perlin.frequency = drawFreq
    perlin.noiseType = fns.NoiseType.SimplexFractal
    perlin.fractal.fractalType = fns.FractalType.FBM
    drawOct = [2, 4, 8]
    freqOct = np.bincount(drawOct)
    rndOct = np.random.choice(np.arange(len(freqOct)), 1, p=freqOct / len(drawOct), replace=False)
    # rndOct = 8
    perlin.fractal.octaves = rndOct
    #perlin.fractal.lacunarity = 2.1
    perlin.fractal.lacunarity = random.uniform(1.0, 3.0)
    perlin.fractal.gain = random.uniform(0.25, 0.75)
    #perlin.fractal.gain = 0.45
    perlin.perturb.perturbType = fns.PerturbType.NoPerturb

    noiseX = np.random.uniform(0.001, 0.01, image[1].shape[1] * image[1].shape[0])  # 0.0001 - 0.1
    noiseY = np.random.uniform(0.001, 0.01, image[1].shape[1] * image[1].shape[0])  # 0.0001 - 0.1
    noiseZ = np.random.uniform(0.01, 0.1, image[1].shape[1] * image[1].shape[0])  # 0.01 - 0.1
    Wxy = np.random.randint(1, 8)  # 1 - 5
    Wz = np.random.uniform(0.0001, 0.007)  # 0.0001 - 0.004

    #noiseX = np.random.uniform(0.001, 0.05, image[1].shape[1] * image[1].shape[0])  # 0.0001 - 0.1
    #noiseY = np.random.uniform(0.001, 0.05, image[1].shape[1] * image[1].shape[0])  # 0.0001 - 0.1
    #noiseZ = np.random.uniform(0.01, 0.1, image[1].shape[1] * image[1].shape[0])  # 0.01 - 0.1
    #Wxy = np.random.randint(1, 7)  # 1 - 5
    #Wz = np.random.uniform(0.0001, 0.01)

    X, Y = np.meshgrid(np.arange(image[1].shape[1]), np.arange(image[1].shape[0]))
    coords0 = fns.empty_coords(image[1].shape[1] * image[1].shape[0])
    coords1 = fns.empty_coords(image[1].shape[1] * image[1].shape[0])
    coords2 = fns.empty_coords(image[1].shape[1] * image[1].shape[0])

    coords0[0, :] = noiseX.ravel()
    coords0[1, :] = Y.ravel()
    coords0[2, :] = X.ravel()
    VecF0 = perlin.genFromCoords(coords0)
    VecF0 = VecF0.reshape((image[1].shape[0], image[1].shape[1]))

    coords1[0, :] = noiseY.ravel()
    coords1[1, :] = Y.ravel()
    coords1[2, :] = X.ravel()
    VecF1 = perlin.genFromCoords(coords1)
    VecF1 = VecF1.reshape((image[1].shape[0], image[1].shape[1]))

    coords2[0, :] = noiseZ.ravel()
    coords2[1, :] = Y.ravel()
    coords2[2, :] = X.ravel()
    VecF2 = perlin.genFromCoords(coords2)
    VecF2 = VecF2.reshape((image[1].shape[0], image[1].shape[1]))

    x = np.arange(image[1].shape[1], dtype=np.uint16)
    x = x[np.newaxis, :].repeat(image[1].shape[0], axis=0)
    y = np.arange(image[1].shape[0], dtype=np.uint16)
    y = y[:, np.newaxis].repeat(image[1].shape[1], axis=1)

    Wxy_scaled = image1 * 0.001 * Wxy
    Wz_scaled = image1 * 0.001 * Wz
    # scale with depth
    fx = x + Wxy_scaled * VecF0
    fy = y + Wxy_scaled * VecF1
    fx = np.where(fx < 0, 0, fx)
    fx = np.where(fx >= image[1].shape[1], image[1].shape[1] - 1, fx)
    fy = np.where(fy < 0, 0, fy)
    fy = np.where(fy >= image[1].shape[0], image[1].shape[0] - 1, fy)
    fx = fx.astype(dtype=np.uint16)
    fy = fy.astype(dtype=np.uint16)
    image1 = image1[fy, fx] + Wz_scaled * VecF2
    '''
    image1 = np.where(image1 > 0, image1, 0.0)
    image1 = np.where(image1 > 2000.0, 0.0, image1)
    image1 = np.repeat(image1[:, :, np.newaxis], 3, axis=2)
    image1 = np.multiply(image1, 255.0/2000.0)
    #print(np.nanmax(image1), np.nanmin(image1))
    #image1 = get_normal(image1, cpara[0], cpara[1], cpara[2], cpara[3])
    image1 = cv2.warpAffine(
        image1,
        matrix[:2, :],
        dsize=(image[1].shape[1], image[1].shape[0]),
        flags=params.cvInterpolation(),
        borderMode=params.cvBorderMode(),
        borderValue=params.cval,
    )
    return [image0, image1]


def get_normal(depth_refine, fx=-1, fy=-1, cx=-1, cy=-1, for_vis=True):
    res_y = depth_refine.shape[0]
    res_x = depth_refine.shape[1]

    # inpainting
    #scaleOri = np.amax(depth_refine)

    #inPaiMa = np.where(depth_refine == 0.0, 255, 0)
    #inPaiMa = inPaiMa.astype(np.uint8)
    #inPaiDia = 5.0
    #depth_refine = depth_refine.astype(np.float32)
    #depPaint = cv2.inpaint(depth_refine, inPaiMa, inPaiDia, cv2.INPAINT_NS)

    #depNorm = depPaint - np.amin(depPaint)
    #rangeD = np.amax(depNorm)
    #depNorm = np.divide(depNorm, rangeD)
    #depth_refine = np.multiply(depNorm, scaleOri)

    #depth_imp = copy.deepcopy(depth_refine)

    centerX = cx
    centerY = cy

    constant = 1.0 / fx
    uv_table = np.zeros((res_y, res_x, 2), dtype=np.int16)
    column = np.arange(0, res_y)

    uv_table[:, :, 1] = np.arange(0, res_x) - centerX  # x-c_x (u)
    uv_table[:, :, 0] = column[:, np.newaxis] - centerY  # y-c_y (v)
    uv_table_sign = np.copy(uv_table)
    #uv_table = np.abs(uv_table)

    # kernel = np.ones((5, 5), np.uint8)
    # depth_refine = cv2.dilate(depth_refine, kernel, iterations=1)
    # depth_refine = cv2.medianBlur(depth_refine, 5 )
    depth_refine = ndimage.gaussian_filter(depth_refine, 2)  # sigma=3)
    # depth_refine = ndimage.uniform_filter(depth_refine, size=11)

    # very_blurred = ndimage.gaussian_filter(face, sigma=5)
    v_x = np.zeros((res_y, res_x, 3))
    v_y = np.zeros((res_y, res_x, 3))

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
    cross[depth_refine > 2000] = 0  # far range cut
    depth_refine[depth_refine > 2000] = 0

    scaDep = 1.0 / 2000.0
    #scaDep = 1.0 / np.nanmax(depth_refine)
    depth_refine = np.multiply(depth_refine, scaDep)
    cross[:, :, 0] = cross[:, :, 0] * (1 - (depth_refine))  # nearer has higher intensity
    cross[:, :, 1] = cross[:, :, 1] * (1 - (depth_refine))
    cross[:, :, 2] = cross[:, :, 2] * (1 - (depth_refine))
    scaCro = 255.0 / np.nanmax(cross)
    cross = np.multiply(cross, scaCro)
    cross = cross.astype(np.uint8)

    return cross


def apply_transform2mask(matrix, mask, params):

    mask = cv2.warpAffine(
        mask,
        matrix[:2, :],
        dsize=(mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return [mask]


def adjust_pose_annotation(matrix, pose, cpara):

    pose[2] = pose[2] / matrix[0, 0]
    pose[0] = pose[0] + ((matrix[0, 2] + ((cpara[2] * matrix[0, 0]) - cpara[2])) * pose[2]) / cpara[0]
    pose[1] = pose[1] + ((matrix[1, 2] + ((cpara[3] * matrix[0, 0]) - cpara[3])) * pose[2]) / cpara[1]

    return pose


def compute_resize_scale(image_shape, min_side=480, max_side=640):
    """ Compute an image scale such that the image size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resizing scale.
    """
    (rows, cols, _) = image_shape

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    return scale


def resize_image(img, min_side=480, max_side=640):
    """ Resize an image such that the size is constrained to min_side and max_side.

    Args
        min_side: The image's min side will be equal to min_side after resizing.
        max_side: If after resizing the image's max side is above max_side, resize until the max side is equal to max_side.

    Returns
        A resized image.
    """
    # compute scale to resize the image
    scale = compute_resize_scale(img.shape, min_side=min_side, max_side=max_side)

    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale
