import cv2
import numpy as np
from scipy import ndimage, signal
import copy
import random
import pyfastnoisesimd as fns
import imgaug.augmenters as iaa


def augmentDepth(depth, obj_mask, mask_ori):

    resY, resX = depth.shape
    drawKern = [3, 5, 7]
    freqKern = np.bincount(drawKern)
    kShadow = np.random.choice(np.arange(len(freqKern)), 1, p=freqKern / len(drawKern), replace=False)
    kShadow.astype(int)
    shadowClK = kShadow[0]
    kMed = np.random.choice(np.arange(len(freqKern)), 1, p=freqKern / len(drawKern), replace=False)
    kMed.astype(int)
    shadowMK = kMed[0]
    kBlur = np.random.choice(np.arange(len(freqKern)), 1, p=freqKern / len(drawKern), replace=False)
    kBlur.astype(int)
    blurK = kBlur[0]
    blurS = random.uniform(0.0, 1.5)

    # mask rendering failed for first rendered dataset
    # erode and blur mask to get more realistic appearance
    #partmask = mask_ori
    #partmask = partmask.astype(np.float32)
    #mask = partmask > (np.median(partmask) * 0.4)
    partmask = np.where(mask_ori > 1, 255.0, 0.0)

    #aug_dep = partmask.astype(np.uint8)
    #cv2.imwrite('/home/stefan/mask.png', aug_dep)

    # apply shadow
    kernel = np.ones((shadowClK, shadowClK))
    partmask = cv2.morphologyEx(partmask, cv2.MORPH_OPEN, kernel)
    partmask = signal.medfilt2d(partmask, kernel_size=shadowMK)
    partmask = partmask.astype(np.uint8)
    mask = partmask > 20
    depth = np.where(mask, depth, 0.0)

    depthFinal = cv2.resize(depth, None, fx=1 / 2, fy=1 / 2)
    res = (((depthFinal / 1000.0) * 1.41421356) ** 2)
    depthFinal = cv2.GaussianBlur(depthFinal, (blurK, blurK), blurS, blurS)
    # quantify to depth resolution and apply gaussian
    dNonVar = np.divide(depthFinal, res, out=np.zeros_like(depthFinal), where=res != 0)
    dNonVar = np.round(dNonVar)
    dNonVar = np.multiply(dNonVar, res)
    noise = np.multiply(dNonVar, random.uniform(0.002, 0.004)) # empirically determined
    depthFinal = np.random.normal(loc=dNonVar, scale=noise, size=dNonVar.shape)
    depth = cv2.resize(depthFinal, (resX, resY))

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

    noiseX = np.random.uniform(0.001, 0.01, resX * resY) # 0.0001 - 0.1
    noiseY = np.random.uniform(0.001, 0.01, resX * resY) # 0.0001 - 0.1
    noiseZ = np.random.uniform(0.01, 0.1, resX * resY) # 0.01 - 0.1
    Wxy = np.random.randint(1, 5) # 1 - 5
    Wz = np.random.uniform(0.0001, 0.004) #0.0001 - 0.004

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

    # vanilla
    # fx = x + Wxy * VecF0
    # fy = y + Wxy * VecF1
    # fx = np.where(fx < 0, 0, fx)
    # fx = np.where(fx >= resX, resX - 1, fx)
    # fy = np.where(fy < 0, 0, fy)
    # fy = np.where(fy >= resY, resY - 1, fy)
    # fx = fx.astype(dtype=np.uint16)
    # fy = fy.astype(dtype=np.uint16)
    # Dis = depth[fy, fx] + Wz * VecF2
    # depth = np.where(Dis > 0, Dis, 0.0)

    x = np.arange(resX, dtype=np.uint16)
    x = x[np.newaxis, :].repeat(resY, axis=0)
    y = np.arange(resY, dtype=np.uint16)
    y = y[:, np.newaxis].repeat(resX, axis=1)

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


def augmentRGB_DEPRECATED(rgb):
    # alpha == simple contrast control
    alpha = 0.5
    # beta == simple brightness control
    beta = 25
    # gamma == color perturbation in percent/100
    gamma = 0.05

    gain_ill = 100.0

    new_rgb = copy.deepcopy(rgb)

    draw = np.random.randint(0, 1)
    if draw == 0:
        # brightness and contrast
        alpha_r = np.random.uniform(1.0 - alpha, 1.0 + alpha)
        beta_r = np.random.randint(-beta, beta)
        alpha_g = np.random.uniform(1.0 - alpha, 1.0 + alpha)
        beta_g = np.random.randint(-beta, beta)
        alpha_b = np.random.uniform(1.0 - alpha, 1.0 + alpha)
        beta_b = np.random.randint(-beta, beta)
        new_rgb[:, :, 0] = np.clip(alpha_r * new_rgb[:, :, 0] + beta_r, 0, 255)
        new_rgb[:, :, 1] = np.clip(alpha_g * new_rgb[:, :, 1] + beta_g, 0, 255)
        new_rgb[:, :, 2] = np.clip(alpha_b * new_rgb[:, :, 2] + beta_b, 0, 255)

        # color
        mean_r = np.mean(new_rgb[:, :, 0])
        mean_g = np.mean(new_rgb[:, :, 1])
        mean_b = np.mean(new_rgb[:, :, 2])
        per_r = np.random.normal(0.0, mean_r * gamma)
        per_g = np.random.normal(0.0, mean_g * gamma)
        per_b = np.random.normal(0.0, mean_b * gamma)
        new_rgb[:, :, 0] = np.clip(new_rgb[:, :, 0] + per_r, 0, 255)
        new_rgb[:, :, 1] = np.clip(new_rgb[:, :, 1] + per_g, 0, 255)
        new_rgb[:, :, 2] = np.clip(new_rgb[:, :, 2] + per_b, 0, 255)

    else:
        # color
        mean_r = np.mean(new_rgb[:, :, 0])
        mean_g = np.mean(new_rgb[:, :, 1])
        mean_b = np.mean(new_rgb[:, :, 2])
        per_r = np.random.normal(0.0, mean_r * gamma)
        per_g = np.random.normal(0.0, mean_g * gamma)
        per_b = np.random.normal(0.0, mean_b * gamma)
        new_rgb[:, :, 0] = np.clip(new_rgb[:, :, 0] + per_r, 0, 255)
        new_rgb[:, :, 1] = np.clip(new_rgb[:, :, 1] + per_g, 0, 255)
        new_rgb[:, :, 2] = np.clip(new_rgb[:, :, 2] + per_b, 0, 255)

        # brightness and contrast
        alpha_r = np.random.uniform(1.0 - alpha, 1.0 + alpha)
        beta_r = np.random.randint(-beta, beta)
        alpha_g = np.random.uniform(1.0 - alpha, 1.0 + alpha)
        beta_g = np.random.randint(-beta, beta)
        alpha_b = np.random.uniform(1.0 - alpha, 1.0 + alpha)
        beta_b = np.random.randint(-beta, beta)
        new_rgb[:, :, 0] = np.clip(alpha_r * new_rgb[:, :, 0] + beta_r, 0, 255)
        new_rgb[:, :, 1] = np.clip(alpha_g * new_rgb[:, :, 1] + beta_g, 0, 255)
        new_rgb[:, :, 2] = np.clip(alpha_b * new_rgb[:, :, 2] + beta_b, 0, 255)

    # illumination
    orig_img = new_rgb.astype(float).copy()
    img = new_rgb / 255.0  # rescale to 0 to 1 range
    img_rs = img.reshape(-1, 3)
    img_centered = img_rs - np.mean(img_rs, axis=0)

    img_cov = np.cov(img_centered, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(img_cov)
    sort_perm = eig_vals[::-1].argsort()
    eig_vals[::-1].sort()
    eig_vecs = eig_vecs[:, sort_perm]

    # get [p1, p2, p3]
    m1 = np.column_stack((eig_vecs))

    # get 3x1 matrix of eigen values multiplied by random variable draw from normal
    # distribution with mean of 0 and standard deviation of 0.1
    m2 = np.zeros((3, 1))
    alpha = np.random.normal(0, gain_ill)
    m2[:, 0] = alpha * eig_vals[:]
    add_vect = np.matrix(m1) * np.matrix(m2)

    for idx in range(3):  # RGB
        orig_img[..., idx] += add_vect[idx]

    orig_img = np.clip(orig_img, 0.0, 255.0)
    new_rgb = orig_img.astype(np.uint8)

    # blur
    drawKernx = [3, 5, 7]
    freqKernx = np.bincount(drawKernx)
    drawKerny = [3, 5, 7]
    freqKerny = np.bincount(drawKerny)
    kBlurx = np.random.choice(np.arange(len(freqKernx)), 1, p=freqKernx / len(drawKernx), replace=False)
    sBlurx = random.uniform(0.00, 2.0)
    kBlury = np.random.choice(np.arange(len(freqKerny)), 1, p=freqKerny / len(drawKerny), replace=False)
    sBlury = random.uniform(0.00, 2.0)

    new_rgb = cv2.GaussianBlur(new_rgb, (kBlurx, kBlury), sBlurx, sBlury)

    #INVERSE MIP MAPPING

    return new_rgb


def augmentAAEext(img):

    seq = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.GaussianBlur(1.5)),
        iaa.Sometimes(0.5, iaa.Add((-25, 25), per_channel=0.3)),
        iaa.Sometimes(0.5, iaa.Multiply((0.6, 1.4), per_channel=0.5)),
        iaa.Sometimes(0.5, iaa.ContrastNormalization((0.4, 2.3), per_channel=0.3)),
    ], random_order=True)

    return seq.augment_image(img)


def augmentRGB(img):

    seq = iaa.Sequential([
        # blur
        iaa.SomeOf((0, 2), [
            iaa.GaussianBlur((0.0, 2.0)),
            iaa.AverageBlur(k=(3, 7)),
            iaa.MedianBlur(k=(3, 7)),
            iaa.BilateralBlur(d=(1, 7)),
            iaa.MotionBlur(k=(3, 7))
        ]),
        #color
        iaa.SomeOf((0, 2), [
           #iaa.WithColorspace(),
            iaa.AddToHueAndSaturation((-15, 15)),
           #iaa.ChangeColorspace(to_colorspace[], alpha=0.5),
            iaa.Grayscale(alpha=(0.0, 0.2))
        ]),
        #brightness
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
                second=iaa.ContrastNormalization((0.7, 1.3), per_channel=0.5))
            ]),
        #contrast
        iaa.SomeOf((0, 2), [
            iaa.GammaContrast((0.75, 1.25), per_channel=0.5),
            iaa.SigmoidContrast(gain=(0, 10), cutoff=(0.25, 0.75), per_channel=0.5),
            iaa.LogContrast(gain=(0.75, 1), per_channel=0.5),
            iaa.LinearContrast(alpha=(0.7, 1.3), per_channel=0.5)
        ]),
        #arithmetic
        iaa.SomeOf((0, 3), [
            iaa.AdditiveGaussianNoise(scale=(0, 0.05), per_channel=0.5),
            iaa.AdditiveLaplaceNoise(scale=(0, 0.05), per_channel=0.5),
            iaa.AdditivePoissonNoise(lam=(0, 8), per_channel=0.5),
            iaa.Dropout(p=(0, 0.05), per_channel=0.5),
            iaa.ImpulseNoise(p=(0, 0.05)),
            iaa.SaltAndPepper(p=(0, 0.05)),
            iaa.Salt(p=(0, 0.05)),
            iaa.Pepper(p=(0, 0.05))
        ]),
        #iaa.Sometimes(p=0.5, iaa.JpegCompression((0, 30)), None),
        ], random_order=True)
    return seq.augment_image(img)


def augmentRGB_V2(img):

    seq = iaa.Sequential([
        # blur
        iaa.SomeOf((1, 2), [
            iaa.Sometimes(0.5, iaa.GaussianBlur(1.5)),
            iaa.Sometimes(0.25, iaa.AverageBlur(k=(3, 7))),
            iaa.Sometimes(0.25, iaa.MedianBlur(k=(3, 7))),
            iaa.Sometimes(0.25, iaa.BilateralBlur(d=(1, 7))),
            iaa.Sometimes(0.25, iaa.MotionBlur(k=(3, 7))),
        ]),

        iaa.Sometimes(0.25, iaa.Add((-25, 25), per_channel=0.3)),
        iaa.Sometimes(0.25, iaa.Multiply((0.6, 1.4), per_channel=0.5)),
        iaa.Sometimes(0.25, iaa.ContrastNormalization((0.4, 2.3), per_channel=0.3)),

        #iaa.Sometimes(0.25, iaa.AddToHueAndSaturation((-15, 15))),
        #iaa.Sometimes(0.25, iaa.Grayscale(alpha=(0.0, 0.2))),
        iaa.Sometimes(0.25,
            iaa.FrequencyNoiseAlpha(
                exponent=(-4, 0),
                first=iaa.Add((-25, 25), per_channel=0.3),
                second=iaa.Multiply((0.6, 1.4), per_channel=0.3)
            )
        ), ], random_order=True)
    return seq.augment_image(img)


def pasteCOCO_BG(img, cocos, mask):
    bg = random.sample(cocos, 1)
    bg = cv2.imread('/home/stefan/data/coco_test2017/' + bg[0], 1)
    bg = cv2.resize(bg, (640, 480))
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    rep_img = np.where(mask > 0, img, bg)
    return rep_img


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
    #cross[depth_refine > depthCut] = 0  # far range cut
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