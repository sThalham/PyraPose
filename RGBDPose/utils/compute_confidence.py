# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Sergey Karayev
# --------------------------------------------------------

import numpy as np


def compute_overlap(an, cp):
    """
    Args
        a: (N, 4) ndarray of float
        b: (K, 16) ndarray of float
        c: (K,) ndarray of uint8

    Returns
        overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.shape[0]
    K = control_points.shape[0]
    overlaps = np.zeros((N, K), dtype=np.float64)

    for k in range(K):
        points = cp.shape[1] * 0.5
        inliers = 0.0
        for n in range(N):
            for p in range(0, int(points)):
                if cp[k, p] > bo[n, 0] and cp[k, p] < bo[n, 2] and cp[k, p+1] > bo[n, 1] and cp[k, p+1] < bo[n, 3]:
                    inliers += 1.0

            overlaps[n, k] = points/inliers
            print('ov: ', n, k, overlaps[n, k])
    return overlaps

