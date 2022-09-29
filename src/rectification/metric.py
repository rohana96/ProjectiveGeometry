from utils import annotate_points, get_line_from_points, MyWarp, transform_line, transform_points
import numpy as np
import cv2
import os
from .affine import affine_rect


def metric_rect_from_affine_rectified(
        datapath='data/rectification',
        imagename='tiles5',
        outpath='out/metric',
        cachedir='cache/metric',
        annodir='annotations/metric',
        load_from_cache=True,
        force_annotate=False,
        compute_angles=False
):
    imagepath = os.path.join(datapath, f'{imagename}.jpg')
    cachepath = os.path.join(cachedir, f'{imagename}_affine_to_metric.npy')
    annopath = os.path.join(annodir, f'{imagename}_affine_to_metric.jpg')
    img = cv2.imread(imagepath)

    H_affine = affine_rect(force_annotate=False, imagename=imagename, compute_angles=compute_angles)

    if load_from_cache and os.path.exists(cachepath) and not force_annotate:
        points = np.load(cachepath)
    else:
        os.makedirs(annodir, exist_ok=True)
        points = np.array(annotate_points(imagepath, annopath=annopath))
        os.makedirs(cachedir, exist_ok=True)
        np.save(cachepath, points)

    img_annotated = cv2.imread(annopath)
    img_annotated_rect = MyWarp(img_annotated, H_affine)
    os.makedirs(outpath, exist_ok=True)
    outimagepath = os.path.join(outpath, imagename + '_affine_rectified_anno.jpg')

    cv2.imwrite(outimagepath, img_annotated_rect)

    points_affine_rect = []
    for point in points:
        pt = transform_points(H_affine, point)
        points_affine_rect.append([pt[0]/pt[2], pt[1]/pt[2]])

    points = np.array(points_affine_rect)

    lines = []
    for i in range(0, len(points), 2):
        line = get_line_from_points(points[i], points[i + 1])
        line = line / line[-1]
        lines.append(line)

    l11, l12, l13 = lines[0]
    m11, m12, m13 = lines[1]
    l21, l22, l23 = lines[2]
    m21, m22, m23 = lines[3]

    A = [
        [l11 * m11, l11 * m12 + l12 * m11, l12 * m12],
        [l21 * m21, l21 * m22 + l22 * m21, l22 * m22]
    ]

    U, S, Vt = np.linalg.svd(A)
    s = Vt[-1] / Vt[-1][-1]
    C = [
        [s[0], s[1], 0],
        [s[1], s[2], 0],
        [0, 0, 0]
    ]

    u, S, _ = np.linalg.svd(C)
    S_inv = np.array([
        [1. / np.sqrt(S[0]), 0., 0.],
        [0., 1. / np.sqrt(S[1]), 0.],
        [0., 0., 1.],
    ])

    H = S_inv.dot(u.T)

    img = cv2.imread(os.path.join('out/affine', imagename + '.jpg'))
    img_rect = MyWarp(img, H)
    os.makedirs(outpath, exist_ok=True)
    outimagepath = os.path.join(outpath, f'{imagename}_affine_to_metric.jpg')
    cv2.imwrite(outimagepath, img_rect)

    img_annotated = cv2.imread(os.path.join(outpath, imagename + '_affine_rectified_anno.jpg'))
    img_rect_anno = MyWarp(img_annotated, H)
    outimagepath = os.path.join(outpath, f'{imagename}_affine_to_metric_anno.jpg')
    cv2.imwrite(outimagepath, img_rect_anno)

    if compute_angles:
        print("Metric Rectification: Angles")
        transform_line(H, img=f'out/affine/{imagename}.jpg', annopath=annodir, outpath=outpath, imagename=imagename)
    return H


def metric_rect_from_proj(
        datapath='data/rectification',
        imagename='checker3',
        outpath='out/metric',
        cachedir='cache/metric',
        annodir='annotations/metric',
        load_from_cache=True,
        force_annotate=False,
        compute_angles=False
):
    imagepath = os.path.join(datapath, f'{imagename}.jpg')
    cachepath = os.path.join(cachedir, f'{imagename}_from_proj.npy')
    annopath = os.path.join(annodir, f'{imagename}_from_proj.jpg')
    img = cv2.imread(imagepath)

    if load_from_cache and os.path.exists(cachepath) and not force_annotate:
        points = np.load(cachepath)
    else:
        os.makedirs(annodir, exist_ok=True)
        points = np.array(annotate_points(imagepath, annopath=annopath))

        os.makedirs(cachedir, exist_ok=True)
        np.save(cachepath, points)

    lines = []
    for i in range(0, len(points), 2):
        line = get_line_from_points(points[i], points[i + 1])
        line = line / line[-1]
        lines.append(line)

    A = []
    for i in range(0, len(lines), 2):
        l1, l2, l3 = lines[i]
        m1, m2, m3 = lines[i + 1]
        row = [
            l1 * m1,
            (l1 * m2 + l2 * m1) / 2,
            l2 * m2,
            (l1 * m3 + l3 * m1) / 2,
            (l2 * m3 + l3 * m2) / 2,
            l3 * m3
        ]
        A.append(row)

    A = np.array(A)
    U, S, Vt = np.linalg.svd(A)
    c = Vt[-1] / Vt[-1][-1]
    C = [[c[0], c[1] / 2, c[3] / 2],
         [c[1] / 2, c[2], c[4] / 2],
         [c[3] / 2, c[4] / 2, c[5]]]

    U, S, Vt = np.linalg.svd(C)
    S_inv = np.array([
        [1. / np.sqrt(S[0]), 0., 0.],
        [0., 1. / np.sqrt(S[1]), 0.],
        [0., 0., 1.]
    ])

    H = np.dot(S_inv, U.T)
    img_rect = MyWarp(img, H)

    os.makedirs(outpath, exist_ok=True)
    outimagepath = os.path.join(outpath, f'{imagename}_from_proj.jpg')
    cv2.imwrite(outimagepath, img_rect)

    if compute_angles:
        print("Metric Rectification: Angles")
        transform_line(H, imagepath, annopath=annodir, outpath=outpath, imagename=imagename)
    return
