from utils import annotate_points, get_line_from_points, MyWarp, transform_line, transform_points
import numpy as np
import cv2
import os


def affine_rect(
        datapath='data/rectification',
        imagename='tiles5',
        outpath='out/affine',
        cachedir='cache/affine',
        annodir='annotations/affine',
        load_from_cache=True,
        force_annotate=False,
        compute_angles=False
):
    imagepath = os.path.join(datapath, f'{imagename}.jpg')
    cachepath = os.path.join(cachedir, f'{imagename}.npy')
    annopath = os.path.join(annodir, f'{imagename}.jpg')

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
        lines.append(line)

    p1_intersection = np.cross(lines[0], lines[1])
    p1_intersection = p1_intersection / p1_intersection[-1]
    p2_intersection = np.cross(lines[2], lines[3])
    p2_intersection = p2_intersection / p2_intersection[-1]

    line_infinity = np.cross(p1_intersection, p2_intersection)
    line_infinity = line_infinity / line_infinity[-1]

    H = np.array([[1, 0, 0], [0, 1, 0], [line_infinity[0], line_infinity[1], 1]])

    img_rect = MyWarp(img, H)
    os.makedirs(outpath, exist_ok=True)
    outimagepath = os.path.join(outpath, imagename + '.jpg')
    cv2.imwrite(outimagepath, img_rect)

    img_annotated = cv2.imread(annopath)
    img_rect_anno = MyWarp(img_annotated, H)
    outimagepath = os.path.join(outpath, f'{imagename}_anno.jpg')
    cv2.imwrite(outimagepath, img_rect_anno)

    if compute_angles:
        print("Affine Rectification: Angles")
        transform_line(H, imagepath, annopath=annodir, outpath=outpath, imagename=imagename)

    return H
