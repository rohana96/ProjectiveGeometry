import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils import annotate_points
import os


def interactive_feature_matching(dims, img2, annopath):
    H, W = dims
    points1 = np.array([[0, 0], [W, 0], [W, H], [0, H]])
    points2 = np.array(annotate_points(imagepath=img2, save_annotation=True, annopath=annopath))
    matches = np.array([(i, i) for i in range(len(points1))])
    return points1, points2, matches


def computeH(p1, p2):
    """
    Compute the homography matrix from point correspondences.

    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
            equation
    """
    assert p1.shape[1] == p2.shape[1]
    assert p1.shape[0] == 2

    A = np.zeros(shape=(2 * p1.shape[1], 9))
    for i in range(0, p1.shape[1]):
        A[2 * i] = np.array([-p2[0, i], -p2[1, i], -1, 0, 0, 0, p2[0, i] * p1[0, i], p1[0, i] * p2[1, i], p1[0, i]])
        A[2 * i + 1] = np.array([0, 0, 0, -p2[0, i], -p2[1, i], -1, p2[0, i] * p1[1, i], p1[1, i] * p2[1, i], p1[1, i]])
    U, S, Vt = np.linalg.svd(A)
    H2to1 = Vt[-1, :].reshape(3, 3)
    return H2to1


def computeH_ransac(matches, locs1, locs2, num_iter=5000, tol=2):
    """
    Returns the best homography by computing the best set of matches using RANSAC.

    INPUTS
        matches - matrix specifying matches between these two sets of point locations
        locs1 and locs2 - matrices specifying point locations in each of the images
        num_iter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH2to1 - homography matrix with the most inliers found during RANSAC
        inliers - a vector of length N (len(matches)) with 1 at the those matches
                  that are part of the consensus set, and 0 elsewhere.
    """
    n_matches = len(matches)
    p1 = locs1[matches[:, 0], :]  # n_matches x 2
    p1 = np.transpose(p1)
    p2 = locs2[matches[:, 1], :]  # n_matches x 2
    p2 = np.transpose(p2)
    p2_homo = np.concatenate((p2, np.ones([1, p2.shape[1]])), axis=0)  # 3 x n_matches
    p1_homo = np.concatenate((p1, np.ones([1, p1.shape[1]])), axis=0)  # 3 x n_matches

    bestH2to1, max_inliers = None, 0
    fin_inliers = []

    for iter in range(num_iter):

        random_indices = np.random.choice(np.arange(n_matches), size=4, replace=False)
        H2to1 = computeH(p1[:, random_indices], p2[:, random_indices])

        pred1 = np.matmul(H2to1, p2_homo)
        pred1 = pred1 / pred1[-1, :]
        assert pred1.shape == p1_homo.shape

        error1 = np.linalg.norm(pred1 - p1_homo, axis=0)
        inliers = error1 < tol
        cnt_inliers = len(error1[error1 < tol])

        if cnt_inliers > max_inliers:
            bestH2to1, max_inliers = H2to1, cnt_inliers
            fin_inliers = inliers

    return bestH2to1, fin_inliers


def compositeH(H2to1, template, img):
    """
    Returns the composite image.

    INPUTS
        H - homography matrix [3x3]
        img - background image
        template - template image to be warped

    OUTPUTS
        composite_img - composite image
    """
    H, W, _ = img.shape
    warp_template = template
    H1to2 = np.linalg.inv(H2to1)
    warp_template = cv2.warpPerspective(warp_template, H1to2, (W, H))
    mask = cv2.cvtColor(warp_template, cv2.COLOR_BGR2GRAY)
    template_ind = np.nonzero(mask)
    img[template_ind] = warp_template[template_ind]
    composite_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return composite_img


def homography_interactive(
        imagepath1='data/homography/hp_cover.jpg',
        imagepath2='data/homography/cv_desk.jpg',
        outpath='out/homography',
        outname='billboard_warped.jpg'
):
    img1 = cv2.imread(imagepath1)
    img2 = cv2.imread(imagepath2)

    H, W, _ = img1.shape
    print("Mark correspondences in order:")
    print("top left --> top right --> bottom right --> bottom left")
    os.makedirs('annotations/homography', exist_ok=True)
    locs1, locs2, matches = interactive_feature_matching((H, W), imagepath2, annopath=os.path.join('annotations/homography', outname))

    print("Finding homography...")
    H2to1, _ = computeH_ransac(matches, locs1, locs2, num_iter=5000, tol=1)  # Hphoto_to_template
    print(H2to1)

    print("Warping new cover on base photo...")
    composite_img = compositeH(H2to1, img1, img2)

    os.makedirs(outpath, exist_ok=True)
    cv2.imwrite(os.path.join(outpath, outname), cv2.cvtColor(composite_img, cv2.COLOR_RGB2BGR))
    fig = plt.figure()
    plt.imshow(composite_img)
    plt.axis('off')
    plt.pause(3)
    plt.close()
