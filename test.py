from utils import annotate_lines, annotate_points, transform_line
from src.rectification import affine_rect, metric_rect_from_affine_rectified, metric_rect_from_proj
from src.homography import sift_feature_matching, homography_interactive, homography_kpdetection
import cv2



# ======== test annotation
# imagepath = 'data/rectification/book1.jpg'
# print(annotate_points(imagepath))
# annotate_lines(imagepath, pointA=(500, 0), pointB=(500, 500))

# ======== test rectification
# affine_rect(force_annotate=True, compute_angles=True)
for imagename in ['tiles4', 'tiles5', 'chess1', 'net', 'london3']:
    # metric_rect_from_affine_rectified(force_annotate=False, imagename=imagename, compute_angles=True)
    metric_rect_from_proj(imagename=imagename, compute_angles=True)

# ======== test sift
# img1 = 'data/homography/cv_cover.jpg'
# img2 = 'data/homography/cv_desk.jpg'
# img1 = cv2.imread(img1)
# img2 = cv2.imread(img2)
# locs1, locs2, matches = sift_feature_matching(img1, img2)


# # ======= test homography via keypoint detection
# img1 = 'data/homography/hp_cover.jpg'
# img2 = 'data/homography/cv_desk.jpg'
# img3 = 'data/homography/cv_cover.jpg'
# homography_kpdetection(imagepath1=img1, imagepath2=img2, imagepath3=img3)


# # ======= test homography via interactive feature matching
# img1 = 'data/homography/hp_cover.jpg'
# img1 = 'data/homography/desk-normal.png'
# img2 = 'data/homography/cv_desk.jpg'
# img2 = 'data/homography/desk-perspective.png'

# img1 = 'data/homography/ghost.jpg'
# img2 = 'data/homography/mirror.jpg'
# homography_interactive(imagepath1=img1, imagepath2=img2, outname='ghost_mirror.jpg')

# img1 = 'data/homography/desk-normal.png'
# img2 = 'data/homography/desk-perspective.png'
# homography_interactive(imagepath1=img1, imagepath2=img2, outname='book_table.jpg')
#
# # ======= BONUS: test homography via interactive feature matching
# img1 = 'data/billboard/img1.jpg'
# img2 = 'data/billboard/base.png'
# homography_interactive(imagepath1=img1, imagepath2=img2, outname='billboard_warped1.jpg')
#
# img1 = 'data/billboard/img2.jpg'
# img2 = 'out/homography/billboard_warped1.jpg'
# homography_interactive(imagepath1=img1, imagepath2=img2, outname='billboard_warped2.jpg')
#
# img1 = 'data/billboard/img3.jpg'
# img2 = 'out/homography/billboard_warped2.jpg'
# homography_interactive(imagepath1=img1, imagepath2=img2, outname='billboard_warped3.jpg')

