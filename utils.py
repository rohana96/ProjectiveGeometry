import numpy as np
import cv2
import os


def normalize(v):
    return v / np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def MyWarp(img, H):
    h, w = img.shape[:2]
    pts = np.array([[0, 0], [0, h], [w, h], [w, 0]], dtype=np.float64).reshape(-1, 1, 2)
    pts = cv2.perspectiveTransform(pts, H)
    [xmin, ymin] = (pts.min(axis=0).ravel() - 0.5).astype(int)
    [xmax, ymax] = (pts.max(axis=0).ravel() + 0.5).astype(int)
    t = [-xmin, -ymin]
    Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])

    result = cv2.warpPerspective(img, Ht.dot(H), (xmax - xmin, ymax - ymin))
    return result


def cosine(u, v):
    return (u[0] * v[0] + u[1] * v[1]) / (np.sqrt(u[0] ** 2 + u[1] ** 2) * np.sqrt(v[0] ** 2 + v[1] ** 2))


def annotate_lines(
        imagepath='sample.jpg',
        pointA=(0, 0),
        pointB=(50, 50)
):
    img = cv2.imread(imagepath)
    cv2.imshow('Original Image', img)
    cv2.waitKey(0)
    # Draw line on image
    imageLine = img.copy()
    cv2.line(imageLine, pointA, pointB, (255, 255, 0), thickness=3, lineType=cv2.LINE_AA)
    cv2.imshow('Image Line', imageLine)
    cv2.imwrite('test.jpg', imageLine)
    cv2.waitKey(0)


annotate_line = False
change_color = True
color = (0, 255, 0)


def annotate_points(
        imagepath='data/rectification/book1.jpg',
        save_annotation=True,
        annopath='.'
):
    points = []

    def click_event(event, x, y, flags, params):
        global annotate_line
        global change_color
        global color

        if event == cv2.EVENT_LBUTTONDOWN:
            # print(x, ' ', y)
            points.append([x, y])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
            # cv2.putText(img, str(x) + ',' +
            #             str(y), (x, y), font,
            #             1, (255, 0, 0), 2)
            cv2.imshow('image', img)

            if annotate_line:
                # if change_color:
                #     color = (x / W * 255, y / H * 255, (x + y) / (H + W) * 255)

                cv2.line(img, points[-1], points[-2], color, thickness=3, lineType=cv2.LINE_AA)
                # change_color = 1 ^ change_color
                cv2.imshow('image', img)

            annotate_line = 1 ^ annotate_line

    img = cv2.imread(imagepath)
    H, W, _ = img.shape
    print(H, W)
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    cv2.waitKey(0)
    if save_annotation is True:
        cv2.imwrite(annopath, img)

    cv2.destroyAllWindows()
    return points


def get_line_from_points(p1, p2):
    """
    :param p1: (x1, y1) coordinates in euclidean space
    :param p2: (x2, y2) coordinates in euclidean space
    :return: line in projective space l = (x1, y1, 1) x (x2, y2, 1)
    """
    p1_proj = np.array([p1[0], p1[1], 1])
    p2_proj = np.array([p2[0], p2[1], 1])
    return np.cross(p1_proj, p2_proj)


def transform_line(H, img, annopath, outpath, imagename):
    print(f"Image: {imagename}")
    print("---------------------")
    annopath = os.path.join(annopath, f'{imagename}_test_lines.jpg')
    outpath = os.path.join(outpath, f'{imagename}_test_lines.jpg')

    points = np.array(annotate_points(img, save_annotation=True, annopath=annopath))
    img_annotated = cv2.imread(annopath)
    img_rect_anno = MyWarp(img_annotated, H)
    cv2.imwrite(outpath, img_rect_anno)

    lines = []
    for i in range(0, len(points), 2):
        line = get_line_from_points(points[i], points[i + 1])
        lines.append(line)

    for i in range(0, len(lines), 2):
        line1 = lines[i]
        line1 = line1 / line1[-1]
        line2 = lines[i + 1]
        line2 = line2 / line2[-1]
        print("Original angles: ", cosine(line1, line2))
        line1_new = np.dot(np.linalg.inv(H).T, line1)
        line2_new = np.dot(np.linalg.inv(H).T, line2)
        print("Rectification angles: ", cosine(line1_new, line2_new))
        print("\n")


def transform_points(H, pt):
    pt_proj = np.array([pt[0], pt[1], 1])
    return H.dot(pt_proj)
