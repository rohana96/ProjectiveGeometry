# Projective Geometry and Homography

We will implement:
1. Simple image rectification algorithm given a single image. We leverage the mental model that humans have of the 3D world and how straight 
   lines in 3D project on to images. 
2. Homography estimation between two images via interactively marking point correspondences. We will show a simple AR application of such an 
   estimation.

## Affine Rectification 
The goal here is to generate affinely correct warps for images that are captured through perspective cameras (assuming pinhole camera 
model), with annotations of at least 2 pairs of parallel lines.

**Dataset Preparation:**

We find images that have sufficient perspective and planar surfaces with atleast 2 identifiable pairs of parallel lines. 

Here's an example:

 | Good Input  |                 Bad Input                 |
 |:-----------------------------------------:| :---------: |
 |  <img src="figures/good.jpg" height="200">  | <img src="figures/bad.jpg" height="200">  |

**Method:**

1. Annotate 2 pairs of identifiable parallel  (in the real 3D world) lines - $l1$ and $l2$. The equation of each line comes from the cross 
   product of two distinct points on it. 
2. The input image has a perspective transform and therefore the projection of real world pair of parallel lines intersect at a finite point. This 
   point can be written as $l_1 \times l_2$
   for each pair of lines. 
3. The points two points of intersection $p1$ and $p2$ lie on a line $l'_{\infty} = p_1 \times p_2$. This constitutes the projection of the line at 
   infinity $l
   {\infty} = [0, 0, 1]$. 
4. We can now define a homography $H$ between the original line at infinity and its projection by $l_{\infty} = H ^{-T} l'{\infty}$. 
5. Solving the above equation we arrive at the solution $H = [[1, 0, 0], [0, 1, 0], [l1, l2, l3]]$ where $l'_{\infty} = l1, l2, l3$.

**Command:**

Place you images under `data/rectification` as a `.jpg` images. 
```
python main.py --type rectification --method affine --imgs "list of image names"
```
**Results:**

|       Input Image        |  Annotated Parallel Lines on Input  |   Affine-Rectified Output   |
|:------------------------:|:-----------------------------------:|:---------------------------:|
| ![](data/rectification/tiles4.jpg)  | ![](annotations/affine/tiles4.jpg)  | ![](out/affine/tiles4.jpg)  |
| ![](data/rectification/tiles5.jpg)  | ![](annotations/affine/tiles5.jpg)  | ![](out/affine/tiles5.jpg)  |
| ![](data/rectification/tiles6.jpg)  | ![](annotations/affine/tiles6.jpg)  | ![](out/affine/tiles6.jpg)  |
| ![](data/rectification/london3.jpg) | ![](annotations/affine/london3.jpg) | ![](out/affine/london3.jpg) |
|   ![](data/rectification/net.jpg)   |   ![](annotations/affine/net.jpg)   |   ![](out/affine/net.jpg)   |

We evaluate angles (more precisely, the cosines) of 2 pairs of parallel lines before and after rectification. This will show how close to correct the final affine rectification is.

| Before rectification | After Rectification | Angles (Before) | Angles (After) |
|----------------------| -----------  |-----------------|----------------|
|![](annotations/affine/tiles4_test_lines.jpg)|![](out/affine/tiles4_test_lines.jpg)| 0.9805              | 0.9999         |
|![](annotations/affine/tiles4_test_lines.jpg)|![](out/affine/tiles4_test_lines.jpg)| 0.9999              | 1.0000         |
|![](annotations/affine/tiles5_test_lines.jpg)|![](out/affine/tiles5_test_lines.jpg)| 0.9999               | 0.9999         |
|![](annotations/affine/tiles5_test_lines.jpg)|![](out/affine/tiles5_test_lines.jpg)| 0.9836               | 1.0000         |
|![](annotations/affine/tiles6_test_lines.jpg)|![](out/affine/tiles6_test_lines.jpg)| 0.9995               | 1.0000         |
|![](annotations/affine/tiles6_test_lines.jpg)|![](out/affine/tiles6_test_lines.jpg)| 0.9997               | 1.0000         |
|![](annotations/affine/london3_test_lines.jpg)|![](out/affine/london3_test_lines.jpg)| 0.9990               | 1.0000        |
|![](annotations/affine/london3_test_lines.jpg)|![](out/affine/london3_test_lines.jpg)| 1.0000              | 1.0000        |
|![](annotations/affine/net_test_lines.jpg)|![](out/affine/net_test_lines.jpg)| 0.9997               | 1.0000       |
|![](annotations/affine/net_test_lines.jpg)|![](out/affine/net_test_lines.jpg)| 0.9997              |  1.0000        |

## Metric Rectification

The goal here is to generate metrically correct warps for images that are captured through perspective cameras (assume pinhole camera 
model), with annotations of at least 2 pairs of perpendicular lines and affine rectified image obtained in the previous section.

**Dataset Preparation:**

We find images that have sufficient perspective and planar surfaces with atleast 2 identifiable pairs of parallel lines and 2 pairs of 
perpendicular lines. 

**Method:**

1. Annotate 2 pairs of identifiable orthogonal  (in the real 3D world) lines - $l1$ and $l2$ on the affine-rectified image. The equation of each line 
   comes from the cross product of two distinct points on it.
2. Projection of orthogonal lines $l'$ and $m'$ are related by a conic $C^{*'}_{\infty}$ as $l' C^{*'}_{\infty} m' = 0$. For an affine rectified 
   image, this conic has 3 degrees of freedom and each pair affords us one constraint. (We annotate only 2 pairs of lines as we need estimation 
   only upto a scale)
3. We can now create a system of linear equations of the form $Ax = 0$. This can be solved using linear least squares for $c$ using SVD.
4. Using the relation $C^{'}_{\infty} = H C^{'}{\infty} H^T$ and $C^{*'}{\infty} = U \Sigma U^T$ (from SVD decomposition of $C^{*'}{\infty}$) and 
   we arrive at $H = \begin{bmatrix} \sqrt {\sigma_{1}^{-1}} & 0 & 0 \ 0 & \sqrt{\sigma_{2}^{-1}} & 0\ 0 & 0 & 1 \end{bmatrix} U^T$

**Command:**
Place you images under `data/rectification` as a `.jpg` images. 
```
python main.py --type rectification --method affine_to_metric  --imgs "list of image names"
```
**Results:**

| Input Image | Annotated perpendicular lines on input image | Annotated perpendicular lines on Affine-Rectified Image |               Metric-Rectified Image               |
|:------------------------:|:-----------------------------------:|:---------------------------:|:----------------------------------------------:|
| ![](data/rectification/tiles4.jpg)  | ![](annotations/metric/tiles4_affine_to_metric.jpg)  | ![](out/metric/tiles4_affine_rectified_anno.jpg)  |  ![](out/metric/tiles4_affine_to_metric.jpg)   |
| ![](data/rectification/tiles5.jpg)  | ![](annotations/metric/tiles5_affine_to_metric.jpg)  | ![](out/metric/tiles5_affine_rectified_anno.jpg)  |  ![](out/metric/tiles5_affine_to_metric.jpg)   |
| ![](data/rectification/tiles6.jpg)  | ![](annotations/metric/tiles6_affine_to_metric.jpg)  | ![](out/metric/tiles6_affine_rectified_anno.jpg)  |  ![](out/metric/tiles6_affine_to_metric.jpg)   |
| ![](data/rectification/london3.jpg) | ![](annotations/metric/london3_affine_to_metric.jpg) | ![](out/metric/london3_affine_rectified_anno.jpg) |  ![](out/metric/london3_affine_to_metric.jpg)  |
|  ![](data/rectification/net.jpg)   |  ![](annotations/metric/net_affine_to_metric.jpg)   |  ![](out/metric/net_affine_rectified_anno.jpg)   |    ![](out/metric/net_affine_to_metric.jpg)    |

We evaluate angles (more precisely, the cosines) of 2 pairs of perpendicular lines before and after rectification. This will show how close to 
correct the final affine rectification is.


| Before rectification                           | After Rectification | Angles (Before) | Angles (After) |
|------------------------------------------------| -----------  |-----------------|----------------|
| ![](annotations/metric/tiles4_test_lines.jpg)  |![](out/metric/tiles4_test_lines.jpg)| 0.3553          | -0.01785       |
| ![](annotations/metric/tiles4_test_lines.jpg)  |![](out/metric/tiles4_test_lines.jpg)| 0.3442          | -0.0462        |
| ![](annotations/metric/tiles5_test_lines.jpg)  |![](out/metric/tiles5_test_lines.jpg)| 0.0574          | -0.0414        |
| ![](annotations/metric/tiles5_test_lines.jpg)  |![](out/metric/tiles5_test_lines.jpg)| -0.2445         | -0.0502        |
| ![](annotations/metric/tiles6_test_lines.jpg)  |![](out/metric/tiles6_test_lines.jpg)|  0.0202        | -0.0245       |
| ![](annotations/metric/tiles6_test_lines.jpg)  |![](out/metric/tiles6_test_lines.jpg)| 0.0235         | -0.0197        |
| ![](annotations/metric/london3_test_lines.jpg) |![](out/metric/london3_test_lines.jpg)| 0.6254       | -0.0392       |
| ![](annotations/metric/london3_test_lines.jpg) |![](out/metric/london3_test_lines.jpg)| 0.6382          | -0.0284        |
| ![](annotations/metric/net_test_lines.jpg)     |![](out/metric/net_test_lines.jpg)| -0.1035         | -0.0039        |
| ![](annotations/metric/net_test_lines.jpg)     |![](out/metric/net_test_lines.jpg)| -0.1021        | -0.0036        |

## Planar Homography from Point Correspondences

Our goal here is to estimate the homography between two images using point correspondences. The point correspondences are marked interactively. 

**Method:**

1. Annotate 4 point correspondences $(x_i, x_i')$ between the normal image and the template image. We can define a homography $H$ such that 
  $x_i' = Hx_i$
2. $H$ is rank 8, and each point correspondence yields 2 constraints of the form:
$$\begin{bmatrix} \mathbf{0} & -w'\mathbf{x}^T & y'\mathbf{x}^T \ w'\mathbf{x}^T & \mathbf{0} & -x'\mathbf{x}^T \end{bmatrix}h = 0$$
3. This sets up a linear system of equatiosn $Ax=0$ and we can now solve this using SVD to find $x$ which constitutes $H$. 

**Command:**
Place you images and template image under `data/homography` as a `.jpg` images. 
```
python main.py --type homography --imgs "list of images" --template "template image name" --outname "output image name"
```
**Results:**

| Normal Image                       |            Perspective Image            | Annotated corners in Perspective Image |        Warped and Overlaid Image         |
|:-----------------------------------|:---------------------------------------:|:---------------------------------:|:----------------------------------------:|
| ![](data/homography/desk-normal.jpg) | ![](data/homography/desk-perspective.jpg) |     ![](annotations/homography/book_table.jpg)  | ![](out/homography/book_table.jpg)  |
| ![](data/homography/ghost.jpg)     |     ![](data/homography/mirror.jpg)     |     ![](annotations/homography/ghost_mirror.jpg)  | ![](out/homography/ghost_mirror.jpg)  |


## Direct Metric Rectification

We now metric-rectify the image directly from at least 5 pairs of perpendicular lines. More than 5 pairs can yield 
more accurate results.

**Dataset Preparation:**

We find images that have sufficient perspective and planar surfaces with atleast 5 identifiable pairs of perpendicular lines. 

**Method:**

1. Annotate 5 pairs of identifiable orthogonal  (in the real 3D world) lines - $l1$ and $l2$ on the affine-rectified image. The equation of each line 
   comes from the cross product of two distinct points on it.
2. Projection of orthogonal lines $l'$ and $m'$ are related by a conic $C^{*'}_{\infty}$ as $l' C^{*'}_{\infty} m' = 0$. This conic has 6 degrees 
   of freedom and each pair affords us one constraint. (We annotate only 5 pairs of lines as we need estimation 
   only upto a scale)
3. We can now create a system of linear equations of the form $Ax = 0$. This can be solved using linear least squares for $c$ using SVD.
4. Using the relation $C^{'}_{\infty} = H C^{'}{\infty} H^T$ and $C^{*'}{\infty} = U \Sigma U^T$ (from SVD decomposition of $C^{*'}{\infty}$) and 
   we arrive at $H = \begin{bmatrix} \sqrt {\sigma_{1}^{-1}} & 0 & 0 \ 0 & \sqrt{\sigma_{2}^{-1}} & 0\ 0 & 0 & 1 \end{bmatrix} U^T$


**Command:**
Place you images under `data/rectification` as a `.jpg` images. 
```
python main.py --type rectification --method direct_metric --imgs "list of image names"
```

**Results:**

| Input Image | Annotated perpendicular lines on input image  |               Metric-Rectified Image               |
|:------------------------:|:-----------------------------------:|:--------------------------:|
| ![](data/rectification/tiles4.jpg)  | ![](annotations/metric/tiles4_from_proj.jpg)  | ![](out/metric/tiles4_from_proj.jpg)   |
| ![](data/rectification/tiles5.jpg)  | ![](annotations/metric/tiles5_from_proj.jpg)  | ![](out/metric/tiles5_from_proj.jpg)   |
| ![](data/rectification/chess1.jpg)  | ![](annotations/metric/chess1_from_proj.jpg)  | ![](out/metric/chess1_from_proj.jpg)   |
| ![](data/rectification/london3.jpg) | ![](annotations/metric/london3_from_proj.jpg) | ![](out/metric/london3_from_proj.jpg)  |
|  ![](data/rectification/net.jpg)   |  ![](annotations/metric/net_from_proj.jpg)   |  ![](out/metric/net_from_proj.jpg)    |

We evaluate angles (more precisely, the cosines) of 2 pairs of perpendicular lines before and after rectification. This will show how close to 
correct the final affine rectification is.


| Before rectification                                    | After Rectification                    | Angles (Before) | Angles (After) |
|---------------------------------------------------------|----------------------------------------|-----------------|----------------|
| ![](annotations/metric/tiles4_from_proj_test_lines.jpg) | ![](out/metric/tiles4_from_proj_test_lines.jpg)  | -0.3650         | -0.0202      |
| ![](annotations/metric/tiles4_from_proj_test_lines.jpg)           | ![](out/metric/tiles4_from_proj_test_lines.jpg)  | 0.2158         | -0.0098       |
| ![](annotations/metric/tiles4_from_proj_test_lines.jpg)           | ![](out/metric/tiles4_from_proj_test_lines.jpg)  |  0.0404         | -0.0157        |
| ![](annotations/metric/tiles5_from_proj_test_lines.jpg)           | ![](out/metric/tiles5_from_proj_test_lines.jpg)  | -0.04988        |  0.0422     |
| ![](annotations/metric/tiles5_from_proj_test_lines.jpg)           | ![](out/metric/tiles5_from_proj_test_lines.jpg)  | -0.1605         | 0.0067        |
| ![](annotations/metric/tiles5_from_proj_test_lines.jpg)           | ![](out/metric/tiles5_from_proj_test_lines.jpg)  | 0.18149        | 0.0828       |
| ![](annotations/metric/chess1_from_proj_test_lines.jpg)           | ![](out/metric/chess1_from_proj_test_lines.jpg)  |  0.4844       | 0.0383       |
| ![](annotations/metric/chess1_from_proj_test_lines.jpg)           | ![](out/metric/chess1_from_proj_test_lines.jpg)  | -0.0658        | 0.2317     |
| ![](annotations/metric/chess1_from_proj_test_lines.jpg)           | ![](out/metric/chess1_from_proj_test_lines.jpg)  | -0.6411        | 0.0182        |
| ![](annotations/metric/london3_from_proj_test_lines.jpg)          | ![](out/metric/london3_from_proj_test_lines.jpg) | -0.0583     | 0.0146      |
| ![](annotations/metric/london3_from_proj_test_lines.jpg)          | ![](out/metric/london3_from_proj_test_lines.jpg) | 0.0282         | 0.0197       |
| ![](annotations/metric/london3_from_proj_test_lines.jpg)          | ![](out/metric/london3_from_proj_test_lines.jpg) | 0.0437        | -0.0112      |
| ![](annotations/metric/net_test_from_proj_lines.jpg)              | ![](out/metric/net_test_from_proj_lines.jpg)     | 0.1902      | 0.0891       |
| ![](annotations/metric/net_test_from_proj_lines.jpg)              | ![](out/metric/net_test_from_proj_lines.jpg)     | -0.0690       |  0.0428        |
| ![](annotations/metric/net_test_from_proj_lines.jpg)              | ![](out/metric/net_test_from_proj_lines.jpg)     | -0.3556       | 0.0032        |


## Homography with Multiple Images
Here we overlay 3 normal images on top of an image with perspective effect. I warp my own images on Times Square billboards.

**Method**
The method is same as described under Planar Homography from Point Correspondences section.

**Command:**
Place you images and template image under `data/homography` as a `.jpg` images. 
```
python main.py --type homography --imgs "list of images" --template "template image name" --outname "output image name"
```

**Results:**

|                                     Normal Image 1                                     |        Normal Image 2        |                  Normal Image 3                   |                Perspective Image                 | Annotated corners in Perspective Image |        Warped and Overlaid Image         |
|:--------------------------------------------------------------------------------------:|:----------------------------:|:-------------------------------------------------:|:------------------------------------------------:|:---------------------------------:|:----------------------------------------:|
|                              ![](data/homography/img1.jpg)                              | ![](data/homography/img2.jpg) |        ![](data/homography/img3.jpg)         |          ![](figures/TimesSquare.png)            | ![](annotations/homography/billboard_warped1.jpg) | ![](out/homography/billboard_warped3.jpg) |

