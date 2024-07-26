# Code from Github - Visualization
import numpy as np
from scipy.spatial import ConvexHull
from AB3DMOT_libs.box import Box3D


def polygon_clip(subjectPolygon, clipPolygon):
    """ Clip a polygon with another polygon.
    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

    Args:
        subjectPolygon: a list of (x,y) 2d points, any polygon.
        clipPolygon: a list of (x,y) 2d points, has to be *convex*
    Note:
        **points have to be counter-clockwise ordered**

    Return:
        a list of (x,y) vertex point for the intersection polygon.
    """
    def inside(p):
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) > (cp2[1] - cp1[1]) * (p[0] - cp1[0])
 
    def computeIntersection():
        dc = [cp1[0] - cp2[0], cp1[1] - cp2[1]]
        dp = [s[0] - e[0], s[1] - e[1]]
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0] 
        n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
        return [(n1 * dp[0] - n2 * dc[0]) * n3, (n1 * dp[1] - n2 * dc[1]) * n3]
 
    outputList = subjectPolygon
    cp1 = clipPolygon[-1]
 
    for clipVertex in clipPolygon:
        cp2 = clipVertex
        inputList = outputList
        outputList = []
        s = inputList[-1]
 
        for subjectVertex in inputList:
            e = subjectVertex
            if inside(e):
                if not inside(s): outputList.append(computeIntersection())
                outputList.append(e)
            elif inside(s): outputList.append(computeIntersection())
            s = e
        cp1 = cp2
        if len(outputList) == 0: return None
    return (outputList)

def convex_hull_intersection(p1, p2):
    """ Compute area of two convex hull's intersection area.
        p1,p2 are a list of (x,y) tuples of hull vertices.
        return a list of (x,y) for the intersection and its volume
    """
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0  

def PolygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area

def compute_inter_2D(boxa_bottom, boxb_bottom):
    # computer intersection over union of two sets of bottom corner points


    _, I_2D = convex_hull_intersection(boxa_bottom, boxb_bottom)

    # a slower version
    # from shapely.geometry import Polygon
    # reca, recb = Polygon(boxa_bottom), Polygon(boxb_bottom)
    # I_2D = reca.intersection(recb).area

    return I_2D

def compute_height(box_a, box_b, inter=True):

    corners1 = Box3D.box2corners3d_camcoord(box_a) 	# 8 x 3
    corners2 = Box3D.box2corners3d_camcoord(box_b)	# 8 x 3
    
    if inter: 		# compute overlap height
        ymax = min(corners1[0, 1], corners2[0, 1])
        ymin = max(corners1[4, 1], corners2[4, 1])
        height = max(0.0, ymax - ymin)
    else:			# compute union height
        ymax = max(corners1[0, 1], corners2[0, 1])
        ymin = min(corners1[4, 1], corners2[4, 1])
        height = max(0.0, ymax - ymin)

    return height

def compute_height_carla(boxa_corners, boxb_corners, inter=True):

    # Carla Format - box point in the bottom - 0,1,5,4 - built in that order, hence to get height 0,2 should work
    if inter: 		# compute overlap height
        ymax = min(boxa_corners[0, 2], boxb_corners[0, 2])
        ymin = max(boxa_corners[2, 2], boxb_corners[2, 2])
        height = max(0.0, ymax - ymin)
    else:			# compute union height
        ymax = max(boxa_corners[0, 2], boxb_corners[0, 2])
        ymin = min(boxa_corners[2, 2], boxb_corners[2, 2])
        height = max(0.0, ymax - ymin)

    return height

def PolyArea2D(pts):
    roll_pts = np.roll(pts, -1, axis=0)
    area = np.abs(np.sum((pts[:, 0] * roll_pts[:, 1] - pts[:, 1] * roll_pts[:, 0]))) * 0.5
    return area

def convex_area(boxa_bottom, boxb_bottom):

    # compute the convex area
    all_corners = np.vstack((boxa_bottom, boxb_bottom))
    C = ConvexHull(all_corners)
    convex_corners = all_corners[C.vertices]
    convex_area = PolyArea2D(convex_corners)

    return convex_area

def compute_iou(box_a, box_b, metric='iou_2d'):

    boxa_bottom = box_a.bbox_8wc_kitti[-5::-1, [0, 2]]
    boxb_bottom = box_b.bbox_8wc_kitti[-5::-1, [0, 2]]
    
    I_2D = compute_inter_2D(boxa_bottom, boxb_bottom)
    # print('IOU 2D', I_2D)

    # only needed for GIoU
    if 'giou' in metric:
        C_2D = convex_area(boxb_bottom, boxb_bottom)

    if '2d' in metric:		 	# return 2D IoU/GIoU
        # U_2D = box_a.dim_2 * box_a.l + box_b.w * box_b.l - I_2D
        U_2D = box_a.dim_w * box_a.dim_l + box_b.dim_w * box_b.dim_l - I_2D

        if metric == 'iou_2d':  return I_2D / U_2D
        if metric == 'giou_2d': return I_2D / U_2D - (C_2D - U_2D) / C_2D