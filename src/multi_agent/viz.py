#!/usr/bin/env python
# Modified code from Github 
import numpy as np
import cv2

def roty(t):
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def project_to_image(pts_3d, P):
    """ Project 3d points to image plane.

    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix

      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)

      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    """

    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]

def compute_box_3d(obj, P, mode='Kitti'):
    '''
    args : Takes an object and a projection matrix (P)
    returns 3D bounding box corners(8 points) for plotting in the shape - 8x3
    Values are in the Kitti format
    '''

    cam_coords = obj[11:14]
    h = obj[8]
    w = obj[9]
    l = obj[10]
    ry = obj[14]
    alpha = obj[3]
    bb2d = obj[4:8]
    threshold = obj[15]
    P_new = np.zeros((3, 4))
    P_new[0, :3] = P[0, :]
    P_new[1, :3] = P[1, :]
    P_new[2, :3] = P[2, :]
    P = P_new

    # compute rotational matrix around yaw axis
    R = roty(ry)
    if mode == 'Kitti':
        # 3d bounding box corners - Kitti Coordinate System = x right ,y - down,  z-forward
        x_corners = [l / 2, l / 2, -l / 2, - l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    else:
        # 3d bounding box corners - Carla Coordinate System = z, x, -y
        y_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        z_corners = [-h, -h, -h, -h, 0, 0, 0, 0]

    # rotate and translate 3d bounding box
    corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))

    corners_3d[0, :] = corners_3d[0, :] + cam_coords[0]
    corners_3d[1, :] = corners_3d[1, :] + cam_coords[1]
    corners_3d[2, :] = corners_3d[2, :] + cam_coords[2]

    # only draw 3d bounding box for objs in front of the camera
    if np.any(corners_3d[2, :] < 0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)

    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P)
    return corners_2d, np.transpose(corners_3d)

def draw_projected_box3d(image, qs, color=(0, 255, 0), thickness=2):
    """ Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    """
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        # cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
        cv2.line(image, (qs[i, 0], qs[i, 1]),
                 (qs[j, 0], qs[j, 1]), color, thickness)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]),
                 (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]),
                 (qs[j, 0], qs[j, 1]), color, thickness)

    #cv2.rectangle(image,(qs[5, 0], qs[5, 1]), (qs[0, 0], qs[0, 1]), color, -1)
    cv2.rectangle(image, (qs[6, 0], qs[6, 1]),
                  (qs[4, 0], qs[4, 1]), (2550, 0, 0), -1)
    #cv2.imwrite("/export/amsvenkat/project/visualization/kitti_object_vis/data/object/6.png", image)
    return image

class BoxData():
    '''
    Object details stored in the Carla format in the World cooridnates system '''

    def __init__(self, obj_cat='None', dim_h=0.0, dim_w=0.0, dim_l=0.0, x_wc=None, y_wc=None, z_wc=None, ry_wc=0.0, ry_wc_kitti= 0.0, score=0.0, alpha=0.0, bbox_4wc=None, bbox_8wc= None, 
                 x_wc_kitti=0.0, y_wc_kitti=0.0, z_wc_kitti=0.0, bbox_4wc_kitti=None, bbox_8wc_kitti=None):
        '''
        Constructor, initializes the object given the parameters in world coordinates
        bounding box rep as (num_point(4/8), 4)
        '''

        self.obj_cat = obj_cat
        self.dim_h = dim_h
        self.dim_w = dim_w
        self.dim_l = dim_l

        self.x_wc = x_wc
        self.y_wc = y_wc
        self.z_wc = z_wc
        self.x_wc_kitti = x_wc_kitti
        self.y_wc_kitti = y_wc_kitti
        self.z_wc_kitti = z_wc_kitti
        self.bbox_4wc = bbox_4wc
        self.bbox_8wc = bbox_8wc
        self.bbox_4wc_kitti = bbox_4wc_kitti
        self.bbox_8wc_kitti = bbox_8wc_kitti

        self.alpha = alpha
        self.ry_wc = ry_wc 
        self.ry_wc_kitti = ry_wc_kitti
        self.score = score


    def get_carla(self):
        '''
        Get Bbox points in the Carla world coordinates format as below
        This will be the standarad format for all the objects
        Bottom - Always the first 4 points
        Top - Always the last 4 points

            6 -------- 7
           /|         /|
          5 -------- 4 .
          | |        | |
          . 2 -------- 3
          |/         |/
          1 -------- 0

        '''

        if self.x_wc_kitti is not None or self.y_wc_kitti is not None or self.z_wc_kitti is not None:
            self.x_wc = self.z_wc_kitti
            self.y_wc = self.x_wc_kitti
            self.z_wc = -1 * self.y_wc_kitti

            # As the rotation direction along y axis in RHS Kitti equals the rotation direction along z in Lhs Carla - Clockwise direction
            self.ry_wc = self.ry_wc_kitti

        if self.bbox_8wc_kitti is not None:
            bbox = self.bbox_8wc_kitti
            self.bbox_8wc = np.transpose(np.vstack([bbox[:, 2], bbox[:, 0], -bbox[:, 1]]))
            self.bbox_4wc =self.bbox_8wc[:4, :]

    def get_kitti(self):
        '''
        TODO - get the box order 
        box corner order is like follows
        top is bottom because y direction is negative

            1 -------- 0 =-h		 
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4 =0
          |/         |/
          6 -------- 7    
	
	    rect/ref camera coord:
        # Getting Kitti from Carla coordinates
        '''
        
        if self.x_wc is not None or self.y_wc is not None or self.z_wc is not None:
            self.x_wc_kitti = self.y_wc
            self.y_wc_kitti = -self.z_wc
            self.z_wc_kitti = self.x_wc

            # As the rotation direction along y axis in RHS Kitti equals the rotation direction along z in Lhs Carla - Clockwise direction
            self.ry_wc_kitti = self.ry_wc
        
        #print(self.x_wc_kitti, self.y_wc_kitti, self.z_wc_kitti)
        # TODO - check if kitti wc are none or not
        if self.bbox_8wc is not None:
            bbox = self.bbox_8wc
            self.bbox_8wc_kitti = np.transpose(np.vstack([bbox[:, 1], -bbox[:, 2], bbox[:, 0]]))
            self.bbox_4wc_kitti =self.bbox_8wc_kitti[:4, :]

    def round_off(self, num):
        self.dim_h = round(self.dim_h, num)
        self.dim_w = round(self.dim_w, num)
        self.dim_l = round(self.dim_l, num)
        self.x_wc = round(self.x_wc, num)
        self.y_wc = round(self.y_wc, num)
        self.z_wc = round(self.z_wc, num)
        self.ry_wc= round(self.ry_wc, num)
        self.x_wc_kitti = round(self.x_wc_kitti, num)
        self.y_wc_kitti = round(self.y_wc_kitti, num)
        self.z_wc_kitti = round(self.z_wc_kitti, num)
        self.alpha = round(self.alpha, num)
        self.ry_wc_kitti = round(self.ry_wc_kitti, num)
        self.score = round(self.score, num)


    def construct_3d_box_carla_v1(self):
        '''
        Takes in object data and returns 3d box in Carla world coordinates 
        args : obj_data - ObjectData class object
        return : new_box - 8*3'''
        
        h = self.dim_h
        w = self.dim_w
        l = self.dim_l
        world_coords = [self.x_wc, self.y_wc, self.z_wc]
        ry = self.ry_wc
        
        # Coordintes are in World frame wrt Carla
        # Easy to way to make the box in Kitti format and then convert to Carla format
        # Kitti - RHS - x - right, y - down, z - fwd
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        # Carla Format - LHS - y - right, z - up, x - fwd
        xc_corners = z_corners
        yc_corners = x_corners
        zc_corners = [-i for i in y_corners]

        c = np.cos(ry)
        s = np.sin(ry)
        # Kitti Rotation - rot around y axis in RHS
        #R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        
        # Carla Rotation - rot around z axis in LHS 
        R = np.array([[c, s, 0 ], [-s, c, 0], [0, 0, 1]])
        #R = np.array([[-s, 0, c], [c, 0, s], [0, -1, 0]])

        x = np.vstack([xc_corners, yc_corners, zc_corners])
        
        # 3*8 = 3*3 X 3*8
        corners_3d = np.dot(R, x)
        corners_3d[0, :] = corners_3d[0, :] + world_coords[0]
        corners_3d[1, :] = corners_3d[1, :] + world_coords[1]
        corners_3d[2, :] = corners_3d[2, :] + world_coords[2]

        # 8*3
        new_box = np.transpose(corners_3d)
        #new_box_bottom = np.vstack((new_box[:2, :3], new_box[5, :3], new_box[4, :3]))
        # TODO _ get the boox points and their order from here
        new_box_bottom = np.vstack((new_box[:4, :3]))

        self.bbox_4wc = new_box_bottom
        self.bbox_8wc = new_box


    def construct_3d_box_carla(self):
        '''
        Process the box worlc coordinates in kitti format and convert to Carla world coordinates 
        return : new_box - 8*3'''
        
        h = self.dim_h
        w = self.dim_w
        l = self.dim_l

        world_coords = [self.x_wc_kitti, self.y_wc_kitti, self.z_wc_kitti]
        ry = self.ry_wc_kitti
        
        # Coordintes are in World frame wrt Kitti
        # Kitti - RHS - x - right, y - down, z - fwd
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        # Kitti Rotation - rot around y axis in RHS
        c = np.cos(ry)
        s = np.sin(ry)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

        coordinates = np.vstack([x_corners, y_corners, z_corners])
        
        # 3*8 = 3*3 X 3*8
        corners_3d = np.dot(R, coordinates)
        corners_3d[0, :] = corners_3d[0, :] + world_coords[0]
        corners_3d[1, :] = corners_3d[1, :] + world_coords[1]
        corners_3d[2, :] = corners_3d[2, :] + world_coords[2]

        # 8*3
        new_box = np.transpose(corners_3d)
        new_box_bottom = np.vstack((new_box[:4, :3]))

        self.bbox_4wc_kitti = new_box_bottom
        self.bbox_8wc_kitti = new_box

        self.get_carla()




