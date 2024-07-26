#!/usr/bin/env python
# Modified code from Carla Simulator
'''
Code to generate the labels for the Carla dataset in Kitti Format
get_images.py to be run first to generate the images and objects information'''


import os
import math
import re
import os.path
import time
import cv2
import re
import numpy as np

global rgb_info,seg_info, depth_info, depth_info_check
global running_index_count

# Kitti Resolution
VIEW_WIDTH = 2484//2
VIEW_HEIGHT = 750//2
VIEW_FOV = 90

Vehicle_COLOR = np.array([142, 0, 0])
Walker_COLOR = np.array([60, 20, 220])

VBB_COLOR = (0, 0, 255)
WBB_COLOR = (255, 0, 0)

rgb_info = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype='i')
seg_info = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype='i')
depth_info = []
depth_info_check = []

area_info = np.zeros(shape=[VIEW_HEIGHT, VIEW_WIDTH, 3], dtype=np.uint8)

main_dir = '/export/amsvenkat/project/data/train_v5/'
dir_label = main_dir + 'label_1/'
dir_draw = main_dir + 'draw_bounding_box_1/'
image_dir = main_dir + 'image_1/'

if not os.path.exists(dir_draw):
    os.makedirs(dir_draw)
if not os.path.exists(dir_label):
    os.makedirs(dir_label)
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

dataEA = len(os.walk(main_dir +'VehicleBBox/').__next__()[2])


class BoxData(object):
    def __init__(self) -> None:
        self.cat = None
        self.truncation = None
        self.occlusion = None
        self.alpha = None
        self.xmin = None
        self.ymin = None
        self.xmax = None
        self.ymax = None
        self.h = None
        self.w = None
        self.l = None
        self.x = None
        self.y = None
        self.z = None
        self.rot_y = None
        self.dist = None
        self.in_image = None
        self.points = None
        self.area = None
    
    def _area(self):
        self.area = (self.xmax - self.xmin) * (self.ymax - self.ymin)
        return self.area

def reading_data(index):
    '''
    process the data stored for image and the objects from the text file'''
    
    global rgb_info, seg_info, depth_info, depth_info_check

    k = 0
    rgb_img = cv2.imread(main_dir + 'image/' + str(index) + '.png', cv2.IMREAD_COLOR)
    seg_img = cv2.imread(main_dir + 'SegmentationImage/seg' + str(index) + '.png', cv2.IMREAD_COLOR)
    depth_img = np.load(main_dir + 'DepthImage/dmgrey_{}.npy'.format(str(index)))

    line_length = {}
    bbox3d_img_pts = {}
    bbox3d_data = { 'Car' : [], 'Pedestrian' : []}

    if str(rgb_img) != 'None' and str(seg_img) != 'None':

        for cat in ['Car', 'Pedestrian']:
            # print('cat',cat)

            v_data = []
            k = 0
            x = 'Vehicle' if cat == 'Car' else 'Pedestrian'
 
            with open(main_dir + x + 'BBox/bbox' + str(index), 'r') as f1:
                    rawdata = f1.read()

            bbox_data = re.findall(r'-?\d+', rawdata)

            # no of objects
            line_len = int(len(bbox_data) / 16)
            # store in array format - 8 points
            v_bbox_data= [[0 for col in range(8)] for row in range(line_len)]

            for i in range(int(len( bbox_data)/2)):
                j = i*2
                v_data.append(
                    tuple((int( bbox_data[j]), int( bbox_data[j+1]))))

            for i in range(int(len(bbox_data)/16)):
                for j in range(8):
                    v_bbox_data[i][j] = v_data[k]
                    k += 1
            
            bbox3d_img_pts[cat] = v_bbox_data
            line_length[cat] = line_len

            with open(main_dir + x + 'BBox_wc/bbox' + str(index), 'r') as f2 :
                label_data = f2.readlines()

                for l in label_data:
                    x = l.split('#')
                    bbox3d_data[cat].append(x)
        
        v_bbox_data = []

        rgb_info = rgb_img
        seg_info = seg_img
        depth_info = depth_img

        # returns car and peds bbox data and their related labels
        return bbox3d_img_pts ,line_length, bbox3d_data
    else:
        return False

def converting(bounding_boxes, line_length):
    '''
    Converting Bounindg Boxes to to x,y format'''
    
    bbox2d_pts = { 'Car' : [], 'Pedestrian' : []}
    
    for cat in ['Car', 'Pedestrian']:
        points_array = []
        b = bounding_boxes[cat]
        l = line_length[cat]
        # print('l',l)
        
        bb_4data = [[0 for col in range(4)] for row in range(l)]
        
        for i in range(l):
            k = 0
            points_array_x = []
            points_array_y = []
            
            for j in range(8):
                points_array_x.append(b[i][j][0])
                points_array_y.append(b[i][j][1])

                max_x = max(points_array_x)
                min_x = min(points_array_x)
                max_y = max(points_array_y)
                min_y = min(points_array_y)

            points_array.append(tuple((min_x, min_y)))
            points_array.append(tuple((max_x, min_y)))
            points_array.append(tuple((max_x, max_y)))
            points_array.append(tuple((min_x, max_y)))

        for i in range(l):
            for j in range(int(len(points_array)/l)):
                bb_4data[i][j] = points_array[k]
                k += 1

        bbox2d_pts[cat] = bb_4data

    return bbox2d_pts

def small_objects_excluded(array, bb_min):
    '''
    Scene objects which are small in size are excluded based on pixel dimensions'''
    
    diff_x = array[1] - array[0]
    diff_y = array[3] - array[2]
    if (diff_x > bb_min and diff_y > bb_min):
        return True
    return False

def post_occluded_objects_excluded(array, color):
    
    global seg_info
    top_left = seg_info[array[2]+1, array[0]+1][0]
    top_right = seg_info[array[2]+1, array[1]-1][0]
    bottom_left = seg_info[array[3]-1, array[0]+1][0]
    bottom_right = seg_info[array[3]-1, array[1]-1][0]
    if top_left == color[0] and top_right == color[0] and bottom_left == color[0] and bottom_right == color[0]:
        return True

    return False

def fitting_x(x1, x2, range_min, range_max, color):
    '''
    Fitting the object in the image based on the color of the object in the X axis'''
    
    global seg_info
    state = False
    cali_point = 0
    if (x1 < x2):
        for search_point in range(x1, x2):
            for range_of_points in range(range_min, range_max):
                if seg_info[range_of_points, search_point][0] == color[0]:
                    cali_point = search_point
                    state = True
                    break
            if state == True:
                break

    else:
        for search_point in range(x1, x2, -1):
            for range_of_points in range(range_min, range_max):
                if seg_info[range_of_points, search_point][0] == color[0]:
                    cali_point = search_point
                    state = True
                    break
            if state == True:
                break

    return cali_point

def fitting_y(y1, y2, range_min, range_max, color):
    '''
    Fitting the object in the image based on the color of the object in the Y axis'''
    
    global seg_info
    state = False
    cali_point = 0
    if (y1 < y2):
        for search_point in range(y1, y2):
            for range_of_points in range(range_min, range_max):
                if seg_info[search_point, range_of_points][0] == color[0]:
                    cali_point = search_point
                    state = True
                    break
            if state == True:
                break

    else:
        for search_point in range(y1, y2, -1):
            for range_of_points in range(range_min, range_max):
                if seg_info[search_point, range_of_points][0] == color[0]:
                    cali_point = search_point
                    state = True
                    break
            if state == True:
                break

    return cali_point

def filtering_non_scene(t, array, color):
    '''
    Filters out the non scene objects
    Or occluded by more than 40 % for Car and 20 % for Pedestrian'''
    
    global seg_info

    total = 1
    corr = 0
    x_mid = int((array[1] + array[0])/2)
    y_mid = int((array[3] + array[2])/2)
    # Taking the box around the center by 50 % of the box
    resized_array = [(x_mid + array[0])/2, (x_mid + array[1])/2,
                     (y_mid + array[2])/2, (y_mid + array[3])/2]
    
    for i in range(array[0], array[1]):
        for j in range(array[2], array[3]):
            total += 1
            if seg_info[j, i][0] == color[0]:
                corr += 1
    
    if (t == 'Car' and ((corr/total) * 100)  > 40) or (t == 'Pedestrian' and ((corr/total) * 100)  > 20):
        return True

    return False

def filtering_max_depth(array, dist, margin):
    '''
    Filters out the objects which are fully occluded and are far away from the camera'''
    

    x_mid = int((array[1] + array[0])/2)
    y_mid = int((array[3] + array[2])/2)

    boxlength = int(array[1] - array[0])
    boxheight = int(array[3] - array[2])

    resized_array = [(x_mid + array[0])/2, (x_mid + array[1])/2,(y_mid + array[2])/2, (y_mid + array[3])/2]
    resized_array_new = [(((x_mid + array[0])/2) - (boxlength/5)),(((x_mid + array[0])/2) + (boxlength/5)), y_mid, (y_mid + (boxheight/2))]
    
    depth = 0
    num = 0
    medianval = []

    for i in range(int(array[0]), int(array[1])):
        for j in range(int(array[2]), int(array[3])):
            depth += depth_info[j, i]
            num += 1
            medianval.append(depth_info[j, i])
    
    if num == 0:
        return False
    
    else:
        #TODO check for median depth also
        avg_depth = depth/num
        median = np.median(medianval)
        # print(dist, median, margin)
        # return True
        if abs(median - dist) < margin:
            return True

def exclude_small_objects(array, bb_min, t):
    '''
    Filters out the objects which are smaller than the threshold based on the real size'''
    
    dist_l = math.sqrt((array[0][0] - array[1][0]) ** 2 + (array[0][1] - array[1][1])**2) 
    dist_b = math.sqrt((array[1][0] - array[2][0]) ** 2 + (array[1][1] - array[2][1])**2)

    if (t == 'Car') and (dist_l > bb_min or dist_b > bb_min) and ((dist_l*dist_b) > 50):
       # print(dist_b, '-', dist_l, '-', bb_min)
        return True
    if (t == 'Pedestrian') and (dist_l > bb_min or dist_b > bb_min):
        #print(dist_b, '-', dist_l, '-', bb_min)
        return True
    
    return False

def processed_actors(v_data, t, dist, margin, bb_min): 
    '''
    Processes the scene objects and satisfies - depth, size constraints '''
    
    # if exclude_small_objects(x, min_dim, cat) and 
    global seg_info, area_info, Vehicle_COLOR, Walker_COLOR, depth_info

    array_x = []
    array_y = []

    for i in range(4):
        array_x.append(v_data[i][0])
    for j in range(4):
        array_y.append(v_data[j][1])

    for i in range(4):
        if array_x[i] <= 0:
            array_x[i] = 1
        elif array_x[i] >= VIEW_WIDTH - 1:
            array_x[i] = VIEW_WIDTH - 2
    for j in range(4):
        if array_y[j] <= 0:
            array_y[j] = 1
        elif array_y[j] >= VIEW_HEIGHT - 1:
            array_y[j] = VIEW_HEIGHT - 2

    min_x = min(array_x)
    max_x = max(array_x)
    min_y = min(array_y)
    max_y = max(array_y)
    v_bb_array = [min_x, max_x, min_y, max_y]
    
    color = Vehicle_COLOR if t == 'Car' else Walker_COLOR

    cali_min_x = fitting_x(min_x, max_x, min_y, max_y, color)
    cali_max_x = fitting_x(max_x, min_x, min_y, max_y, color)
    cali_min_y = fitting_y(min_y, max_y, min_x, max_x, color)
    cali_max_y = fitting_y(max_y, min_y, min_x, max_x, color)
    
    v_cali_array = [cali_min_x, cali_max_x, cali_min_y, cali_max_y]

    if v_bb_array and v_cali_array:

        if t == 'Car' and filtering_non_scene(t, v_bb_array, Vehicle_COLOR) and filtering_max_depth(v_cali_array, dist, margin) :
            if small_objects_excluded(v_cali_array, bb_min):
                return True

        if t == 'Pedestrian' and filtering_non_scene(t, v_bb_array, Walker_COLOR): # and filtering_occlusion_depth(v_cali_array, dist, margin):
            if small_objects_excluded(v_cali_array, bb_min):
                return True
    
    return False

def check_depth():
    '''
    Checks the depth of the objects in the scene'''
    
    global depth_info

    i = []
    for y in range(VIEW_HEIGHT-1):
        for x in range(VIEW_WIDTH-1):
            i.append(depth_info[y, x])
    i = np.array(i)

def draw_in_carla(dim,loc, ry):
    '''
    Draws the bounding box in Carla notation'''
    
    h = dim[0]
    w = dim[1]
    l = dim[2]
    # carla
    world_coords =[loc[2], loc[0], -loc[1]]

    # Coordintes are in World frame wrt Carla

    # Kitti - RHS - x - right, y - down, z - fwd
    x_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2] 
    y_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    z_corners = [0, 0, 0, 0, h, h, h, h]
    
    c = np.cos(ry)
    s = np.sin(ry)
    # Kitti Rotation - rot around y axis in RHS
    #R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    
    # Carla Rotation - rot around z axis in LHS 
    R = np.array([[c, s, 0 ], [-s, c, 0], [0, 0, 1]])
    #R = np.array([[-s, 0, c], [c, 0, s], [0, -1, 0]])
    x = np.vstack([x_corners, y_corners, z_corners])
    
    # 3*8 = 3*3 X 3*8
    corners_3d = np.dot(R, x)
    corners_3d[0, :] = corners_3d[0, :] + world_coords[0]
    corners_3d[1, :] = corners_3d[1, :] + world_coords[1]
    corners_3d[2, :] = corners_3d[2, :] + world_coords[2]
    
    # 8*3
    new_box = np.transpose(corners_3d)
    new_box_bottom = np.vstack((new_box[:4, :3]))

def get_roty(obj_data):
    '''
    Get the rotation in the Kitti format for the get_images version 2'''

    actortrans = round(float(obj_data[1].replace(' ', '').replace(',', '=').split('=')[9]), 2)
    camtrans = round(float(obj_data[5].replace(' ', '').replace(',', '=').split('=')[9]), 2)
    agenttrans = round(float(obj_data[8].replace(' ', '').replace(',', '=').split('=')[9]),2)
    
    # Check which one to use - camtrans or agenttrans
    rel_angle = math.radians(actortrans - agenttrans -90)
    if rel_angle > math.pi:
        rel_angle = rel_angle - 2 * math.pi
    elif rel_angle < - math.pi:
        rel_angle = rel_angle + 2 * math.pi

    return rel_angle

def get_modified_roty(rotation):
    '''
    Get the rotation in the Kitti format for the get_images version 1'''

    rot_y = float(rotation) - math.radians(90)
    
    if rot_y > math.pi:
        rot_y = rot_y - 2*math.pi
    elif rot_y < - math.pi:
        rot_y = rot_y + 2*math.pi
    
    return rot_y

def get_modified_alpha(actorloc):
    ''''
    Get the alpha in the Kitti format for the get_images version 1'''
    
    alpha = float(math.atan2(float(actorloc[2]), float(actorloc[0])))
    
    if alpha > math.pi:
        alpha = alpha - 2*math.pi
    elif alpha < - math.pi:
        alpha = alpha + 2*math.pi

    return alpha

def get_depth_margin(width, length, rot_y):
    '''
    Get the depth margin for the object based on the rotation of the object'''
    
    rot_y_deg = math.degrees(rot_y)

    margin_cat = {}

    # # Solution 1 
    # if abs(rot_y_deg) in range(85,  95):
    #     # y = length, x = width  w/2 or could be the other way - check
    #     margin = float(width)/2 
    
    # elif abs(rot_y_deg) in range(0, 20)  or abs(rot_y_deg) in range(160, 180):
    #     # y = length, x = width   l/2  or could be the other way - check
    #     margin = float(length)/2 
        
    # else:
    #     margin =(math.sqrt((float(width)**2) + (float(length)**2))/2) -1
    
    # #margin = margin + (0.1 * margin)
    # margin = margin + 1
    
    arr = np.array([float(width)* 2, float(length) * 2], dtype=np.float32)
    
    # Solution 2
    if width and length:
        # margin = math.sqrt((float(width)**2) + (float(length)**2))/2
        margin = np.max(arr)

    margin_cat = {'Car' : margin , 'Pedestrian' : 5 }

    return margin_cat

def draw_3dbox(rgb_info, points):
    #base
    cv2.line(rgb_info, points[0], points[1], VBB_COLOR)
    cv2.line(rgb_info, points[1], points[2], VBB_COLOR)
    cv2.line(rgb_info, points[2], points[3], VBB_COLOR)
    cv2.line(rgb_info, points[3], points[0], VBB_COLOR)
    #cv2.line(img, points[3], x, VBB_COL
    cv2.line(rgb_info, points[4], points[5], VBB_COLOR)
    cv2.line(rgb_info, points[5], points[6], VBB_COLOR)
    cv2.line(rgb_info, points[6], points[7], VBB_COLOR)
    cv2.line(rgb_info, points[7], points[4], VBB_COLOR)
    # se-top
    cv2.line(rgb_info, points[0], points[4], VBB_COLOR)
    cv2.line(rgb_info, points[1], points[5], VBB_COLOR)
    cv2.line(rgb_info, points[2], points[6], VBB_COLOR)
    cv2.line(rgb_info, points[3], points[7], VBB_COLOR)

def get_iou(obj1, obj2):

    x_left =  max(obj1.xmin, obj2.xmin)
    y_top = max(obj1.ymin, obj2.ymin)
    x_right = min(obj1.xmax, obj2.xmax)
    y_bottom = min(obj1.ymax, obj2.ymax)

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    score = abs(intersection_area / float(obj1.area + obj2.area - intersection_area))
    area_score = abs(intersection_area / float(obj2.area))

    return score , intersection_area

def get_final_scene(object_list):

    max_iou = 0
    area_list = sorted(object_list, key=lambda x: (x.area), reverse=True)
    area_list_copy = area_list.copy()
    for obj in area_list:
        filter_box = None
        for obj2 in area_list:
            if obj.cat == obj2.cat and obj != obj2  and abs(abs(obj.rot_y) - abs(obj2.rot_y)) < math.radians(20):
                    iou , area_score = get_iou(obj, obj2)
                    if area_score > 0.35 :
                        filter_box = obj2

        if filter_box is not None:
            if obj.dist < filter_box.dist:
                area_list.remove(filter_box)
            else:
                area_list.remove(obj)

    return area_list

def write_label(bboximg_2d, bboximg_3d, bbox_data, version):
    '''
    Write Kitti format label to file
    bboximg_2d : 2d bbox points
    bboximg_3d : 3d bbox points
    bbox_data : 3d bbox data
    version : 1 or 2
    '''
    
    global running_index_count, dir_label, rgb_info
    # img, imd2d, img3d, label_info
    # to rewrite the images
    cv2.imwrite(image_dir + str(running_index_count) + '.png', rgb_info)
    # to write the label
    f2 = open(dir_label + str(running_index_count) + '.txt', 'w')

    dimension = {'Car' : 10 , 'Pedestrian' : 7}
    object_list = []
    count = 0

    for cat in ['Car', 'Pedestrian']:
        # print(cat)
        img_2d = bboximg_2d[cat]
        img_3d = bboximg_3d[cat]
        data = bbox_data[cat]
        min_dim = dimension[cat]

        if bboximg_2d != False:
            
            # Iterating through every line in the file
            for k, v in enumerate(img_3d):
                
                bbox2d = img_2d[k]
                line_data = data[k]

                dist = float(line_data[7])
                
                if ((dist > 50) or (cat == 'Pedestrian' and dist > 40)):
                    continue

                points = [((v[i])[0], (v[i])[1]) for i in range(8)]
                x = [points[0], points[1], points[2], points[3]]

                actordim = line_data[3].split(' ')
                actorloc = line_data[4].split(' ')
                
                # as Carla has the object centers in Cars and Pedestrians differently
                if cat == 'Car':
                    y = round(float(actorloc[1]),2)
                if cat == 'Pedestrian':
                    y = round((float(actorloc[1]) + (float(actordim[0])/2)), 2)

                # In old version of get_images v1, rot_y is calcuated but needs to be modified for Kitti, and alpha is calculated in this file
                if version == 'v1':
                    alpha = get_modified_alpha(actorloc)
                    rot_y = get_modified_roty(line_data[5])

                # In new version of get_images v2 , alpha is calculated and taken as is , rot_y needs to be calculated in this file
                if version == 'v2':
                    # print(obj_data)
                    alpha = round(float(line_data[2]), 2)
                    rot_y = get_roty(line_data)

                margins= get_depth_margin(actordim[1], actordim[2], rot_y)
                # print(margins)

                obj_data  = BoxData()
                obj_data.cat = cat
                obj_data.truncation = 0
                obj_data.occlusion = 0
                obj_data.alpha = alpha
                obj_data.xmin = bbox2d[0][0]
                obj_data.ymin = bbox2d[0][1]
                obj_data.xmax = bbox2d[2][0]
                obj_data.ymax = bbox2d[3][1]
                obj_data.h = round(float(actordim[0]), 2)
                obj_data.w = round(float(actordim[1]), 2)
                obj_data.l = round(float(actordim[2]), 2)
                obj_data.x = round(float(actorloc[0]), 2)
                obj_data.y = y
                obj_data.z = round(float(actorloc[2]), 2)
                obj_data.rot_y = rot_y
                obj_data.dist = dist
                obj_data.points = points
                obj_data._area()

                if processed_actors(bbox2d, cat, dist, float(margins[cat]), min_dim):
                    object_list.append(obj_data)
                    # draw_in_carla(actordim, actorloc, rot_y)

    object_list = get_final_scene(object_list)

    for actor in object_list: 
        
            draw_3dbox(rgb_info=rgb_info, points=actor.points)
            
            f2.write(actor.cat + ' ' + str(0) + ' ' + str(0) + ' ' + str(round(actor.alpha, 2)) + ' ' +
                str(actor.xmin) + ' ' + str(actor.ymin) + ' ' + str(actor.xmax) + ' ' + str(actor.ymax) + ' ' + 
                str(round(float(actor.h), 2)) + ' ' + str(round(float(actor.w), 2)) + ' ' + str(round(float(actor.l), 2)) + ' ' +
                str(round(float(actor.x), 2)) + ' ' + str(actor.y) + ' ' + str(round(float(actor.z), 2)) + ' ' +
                str(actor.rot_y) + ' ' + str(0) + '\n')
            
            # f2.write(obj_data.cat + ' ' + str(0) + ' ' + str(0) + ' ' + str(round(obj_data.alpha, 2)) + ' ' +
            #     str(obj_data.xmin) + ' ' + str(obj_data.ymin) + ' ' + str(obj_data.xmax) + ' ' + str(obj_data.ymax) + ' ' + 
            #     str(round(float(obj_data.h), 2)) + ' ' + str(round(float(obj_data.w), 2)) + ' ' + str(round(float(obj_data.l), 2)) + ' ' +
            #     str(round(float(obj_data.x), 2)) + ' ' + str(obj_data.y) + ' ' + str(round(float(obj_data.z), 2)) + ' ' +
            #     str(obj_data.rot_y) + ' ' + str(0) + '\n')

            count += 1
                           
    f2.close()
    # to draw bboxes on the image
    cv2.imwrite(dir_draw + 'image_' + str(running_index_count)+'.png', rgb_info)
    
    if count > 0:
        running_index_count += 1
        return running_index_count
    else:
        return running_index_count 

def run():
    global running_index_count
    running_index_count = 8115

    #label1 
    # Town 02 0 - 5000 = 0 - 4760
    # Town 05 5001 - 6000 = 4761 -5628
    # Town 05 6001 - 7000 = 5629 - 6507
    # town 01 7001 - 8000 = 6508 - 7317
    # town 03 8001 - 9000 = 7318 - 8114
    # town 02 9000 - 10000 = 8115 - 

    #label 2
    # 0 - 7000 = 0 - 6504
    # 7001 - 8000 = 6505 - 7314

    # check the version of get_images.py
    version = 'v2'
    
    for i in range(9000, 12001):
        # print(i)
        
        if i % 100 == 0:
            print(i)
        
        info = reading_data(i)
        
        if info != False:
            bbox3d_img_pts, line_length, bbox3d_data = info
            box2d_img_pts = converting(bbox3d_img_pts, line_length)
            write_label( box2d_img_pts, bbox3d_img_pts, bbox3d_data , version)


if __name__ == '__main__':
    print('Starting..')
    start = time.time()
    run()
    end = time.time()
    print(float(end - start))
    print('Done')