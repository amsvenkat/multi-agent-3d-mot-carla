#!/usr/bin/env python
# Code by Amrutha Venkatesan
import itertools
import math
import numpy as np
import copy

import yaml
import os, numpy as np

from iou_calc import compute_iou
from viz import BoxData


def get_config(path):
    with open(path) as file:
        config = yaml.safe_load(file)
    
    return config

def get_coordinates( box1, box2):
    # new - (num_points - (b0, b1, b2, b3)) * 3  (x,y,z)
    '''
    BBox of the order in Carla coordinates - X-fwd, Y-right
    0 ---- 1
    |      |
    3 ---- 2
    '''
    box1_dict = {}
    box2_dict = {}

    box1_dict ['xmax'] = np.max(box1[:, 0])
    box1_dict ['xmin'] = np.min(box1[:, 0])
    box1_dict ['ymax'] = np.max(box1[:, 1])
    box1_dict ['ymin'] = np.max(box1[:, 1])

    box2_dict ['xmax'] = np.max(box2[:, 0])
    box2_dict ['xmin'] = np.max(box2[:, 0])
    box2_dict ['ymax'] = np.max(box2[:, 1])
    box2_dict ['ymin'] = np.max(box2[:, 1])
    return box1_dict, box2_dict


def get_aligned_boxes(box1, box2):

    rot_1 = box1.ry_wc
    rot_2 = box2.ry_wc

    ry = []

    if abs(rot_1 - rot_2) > 0.0 and abs(rot_1 - rot_2) < 90:
        ry = [rot_1, rot_2]
        edge_angles_sin = [np.sin(angle) for angle in ry]
        edge_angles_cos = [np.cos(angle) for angle in ry]
        ry_merged = np.arctan2(np.mean(edge_angles_sin), np.mean(edge_angles_cos))

        for box in [box1, box2]:
            l =box.dim_l
            w = box.dim_w
            h = box.dim_h
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

    
    else:
        return False


def iou_score(box1, box2):
    '''
    Boxes are of the
    IOU for the aligned boxes - boxes of the shape (4,5) -> rows - (x,y,z,1), columns - (b0, b1, b2, b3) 
    IOU calculated as 
    
    per the image coordinates,
    xleft,ytop    ---------- xright,ytop
                 |          |
    xleft,ybottom ----------  xright,ybottom
    
    To Carla as,
    xtop,yleft   ---------- xtop,yright
                |          |    
    xbottom,yleft ---------- xbottom,yright
    (0,0)
    '''
    
    box1, box2 = get_coordinates(box1, box2)

    # x_left = max(box1['x1'], box2['x1'])
    # y_top = max(box1['y1'], box2['y1'])
    # x_right = min(box1['x2'], box2['x2'])
    # y_bottom = min(box1['y2'], box2['y2'])

    # x_left = min(box1['y2'], box2['y2'])
    # y_top =  max(box1['x1'], box2['x1'])
    # x_right = max(box1['y1'], box2['y1'])
    # y_bottom = min(box1['x2'], box2['x2'])

    # y_left =  max(box1['x1'], box2['x1'])
    # x_top = max(box1['y1'], box2['y1'])
    # y_right = min(box1['x2'], box2['x2'])
    # x_bottom = min(box1['y2'], box2['y2'])
    
    # if x_right > x_left or y_bottom < y_top:
    #     print('0 score')
    #     return 0.0  , []
    
    # intersection_area = (x_right - x_left) * (y_bottom - y_top)
    # # Box coordinates in the counter clockwise direction - 1,2,3,4,1
    # intersection_area_box = np.array([[x_left, x_left, x_right, x_right, x_left], [y_top, y_bottom, y_bottom, y_top, y_top], [1,1,1,1,1]])
    
    # box1_area = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    # box2_area = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
    # score = abs(intersection_area / float(box1_area + box2_area - intersection_area))
    # print('score', score)

    x_top = min(box1['xmax'], box2['xmax'])
    y_left =  max(box1['ymin'], box2['ymin'])
    x_bottom = max(box1['xmin'], box2['xmin'])
    y_right = min(box1['ymax'], box2['ymax'])

    inter_area = (y_right - y_left) * (x_top - x_bottom)
    b1_area = (box1['xmax'] - box1['xmin']) * (box1['ymax'] - box1['ymin'])
    b2_area = (box2['xmax'] - box2['xmin']) * (box2['ymax'] - box2['ymin'])

    score2 = abs(inter_area / float(b1_area + b2_area - inter_area))
    # print('score2', score2)

    # return score2, intersection_area_box
    return score2


class Merger(object): 
    '''
    This class is used to merge the information from multiple agents'''
    def __init__(self,  info):
        self.info= info
        self.centers = [[obj.x_wc, obj.y_wc, obj.z_wc] for obj in info]
        self.dimensions =[[obj.dim_h, obj.dim_w, obj.dim_l] for obj in info]
        self.ry =[obj.ry_wc for obj in info]
        self.cat = info[0].obj_cat
        self.score = [obj.score for obj in info]
        self.alpha = [obj.alpha for obj in info]

    def add_center(self, location) -> None:
        self.centers.append(location)

    def add_dim(self, dims) -> None:
        self.dimensions = dims
    
    def add_boxes(self, boxes) -> None:
        self.boxes = boxes
    
    def get_avg_score(self) -> float:
        score = np.mean(self.score)
        return score

    def get_merged_center(self) -> np.ndarray:
        center_merged = np.mean(self.centers, axis=0)
        return center_merged
    
    def get_dim(self) -> np.ndarray:
        dim_merged = np.mean(self.dimensions, axis=0)
        return dim_merged

    def get_ry(self) -> float:

        edge_angles_sin = [np.sin(angle) for angle in self.ry]
        edge_angles_cos = [np.cos(angle) for angle in self.ry]
    
        # TODO - filtering needed to discard the outliers - may be part of iou
        ry_merged = np.arctan2(np.mean(edge_angles_sin), np.mean(edge_angles_cos))

        assert ry_merged >= -math.pi and ry_merged <= math.pi, 'Rotation is not correct'
    
        return ry_merged

    @staticmethod
    def get_label(obj_data) -> np.ndarray:
        '''
        Process the the object data in the AB3DMOT Kitti World coordinates
        args : obj_data - ObjectData class object
        returns the label for the object
        '''

        # Label Format as per the below for 3D Object Detections
        # Frame |   Type  |   2D BBOX (x1, y1, x2, y2)  | Score |    3D BBOX (h, w, l, x, y, z, rot_y)      | Alpha | 
        # label = np.array([0.0, obj_data.obj_cat, 0.0, 0.0, 0.0, 0.0, obj_data.score, obj_data.h, obj_data.w, obj_data.l,
        # obj_data.x_wc, obj_data.y_wc, obj_data.z_wc, obj_data.ry, obj_data.alpha], dtype=object)

        label = np.array([0.0, obj_data.obj_cat, 0.0, 0.0, 0.0, 0.0, obj_data.score, obj_data.dim_h, obj_data.dim_w, obj_data.dim_l,
        obj_data.x_wc_kitti, obj_data.y_wc_kitti, obj_data.z_wc_kitti, obj_data.ry_wc_kitti, obj_data.alpha], dtype=object)

        return label
    
    def get_merged(self):
        '''
        Returns the merged object data for the object in Carla Format(LHS)'''
        # print('get merged')
        center_merged = self.get_merged_center()
        orientation_merged = self.get_ry()
        dim_merged = self.get_dim()
        score = self.get_avg_score()
        
        obj_data = BoxData()
        obj_data.obj_cat = self.cat
        obj_data.dim_h = dim_merged[0]
        obj_data.dim_w = dim_merged[1]
        obj_data.dim_l = dim_merged[2]
        
        obj_data.x_wc = center_merged[0]
        obj_data.y_wc = center_merged[1]
        obj_data.z_wc = center_merged[2]
       
        # TODO  - check if alpha value is needed
        obj_data.alpha = 0.0
        obj_data.ry_wc = orientation_merged
        obj_data.score = score

        obj_data.get_kitti()

        obj_data.construct_3d_box_carla()

        return obj_data

def check_rotation(obja, objb):
    rot1 = obja.ry_wc_kitti
    rot2 = objb.ry_wc_kitti
    if (rot1 > 0) and (rot2> 0) or (rot1 < 0) and (rot2 < 0):
        if abs(rot1 - rot2) < math.radians(45):
            return True
        else:
            return False
        
    elif (rot1 > 0) and (rot2 < 0) or (rot1 < 0) and (rot2 > 0):
        rot2 = rot2 + math.radians(180) if rot2 < 0 else rot2
        rot2 = rot2 - math.radians(180) if rot2 > 0 else rot2
        
        if abs(rot1 - rot2) < math.radians(45):
            return True
        else:
            return False


class Overlap(object):
    def __init__(self, all_detections, config, obj_ind):
        self.config = config
        self.all_detections = all_detections
        self.cluster_threshold = self.config['cluster_threshold']
        self.iou_threshold =  self.config['iou_threshold']
        self.sim_threshold = self.config['sim_threshold']
        self.clusters = {}
        self.obj_ind = obj_ind
        self.track_id_lookup  = {}
        self.actor_taken_list = []

    def clustering(self):
        '''
        Clustering based on the merging threshold in the config file 
        eturns a dictionary of clusters with object count as key and clusters of (agent_id, objects ids, cat) as values
        '''

        predictions_cpy = copy.copy(self.all_detections)
        clusters = {}
    
        taken = set()
        # Index to keep track of the cluster id
        OBJ_ID = 0

        for agent_id, agent_dets in self.all_detections.items() :
            
            # Get ids of all other agents only 
            other_agent_ids = [x for x in predictions_cpy.keys() if x != agent_id]

            # Compare the detections of the agent with all other agents detections
            for det_a in agent_dets:

                loc_a = np.array([det_a.x_wc, det_a.y_wc, det_a.z_wc])

                if det_a in taken:
                    continue

                matched = [(agent_id, det_a)]
                taken.add(det_a)
                
                for id in other_agent_ids:

                    proximity = []
                    
                    for det_b in predictions_cpy[id]:
                        
                        if det_b not in taken and det_b.obj_cat == det_a.obj_cat :
                            loc_b = np.array([det_b.x_wc, det_b.y_wc, det_b.z_wc])
                            metric = np.linalg.norm(loc_a - loc_b)
                            proximity.append(dict(obj=det_b, dist=metric))
                        else:
                            continue
                        
                    if proximity != []:
                        # sort the distances in ascending order
                        proximity = sorted(proximity, key=lambda x: ( x['dist']))
                        closest = proximity[0]

                        # Check if the closest object is within the threshold
                        if closest == [] or closest['dist'] > self.cluster_threshold[det_a.obj_cat]:
                            continue
                        
                        else:
                            # print('Matched')
                            matched.append((id, closest['obj']))
                            taken.add(closest['obj'])                           

                clusters[OBJ_ID] = matched
                OBJ_ID += 1
                matched = []

                # Remove agent that are already checked for clustering
            predictions_cpy.pop(agent_id)
        return clusters

    def hierarchical_iou(self, cluster):
        '''
        Takes in a cluster of objects and merges them based on the IOU score
        args : clusters of potential merging objects
        returns : merged object data
        Assumption : Majority score is always given priority
        '''

        score_tracker = []
        merging_list = list()
        merging = 0
        ranking = {}
        main_list = []

        # Create a list of prediction objects for every cluster
        for ag_id, obj in cluster:
            main_list.append(obj)
        
        current_list = copy.copy(main_list)
        ranking = {k:0 for k in main_list}
        counter = len(main_list)

        # Run till length of the tree (no. of objects in clusster =1)
        while counter > 1:
            
            # Create combos of pairs of the boxes in the cluster 
            combines = itertools.combinations(current_list, 2)
    
            for (c1, c2) in combines: 

                box1 = np.array(c1.bbox_4wc)
                box2 = np.array(c2.bbox_4wc)
                
                # Convex Hull IOU for non aligned boxes                
                # Check if the rotation is not antiparallel
                if c1.obj_cat == 'Car': 
                    rot_flag = check_rotation(c1, c2)
                    if (not rot_flag) :
                        continue
                
                score = compute_iou(c1, c2)
                score_tracker.append([score, c1, c2])

            # To get a sorted list of iou scores in descending order of every pair of boxes in the cluster
            sorted_scores = sorted(score_tracker, key=lambda x: x[0], reverse=True)

            if sorted_scores == []:
                continue
            
            # If the score of the highest IOU is greater than 0.5, merge the boxes
            if sorted_scores[0][0] > self.config['iou_threshold']:
                merging +=1

                merging_list.extend(sorted_scores[0][1:])
                merger = Merger(merging_list)
                
                # Add the new merged object to the current list
                m = merger.get_merged()
                current_list.append(m)
                
                # Remove the individual objecs that have been merged
                current_list = [x for x in current_list if x not in merging_list]
                ranking[m] = 0
                # To get the ranking of the merged object based on number of objects merged    
                
                for i in [c1, c2]:
                    if i in main_list:
                        ranking[i] +=1

                ranking[m] = ranking[c1] + ranking[c2]
                merging_list = []
                
            counter -=1
                
        # If no merging has happened or length of cluster = 1, return the object with the highest score
        if merging == 0:
            obj_data = sorted(main_list, key=lambda x: x.score, reverse=True)[0]
            obj_data.construct_3d_box_carla()
            return obj_data
        
        # Returning the merged object with the highest ranking
        else :
            multi_merged = sorted(ranking, key=lambda x: ranking[x], reverse=True)[0]
            return multi_merged


    def update_lookup(self, gt_actors):
        '''
        Update the lookup table with the track ids of the actors in the vicinity of the agent'''

        for id in self.track_id_lookup.keys():
            if id not in gt_actors.keys():
                self.track_id_lookup.pop(id)

    def get_track_ids(self, merged_obj, gt_actors):
        '''
        Gets the track id for the merged box for evaluation purposes
        By checking the distance between the center of the merged box and ground truth boxes 
        Taking the closest ground truth box as the true label for the merged box
        
        args - merged object and gt_actors in the vicinity of the agents
        returns - track id for the merged box
        '''
        
        # Lookup table contains only the track ids of the actors in the vicinity of the agent
        # It is constantly updated every frame to avoid reusing track_ids of actors previously detected
        self.update_lookup(gt_actors)
        min_dist = 10000

        
        # Getting min distance and track id of the closest gt actor for the merged object
        for actor, info in gt_actors.items():
            obj_data = info['obj_data']
            if obj_data.obj_cat != merged_obj.obj_cat:
                continue
            # X,y,z in world Carla coordinates - x - forward, y - left, z - up
            gt_center = np.array([obj_data.x_wc, obj_data.y_wc, obj_data.z_wc])
            merged_center = np.array([merged_obj.x_wc, merged_obj.y_wc, merged_obj.z_wc])
            dist = np.linalg.norm(gt_center - merged_center)
            
            if dist < min_dist and actor not in self.actor_taken_list :
                min_dist = dist
                id = actor
        
        # Car
        if merged_obj.obj_cat == 'Car':
            
            if min_dist < 0.75 :

                self.actor_taken_list.append(id)
                if id in self.track_id_lookup:
                    index = self.track_id_lookup[id]
                else:
                    self.track_id_lookup[id] = self.obj_ind
                    self.obj_ind += 1
                    index = self.track_id_lookup[id]
            
            else:
                index = self.obj_ind
                self.obj_ind += 1
        
        # Pedestrian
        else:
            if min_dist < 0.25 :
                
                self.actor_taken_list.append(id)
                if id in self.track_id_lookup:
                    index = self.track_id_lookup[id]
                else:
                    self.track_id_lookup[id]  = self.obj_ind
                    self.obj_ind += 1
                    index = self.track_id_lookup[id]
            else:
                index = self.obj_ind
                self.obj_ind += 1
            
        return index
          
    def merge_boxes(self, cluster_boxes,frame, path, seq_num ,gt_actors):
        '''
        Merge the boxes in the cluster and get the track id for the merged box
        args - frame, path to store the ground truth data, sequence number and gt_actors in the vicinity of the agent
        Return - merged_boxes - dictionary containing the merged boxes'''
        
        merged_boxes = {}
        
        # Creating directory to store the ground truth data sequence wise
        p = os.path.join(path, 'merged_data')
        if not os.path.exists(p):
            os.makedirs(p)
        
        f = open(p + '/%04d' % seq_num + '.txt', 'a')
        self.actor_taken_list = []
        ids = {}
        
        for obj_ind, obj_cluster in cluster_boxes.items() :
            
            # Check for the overlap between the boxes in the cluster
            merged_obj = self.hierarchical_iou(obj_cluster)
            merged_obj.round_off(6)
            label_3d_det = Merger.get_label(merged_obj)

            # Get the track id for the merged box as GT for evaluation
            track_id = self.get_track_ids(merged_obj, gt_actors)

            # Label 3d det contains the label for the merged box in Kitti World Coordinates
            merged_boxes[track_id] = merged_obj

            # Label Format for the 3D MOT and Evaluation - Creating GT labels
            # Format - frame, track_id, type, truncated, occluded, 
            # alpha, bbox_left, bbox_top, bbox_right, bbox_bottom, 
            # height, width, length, 
            # x, y, z, 
            # rotation_y
            # TODO - check if score is needed, if needed it needs to be added

            f.write(str(frame) + ' ' + str(track_id) + ' ' + str(label_3d_det[1]) + ' ' + str('0') + ' ' + str('0') + ' ' + 
                str(label_3d_det[14]) + ' ' + str('0.0') + ' ' + str('0.0') + ' ' + str('0.0') + ' ' + str('0.0') + ' ' + 
                str(label_3d_det[7]) + ' ' + str(label_3d_det[8]) + ' ' + str(label_3d_det[9]) + ' ' + 
                str(label_3d_det[10]) + ' ' + str(label_3d_det[11]) + ' ' + str(label_3d_det[12]) + ' ' + 
                str(label_3d_det[13]) + ' ' + str(label_3d_det[6]) + '\n')

            ids[obj_ind] = track_id

        return ids, merged_boxes

if __name__ == '__main__':
    
    with open('config.yaml') as file:
        config = yaml.safe_load(file)

    merger = Merger(config)