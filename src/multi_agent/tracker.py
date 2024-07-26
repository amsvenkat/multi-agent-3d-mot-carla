#!/usr/bin/env python
# Modified code from AB3DMOT
import copy
import numpy as np
import os
from AB3DMOT_libs.box import Box3D
from AB3DMOT_libs.matching import data_association
from AB3DMOT_libs.kalman_filter import KF
from viz import BoxData

def get_det_format(dets_all):
    '''
    Convert format of detections to AB3DMOT format
    args : All detections in the frame in the BoxData format
    returns : Dictionary of detections in the format required by AB3DMOT'''
    
    # get required and irrelevant information associated with an object, not used for associationg
    ori_array = []
    other_array = []
    dets=[]
    #print('dets_all', dets_all)
    for det in dets_all:
        ori_array.append([det.alpha])                    # orientation alpha 
                 
        other_info = [det.obj_cat, 0, 0, 0, 0, det.score]  
        other_array.append(other_info) 					# other information, e.g, | 1 - Type  |  2-5 2D BBOX (x1, y1, x2, y2) | Score 
        
        det_info = [det.dim_h, det.dim_w, det.dim_l, det.x_wc_kitti, det.y_wc_kitti, det.z_wc_kitti, det.ry_wc_kitti]
        dets.append(det_info)                            # 3D BBOX (h, w, l, x, y, z, theta)

    ori_array = np.array(ori_array)
    
    if dets == []:
        dets = np.zeros((0,7))
        additional_info = np.zeros((0,7))
    else:
        additional_info = np.concatenate((ori_array, other_array), axis=1)	

    dets_frame = {'dets': dets, 'info': additional_info}
    
    return dets_frame


class Tracker(object):
    '''
    This class represents the internal state of individual tracked objects observed as bbox.
    Each category is associated with an independent tracker object to handle category-specific information.
    '''
    def __init__(self ,config, cat, ID_init=0):
        self.config = config
        #hw = self.config['img_size']
        self.trackers = []
        self.cat = cat
        self.frame_count = 0
        self.ID_count = [ID_init]
        self.id_now_output = []
        self.debug_id = None
        #self.algm, self.metric, self.thres, self.min_hits, self.max_age = 'hungar', 'giou_3d', -0.2, 3, 2
        if self.cat == 'Car': 			self.algm, self.metric, self.thres, self.min_hits, self.max_age = 'hungar', 'giou_3d', -0.2, 2, 2
        elif self.cat == 'Pedestrian': 	self.algm, self.metric, self.thres, self.min_hits, self.max_age = 'greedy', 'dist_3d', 1, 2, 4
        # algm, metric, thres, min_hits, max_age = 'greedy', 'giou_3d', -0.4, 1, 4 	
        # 'greedy', 'giou_3d', -0.4, 3, 6


    def within_range(self, theta):
        # make sure the orientation is within a proper rang     
        if theta >= np.pi: theta -= np.pi * 2    # make the theta still in the range
        if theta < -np.pi: theta += np.pi *  2       
        
        return theta

    def orientation_correction(self, theta_pre, theta_obs):
        # update orientation in propagated tracks and detected boxes so that they are within 90 degree
        
        # make the theta still in the range
        theta_pre = self.within_range(theta_pre)
        theta_obs = self.within_range(theta_obs)    
        # if the angle of two theta is not acute angle, then make it acute
        
        if abs(theta_obs - theta_pre) > np.pi / 2.0 and abs(theta_obs - theta_pre) < np.pi * 3 / 2.0:     
            theta_pre += np.pi       
            theta_pre = self.within_range(theta_pre)

        # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
        if abs(theta_obs - theta_pre) >= np.pi * 3 / 2.0:
            if theta_obs > 0: theta_pre += np.pi * 2
            else: theta_pre -= np.pi * 2

        return theta_pre, theta_obs

    def process_dets(self, dets):
        # convert each detection into the class Box3D 
        # dets - a numpy array of detections in the format [[h,w,l,x,y,z,theta],...]

        dets_new = []
        for det in dets:
            #print(det)
            det_tmp = Box3D.array2bbox_raw(det)
            dets_new.append(det_tmp)
        return dets_new

    def prediction(self):
        # get predicted locations from existing tracks

        trks = []
        for t in range(len(self.trackers)):
        
            # propagate locations
            kf_tmp = self.trackers[t]
            if kf_tmp.id == self.debug_id:
                print('\n before prediction')
                print(kf_tmp.kf.x.reshape((-1)))
                print('\n current velocity')
                print(kf_tmp.get_velocity())
            kf_tmp.kf.predict()
            if kf_tmp.id == self.debug_id:
                print('After prediction')
                print(kf_tmp.kf.x.reshape((-1)))
            kf_tmp.kf.x[3] = self.within_range(kf_tmp.kf.x[3])  
            # update statistics
            kf_tmp.time_since_update += 1 		
            trk_tmp = kf_tmp.kf.x.reshape((-1))[:7]
            trks.append(Box3D.array2bbox(trk_tmp))

        return trks

    def update(self, matched, unmatched_trks, dets, info):
        # update matched trackers with assigned detections
        
        dets = copy.copy(dets)
        for t, trk in enumerate(self.trackers):
            if t not in unmatched_trks:
                d = matched[np.where(matched[:, 1] == t)[0], 0]     # a list of index
                assert len(d) == 1, 'error'  
                # update statistics
                trk.time_since_update = 0		# reset because just updated
                trk.hits +=    1     
                # update orientation in propagated tracks and detected boxes so that they are within 90 degree
                bbox3d = Box3D.bbox2array(dets[d[0]])
                trk.kf.x[3], bbox3d[3] = self.orientation_correction(trk.kf.x[3], bbox3d[3]) 
                if trk.id == self.debug_id:
                    print('After ego-compoensation')
                    print(trk.kf.x.reshape((-1)))
                    print('matched measurement')
                    print(bbox3d.reshape((-1)))
                    # print('uncertainty')
                    # print(trk.kf.P)
                    # print('measurement noise')
                    # print(trk.kf.R        
                # kalman filter update with observation
                trk.kf.update(bbox3d)        
                if trk.id == self.debug_id:
                    print('after matching')
                    print(trk.kf.x.reshape((-1)))
                    print('\n current velocity')
                    print(trk.get_velocity())     
                trk.kf.x[3] = self.within_range(trk.kf.x[3])
                trk.info = info[d, :][0]
            # debug use only
            # else:
                # print('track ID %d is not matched' % trk.id)   
    
    def output(self):

        num_trks = len(self.trackers)
        results = []
        
        for trk in reversed(self.trackers):
            # change format from [x,y,z,theta,l,w,h] to [h,w,l,x,y,z,theta]
            #print('track estmate',trk.kf.x)
            d = Box3D.array2bbox(trk.kf.x[:7].reshape((7, )))     # bbox location self
            d = Box3D.bbox2array_raw(d)   

            if ((trk.time_since_update < self.max_age) and 
                (trk.hits >= self.min_hits )): #or self.frame_count <= self.min_hits)):      
                results.append(np.concatenate((d, [trk.id], trk.info)).reshape(1, -1)) 		
            num_trks -= 1   
            
            # Results - bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry, track_id, 
            # info ( 1 - Type  |  2-5 2D BBOX (x1, y1, x2, y2) | Score )
            # deadth, remove dead tracklet

            if (trk.time_since_update >= self.max_age): 
                self.trackers.pop(num_trks)
    
        return results

    def birth(self, dets, info, unmatched_dets):
        new_id_list = list()					# new ID generated for unmatched detections
        
        for i in unmatched_dets:        			# a scalar of index
            trk = KF(Box3D.bbox2array(dets[i]), info[i, :], self.ID_count[0])
            # trk = KF(Box3D.bbox2array(dets[i]), info[i, :], track_ID)
            self.trackers.append(trk)
            new_id_list.append(trk.id)
            # print('track ID %s has been initialized due to new detection' % trk.id        
            self.ID_count[0] += 1
            # track_ID += 1
       
        return new_id_list
    
    def track(self, merged_boxes, frame, path, seq_num, track_ID, single_agent=False):

        '''
        Tracking of the objects in the frame
        args : merged_boxes , frame, path, seq_num'''

        if single_agent:
            t = 'singleagent'
        else:    
            t = 'multiagent'

        p = os.path.join(path, 'tracking_data', t)
        if not os.path.exists(p):
            os.makedirs(p)

        self.ID_count[0] = track_ID
        # get all the dets for the frame in the format [[h,w,l,x,y,z,theta],...] for the current category
        #merged_boxes_cat = { k : v for k, v in merged_boxes.items() if v[0][1] == self.cat}
        dets_all = get_det_format(merged_boxes)
        dets, info = dets_all['dets'], dets_all['info'] 
        
        # Get detections
        dets = self.process_dets(dets)
        
        self.id_past_output = copy.copy(self.id_now_output)
        self.id_past = [trk.id for trk in self.trackers]
        
        # Get tracks prediction
        trks = self.prediction()
        
        trk_innovation_matrix = np.zeros((len(trks), len(dets)))
        matched, unmatched_dets, unmatched_trks, cost, affi = \
        data_association(dets, trks, self.metric, self.thres, self.algm, trk_innovation_matrix)
        
        self.update(matched, unmatched_trks, dets, info)
        new_id_list = self.birth(dets, info, unmatched_dets)
        
        results = self.output()
        # Results - bbox.h, bbox.w, bbox.l, bbox.x, bbox.y, bbox.z, bbox.ry, track_id, info (| orientation | 1 - Type  |  2-5 2D BBOX (x1, y1, x2, y2) | Score  )
       
        # f = open( p + '/' + t + '_' + '%04d' % seq_num + '.txt', 'a')
        f = open( p + '/' + '%04d' % seq_num + '.txt', 'a')

        tracked_boxes = {}
        ids = []

        if len(results) > 0: 	# h,w,l,x,y,z,theta, ID, other info, confidence
            # print('results len greater', results)
            results = np.concatenate(results)	
            self.id_now_output = results[:, 7].tolist()					# only the active tracks that are outputed  
            
            for result in results:
                result_box = BoxData()
                #print(type(result[0]))
                result_box.obj_cat = result[9]
                result_box.dim_h = float(result[0])
                result_box.dim_w = float(result[1])
                result_box.dim_l = float(result[2])

                result_box.x_wc_kitti = float(result[3])
                result_box.y_wc_kitti = float(result[4])
                result_box.z_wc_kitti = float(result[5])

                result_box.ry_wc_kitti = float(result[6])
                result_box.score = float(result[-1])
                result_box.alpha = float(result[8])

                result_box.get_carla()
                result_box.construct_3d_box_carla()
                result_box.get_kitti()

                id = int(result[7])
                tracked_boxes[id] = result_box

                result_box.round_off(6)

                # Tracking results -
                # Format - frame, track_id, type, truncated, occluded, 
                # alpha, bbox_left, bbox_top, bbox_right, bbox_bottom, 
                # height, width, length, 
                # x, y, z
                # rotation_y, score
                # Make a list of track IDS generated in this frame
                ids.append(id)
                f.write(str(frame) + ' ' + str(id) + ' ' + str(result_box.obj_cat) + ' ' + str(0) + ' ' + str(0) + ' ' + 
                    str(result_box.alpha) + ' ' + str('0.0') + ' ' + str('0.0') + ' ' + str('0.0') + ' ' + str('0.0') + ' ' + 
                    str(result_box.dim_h) + ' ' + str(result_box.dim_w) + ' ' + str(result_box.dim_l) + ' ' + 
                    str(result_box.x_wc_kitti) + ' ' + str(result_box.y_wc_kitti) + ' ' + str(result_box.z_wc_kitti) +  ' ' +
                    str(result_box.ry_wc_kitti) + ' ' + str(result_box.score)  + '\n')

        else: 
            
            # print('results', results)
            results = np.empty((0, 15))
            self.id_now_output = results[:, 7].tolist()					# only the active tracks that are outputed
            f.write(str(frame) + ' ' + str(-99) + ' ' + str('DontCare') + ' ' + str(0) + ' ' + str(0) + ' ' + 
                str(0.0) + ' ' + str('0.0') + ' ' + str('0.0') + ' ' + str('0.0') + ' ' + str('0.0') + ' ' + 
                str(0.0) + ' ' + str(0.0) + ' ' + str(0.0) + ' ' + 
                str(0.0) + ' ' + str(0.0) + ' ' + str(0.0) +  ' ' +
                str(0.0) + ' ' + str(0.0)  + '\n')
            return ids, tracked_boxes
        
        return ids, tracked_boxes