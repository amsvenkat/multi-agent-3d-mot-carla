#!/usr/bin/env python
# Code by Amrutha Venkatesan
import sys
import glob
import itertools
import math
import os
import sys
import numpy as np

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc
import weakref
import random
import time
import argparse
import pygame

from viz import BoxData, compute_box_3d
from utils import convert_to_world, cam_calib
from oracle import Overlap, get_config
from plot import Plotter
from control_pygame import PygameFunctions
from detector import DetectionModel
from tracker import Tracker


def get_corrected_roty(ry_wc):

    if ry_wc > math.radians(180):
        ry_wc = ry_wc - math.radians(360) 
    elif ry_wc < -math.radians(180) :
        ry_wc = ry_wc + math.radians(360)

    return ry_wc

class CarlaClient():

    def __init__(self, config):
        self.config = config
        self.host = self.config['carla']['host']
        self.port = self.config['carla']['port']
        self.timeout = self.config['carla']['timeout']
        self.num_agents = self.config['num_agents']
        self.agent_threshold = self.config['agent_threshold']
        self.width = self.config['camera']['width']
        self.height = self.config['camera']['height']
        self.seq_num = self.config['seq_num']

        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model = None
        self.cam_calib = cam_calib(self.config)

        self.control = None
        self.main_agent = None
        self.main_image = None
        self.loc_hist = []
        self.tracker = {}

        self.agent_list = {}
        self.cam_list = {}
        self.cam_collect = {}
        self.cam_final = []

        self.agent_list_hist = {}
        self.flag = False
        self.capture = False
        self.frame_count = 0

        self.rgb_bev = False
        self.bev_cam = None
        self.tracker = None
        self.plotter = None
        self.single_agent_tracker = None

    @property
    def cam_transform(self):
        ''' 
        General transform of a camera as per Kitti Setup'''

        return carla.Transform(carla.Location(x=0, z=1.65), carla.Rotation(pitch=0))

    @property
    def cam_blueprint(self):
        '''
        General blueprint of a camera with Kitti format attributes'''
        
        cam_bp = self.world.get_blueprint_library().find(self.config['camera_type'])
        cam_bp.set_attribute('image_size_x', str(self.width))
        cam_bp.set_attribute('image_size_y', str(self.height))
        cam_bp.set_attribute('fov', str(self.config['camera']['fov']))
        cam_bp.set_attribute('sensor_tick', str(self.config['mode']['delta']))
        return cam_bp

    def set_synchronous_mode(self, mode):
        '''
        Set synchronous mode for the simulation'''
        
        settings = self.world.get_settings()
        settings.synchronous_mode = mode
        # settings.max_substep_delta_time = 0.01
        # settings.max_substeps = 10
        settings.fixed_delta_seconds = self.config['mode']['delta']
        self.world.apply_settings(settings)

    def agent_blueprint(self, filter):
        '''
        General blueprint of a vehicle'''
        
        agent_bp = self.world.get_blueprint_library().find(filter)
        agent_bp.set_attribute('role_name', 'autopilot')
        return agent_bp

    def spawn_agent(self, agent_bp, transform):
        ''' 
        Spawn a vehicle'''
        
        agent = self.world.spawn_actor(agent_bp, transform)
        self.agent_list[agent.id] = agent
        return agent

        
    def spawn_camera(self, attach_to):
        ''' 
        Spawn a camera and attach to a vehicle
        '''

        cam = self.world.spawn_actor(self.cam_blueprint, self.cam_transform, attach_to=attach_to)
        cam.calibration = self.cam_calib
        c = True if attach_to == self.main_agent else False
        weak_self = weakref.ref(self)
        cam.listen(lambda data: weak_self().collect_image(
            weak_self, data, attach_to.id, cam.id, c))
        self.cam_list[attach_to.id] = (cam.id, cam)
        return cam


    def spawn_bev_camera(self, attach_to):
        '''
        Spawn a bird's eye view camera'''

        try :
            cambp = self.world.get_blueprint_library().find(self.config['camera_type'])
            cambp.set_attribute('image_size_x', str(800))
            cambp.set_attribute('image_size_y', str(600))
            cambp.set_attribute('fov', str(self.config['camera']['fov']))
            cambp.set_attribute('sensor_tick', str(self.config['mode']['delta']))
            t= carla.Transform(carla.Location(x=0.5, z=3), carla.Rotation(pitch=-40))
            self.bev_cam =  self.world.spawn_actor(cambp, t, attach_to=attach_to)
            #tcam.calibration = self.cam_calib  
            weak_self = weakref.ref(self)
            self.bev_cam.listen(lambda data: weak_self().collect_bev(weak_self, data))
            #tcam.listen(lambda image: image.save_to_disk('TEST.png' % image.raw_data))

        except :
            print("Error in spawning camera")
  
    @staticmethod
    def collect_bev(weak_self, img):
        '''
        Called to collect bev image from each camera sensor'''
        
        self = weak_self()
        image_count = self.world.get_snapshot().frame

        if self.rgb_bev:
            
            img = np.array(img.raw_data)
            img = img.reshape((800, 600, 4))
            img = img[:, :, :3]
            # if self.frame_count % 5 == 0:
            #     cv2.imwrite('figures/bev_video/' + str(image_count) + '.png', img) 
            self.rgb_bev = False 

    def collect_loc(self):
        '''
        Collect the location of each camera sensor'''

        self.loc_hist = []
        for agent_id, (cam_id, cam) in self.cam_list.items():
            self.loc_hist.append(cam.get_transform())

    @staticmethod
    def collect_image(weak_self, image, agent_id, cam_id, c):
        '''
        Called to collect image from each of the camera sensor'''
        
        self = weak_self()
        if c:
            self.main_image = image

        if self.capture:
            self.cam_collect[agent_id] = (cam_id, image)

            if len(self.cam_collect) == len(self.agent_list):
                self.image_process()
                self.capture = False

    def image_process(self):
        '''
        Convert the raw image to numpy array and append to dictionary for each camera sensor'''
        
        self.cam_final = []

        for agent_id, (cam_id, image) in self.cam_collect.items():
            img = image
            frame = img.frame
            img = np.array(img.raw_data)
            img = img.reshape((self.height, self.width, 4))
            img = img[:, :, :3]
            c = {'image': img, 'frame': frame,
                 'cam_id': cam_id, 'agent_id': agent_id}
            self.cam_final.append(c)
        self.cam_collect = {}
        self.flag = True

    def check_dist(self) :
        '''
        Check the distance of the agents from the main agent and destroy the agents which are far away'''
        
        #TODO - Handle for key error, if no agents are present
        agents = {k: v for (k, v) in self.agent_list.items()
                  if k != self.main_agent.id}
        for agent_id, agent in agents.items():
            if agent.get_location().distance(self.main_agent.get_location()) > self.config['agent_threshold']:
                self.agent_list_hist[agent_id] = agent
                cam_id, cam = self.cam_list[agent_id]
                self.agent_list.pop(agent_id)
                self.cam_list.pop(agent_id)
                cam.destroy()

    def get_nearest_agent(self):
        '''
        Constantly check the list of nearest agents and 
        spawn a new agent if the number of agents is less than the required number of agents'''

        if len(self.agent_list) < self.num_agents:
            present_actors = self.world.get_actors().filter('vehicle.*')
            dist = []
            main = [self.main_agent]
            present_actors = [
                x for x in present_actors if x.id != self.main_agent.id]
            
            # TODO - for the case when there are no actors in the world
            if len(present_actors) == 0:
                return None
            
            for a, b in itertools.product(main, present_actors):
                if a.get_transform().rotation.yaw - b.get_transform().rotation.yaw == 0 and a.get_location().distance(b.get_location()) < 10 :
                    continue
                dist.append(
                    [a, b, a.get_location().distance(b.get_location())])
            dist.sort(key=lambda x: x[2])

            i = 0
            while len(self.agent_list) < self.num_agents:
                if dist[i][2] > self.config['agent_threshold']:
                    break
                actor = (dist[i][1])
                self.agent_list[actor.id] = actor
                self.spawn_camera(actor)
                i += 1

    def get_detection_data(self, detections):
        '''
        Processes all the agents detections and returns a dictionary with key as agent id and value as a list of detections
        args : predictions is a dictionary with key as agent id and value as a list of detections
        Predictions are in the Kitti format
        Converting to Carla format with world coordinates values 
        
        Box retrieved in Kitti predictions
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
        
        '''

        if len(detections) == 0:
            print('No predictions available')
            return None

        # return all the info - bounding boxes, camera transform, camera calibration, info for each agent
        agents_dets = {}
        
        for agent_id, dets in detections.items():
            obj_preds = []

            # if len(dets) == 0:
            #     agents_dets[agent_id] = obj_preds
            
            # to handle the case when detections are present but agent has been destroyed
            try:
                cam_transform = self.cam_list[agent_id][1].get_transform()
            except:
                print('camera already destroyed')
                continue
            
            for i , det in enumerate(dets):
                
                if det[15] < self.config['score_threshold']:
                    continue
                
                # Getting the bounding box coordinates in the camera frame in kitti Format - 3d in the shape (8,3)
                bbox_img, bbox_cam = compute_box_3d(det, self.cam_calib)
                # print('score', pred[15])

                if bbox_img is None:
                    print(" Object not in front of camera")
                    continue
                
                color = (0, 255, 0) if det[0] == 'Car' else (0, 0, 255)
                if agent_id == self.main_agent.id:
                    bb_surface = pygame.Surface((self.width, self.height))
                    bb_surface.set_colorkey((0, 0, 0))
                    # Draw bounding boxes using the Kitti format only
                    self.control.draw_bounding_boxes(self.display, bbox_img, bb_surface, color)

                obj_data = BoxData()
                t = np.ones([8, 1])

                # Converting the bbox coordinates to Carla format with shape 8,3 from Kitti format
                bbox_cam = np.transpose(np.vstack([bbox_cam[:, 2], bbox_cam[:, 0], -bbox_cam[:, 1], t[:, 0]]))
                
                # bbox shape to 4,8  = (4,num_points) = ((x,y,z,1) * num_points)
                bbox_cam = np.transpose(bbox_cam[: , :])
                
                # points in world coordinates - shape 4,8
                bbox_pts_wc = np.transpose(convert_to_world(bbox_cam, cam_transform))
                
                # get x,y,z in carla coordinate system
                l = [det[13], det[11], -det[12], 1]
                
                bbox_loc_cam = np.array(l)
                bbox_loc_wc = convert_to_world(bbox_loc_cam, cam_transform)               

                obj_data.obj_cat = det[0]
                obj_data.dim_h = det[8]
                obj_data.dim_w = det[9]
                obj_data.dim_l = det[10]
                # print('obj cat', obj_data.obj_cat)
                
                # Standard Carla Coordinates
                obj_data.x_wc = bbox_loc_wc[0][0]
                obj_data.y_wc = bbox_loc_wc[0][1]
                obj_data.z_wc = bbox_loc_wc[0][2]

                ry_wc = math.radians(cam_transform.rotation.yaw ) + det[14]

                # Get the 8 points and 4 points in Standara Carla World Coordinates
                obj_data.bbox_4wc = bbox_pts_wc[:4, :3]
                obj_data.bbox_8wc = bbox_pts_wc[:, :3]

                obj_data.alpha = det[3]
                obj_data.ry_wc= get_corrected_roty(ry_wc)
                obj_data.score = det[15]

                obj_data.get_kitti()
                
                obj_preds.append(obj_data)

            agents_dets[agent_id] = obj_preds

        return agents_dets
    
    def get_box_data(self, actor):
        ''' 
        Process the bounding box data
        args : Carla actor object
        returns: BoxData object of the actor
        
        Bbox points are in this from the get_world_vertices()
            7 -------- 3
           /|         /|
          5 -------- 1 .
          | |        | |
          . 6 -------- 2
          |/         |/
          4 -------- 0
        
        Convert to Bbox points in the Carla World Coordinates format(Standard format)

            6 -------- 7
           /|         /|
          5 -------- 4 .
          | |        | |
          . 2 -------- 3
          |/         |/
          1 -------- 0
          
        '''
        
        conv = {'vehicle': 'Car', 'walker': 'Pedestrian'}
        final_box = BoxData()
        
        new_loc = np.zeros((8, 4))
        l= actor.bounding_box.get_world_vertices(actor.get_transform())

        # Converting the bbox coordinates to Standar Carla format with shape 8,3
        new_loc[0, :] = [l[0].x, l[0].y, l[0].z, 1]
        new_loc[1, :] = [l[4].x, l[4].y, l[4].z, 1]
        new_loc[2, :] = [l[6].x, l[6].y, l[6].z, 1]
        new_loc[3, :] = [l[2].x, l[2].y, l[2].z, 1]
        new_loc[4, :] = [l[1].x, l[1].y, l[1].z, 1]
        new_loc[5, :] = [l[5].x, l[5].y, l[5].z, 1]
        new_loc[6, :] = [l[7].x, l[7].y, l[7].z, 1]
        new_loc[7, :] = [l[3].x, l[3].y, l[3].z, 1]

        dim1 = np.linalg.norm(new_loc[0, :3] - new_loc[1, :3])
        dim2 = np.linalg.norm(new_loc[1, :3] - new_loc[2, :3])
        dim3 = np.linalg.norm(new_loc[0, :3] - new_loc[4, :3])

        final_box.dim_h = dim3
        if dim1 > dim2:
            final_box.dim_w = dim2
            final_box.dim_l = dim1
        else:
            final_box.dim_w = dim1
            final_box.dim_l = dim2
            
        final_box.score = 0.0

        final_box.bbox_4wc = new_loc[:4, :3]
        final_box.bbox_8wc = new_loc[:, :3]
        final_box.obj_cat = conv[actor.type_id.split('.')[0]]
        
        # TODO To see if this is to be in Kitti corrected format of adding -90 to the yaw angle
        ry_wc = math.radians(actor.get_transform().rotation.yaw)
        final_box.ry_wc = get_corrected_roty(ry_wc)
        
        final_box.x_wc = actor.get_location().x
        final_box.y_wc = actor.get_location().y
        final_box.z_wc = actor.get_location().z

        # print(actor.get_location().x, actor.get_location().y, actor.get_location().z)

        final_box.get_kitti()
        final_box.round_off(6)
        
        return final_box

    def get_scene_actors(self, frame, seq):
        '''
        Returns all the actors in the world at the current time step within a max distance agent_threshold from each agent
        args : None
        returns : dict of all the actors and corresponding info in the world'''
        
        path = self.config['results_path']
        p = os.path.join(path, 'gt_data')
        if not os.path.exists(p):
            os.makedirs(p)
        
        present_actors = list(self.world.get_actors().filter('vehicle.*'))
        present_actors.extend(list(self.world.get_actors().filter('walker.*')))
        
        if present_actors == []:
            print('no actors intially')
            return {}
        
        with open(os.path.join(p + '/%04d' % seq + '.txt'), 'a') as f:
        
            final_actors = {}
            for agent in self.agent_list.values():
                
                for actor in present_actors:
                    if agent.get_location().distance(actor.get_location()) < self.config['agent_threshold'] and actor.id not in final_actors.keys():
                        flag = True if actor.id in self.agent_list.keys() else False
                        box = self.get_box_data(actor)
                        
                        f.write(str(frame) + ' ' + str(actor.id) + ' ' + str(box.obj_cat) + ' ' + str(0) + ' ' + str(0) + ' ' + 
                        str(box.alpha) + ' ' + str('0.0') + ' ' + str('0.0') + ' ' + str('0.0') + ' ' + str('0.0') + ' ' + 
                        str(box.dim_h) + ' ' + str(box.dim_w) + ' ' + str(box.dim_l) + ' ' + 
                        str(box.x_wc_kitti) + ' ' + str(box.y_wc_kitti) + ' ' + str(box.z_wc_kitti) +  ' ' +
                        str(box.ry_wc_kitti) + ' ' + str(box.score)  + '\n')
                        
                        final_actors[actor.id] = {'obj_data' : box, 'agent' : flag}

        return final_actors

    def initialize(self):
        '''
        Initialize the environment with the agents and the cameras
        Along with Tracker and Detection Model'''

        bp_vehicle = self.agent_blueprint(self.config['car'])
        transform = random.choice(self.world.get_map().get_spawn_points())
        
        print('transform of the main agent', [transform.location.x, transform.location.y, transform.location.z])
        
        try:
            self.main_agent = self.spawn_agent(bp_vehicle, transform)
        except:
            print('spawn collision')
            self.initialize()
        
        self.set_synchronous_mode(self.config['mode']['sync'])
        self.main_camera = self.spawn_camera(self.main_agent)
        # self.spawn_bev_camera(self.main_agent)
        self.control = PygameFunctions(self.main_agent)
        self.model = DetectionModel(self.config['model_path'], self.cam_calib)
        self.tracker = {'Car' : Tracker(self.config, cat='Car'), 'Pedestrian' : Tracker(self.config, cat='Pedestrian')}
        self.single_agent_tracker = {'Car' : Tracker(self.config, cat='Car'), 'Pedestrian' : Tracker(self.config, cat='Pedestrian') }

    def check_predictions(self, agents_detections):
        '''
        Checks if the detections are empty or not'''

        count = 0

        if agents_detections is not None:
        
            if len(agents_detections) == 0:
                return False

            for k, v in agents_detections.items():
                if v == []:
                    count += 1

            if count == len(agents_detections):
                return False
            
            return True

        else : 
            return False

    def main(self):

        try:
            pygame.init()
            time.sleep(0.03)
            pygame_clock = pygame.time.Clock()
            self.display = pygame.display.set_mode(
                (self.width, self.height), pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.initialize()
            # Fo results tracking
            ticking_frame=0
            obj_index = 0 
            object_classes = ['Car', 'Pedestrian']

            last_id = 0
            single_last_id = 0

            # Seq number for identifiying different sequences of tracking
            seq_num = self.seq_num
            plot = {}
            plot_v3 = {}

            # For recording the simulation
            record_path = self.config['record_path']
            
            if not os.path.exists(record_path):
                os.makedirs(record_path)
            
            # recording = record_path + '/recording_sceneTown02.log'
            # print('location of agent spawned', self.main_agent.get_location())

            # self.client.start_recorder(recording, True)
            # self.client.replay_file(recording, 0, 0, 0)

            self.main_agent.set_autopilot(True) 
            while True:
                self.world.tick()
                self.get_nearest_agent()
                
                if self.frame_count % 20 == 0 :
                    print('tick')
                    print('frame', self.world.get_snapshot().frame)
                    print('frame 2', self.frame_count)

                self.capture = True
                self.rgb_bev = True
                pygame_clock.tick_busy_loop(30)
                self.control.render(self.display, self.main_image)

                if not self.flag:
                    print('flag not set')
                    continue

                preds = self.model.main(self.cam_final)
                
                # Get all the info of every detection in world coordinates
                agents_detections = self.get_detection_data(preds)
                
                ids_merged, ids_tracked = {}, {} 
                tracked_boxes, merged_boxes ,ids = {}, {},{}

                # if self.frame_count == 0 or self.frame_count % 40 == 0:
                #     print('len of gt actors', len(gt_actors))
                #     plot = Plotter(gt_actors, object_classes)

                # For Tracking and Merged object data to be saved
                path = self.config['results_path']
                if not os.path.exists(path):
                    os.makedirs(path)
                
                # if agents_detections is not None :
                if self.check_predictions(agents_detections) :
                    # Get all the actors in the world including the agents informations
                    gt_actors = self.get_scene_actors(self.frame_count,seq_num)

                    if self.frame_count == 0 or self.frame_count % 30 == 0 or ticking_frame > 10 :
                    # if self.frame_count == 0 or self.frame_count % 50 == 0 :  #8 secs worth of frames in 1 plot
                        ticking_frame = 0
                        for i in object_classes:
                            # plot[i] = Plotter(gt_actors, i)
                            plot_v3[i] = Plotter(gt_actors, i)

                    merger = Overlap(agents_detections, self.config, obj_index)
                    cluster_boxes = merger.clustering()
                    ids, boxes = merger.merge_boxes(cluster_boxes, self.frame_count, path, seq_num, gt_actors)
                    
                    for i in object_classes:
                        
                        if self.main_agent.id in agents_detections.keys() and agents_detections[self.main_agent.id] != []:
                            single_agent_boxes = [ box for box in agents_detections[self.main_agent.id] if box.obj_cat == i]
                            # print('single agent')
                            self.single_agent_tracker[i].track(single_agent_boxes, self.frame_count, path, seq_num, single_last_id, single_agent=True)
                            single_last_id = self.single_agent_tracker[i].ID_count[0]
                        else:
                            self.single_agent_tracker[i].track([], self.frame_count, path, seq_num, last_id, single_agent=True)

                        # if self.main_agent.id in agents_detections.keys() and agents_detections[self.main_agent.id] != []:
                        #     single_agent_boxes = agents_detections[self.main_agent.id]
                        #     self.single_agent_tracker[i].track([], self.frame_count, path, seq_num, last_id, single_agent=True)
                        # print('multi agent')
                        merged_boxes[i] = [ box for k, box in boxes.items() if box.obj_cat == i]
                        ids_tracked[i], tracked_boxes[i]= self.tracker[i].track(merged_boxes[i], self.frame_count, path, seq_num, last_id)
                        last_id = self.tracker[i].ID_count[0]
                    
                        if self.frame_count % 5 == 0: #get frame for every second -
                            # plot[i].plot_boxes( gt_actors, self.agent_list, self.world, agents_detections, boxes, tracked_boxes) 
                            plot_v3[i].plot_boxes( gt_actors, self.agent_list, self.world, agents_detections, boxes, tracked_boxes) 
                    
                    self.frame_count += 1
                
                ticking_frame += 1

                self.check_dist()
                # changed from 0.9
                time.sleep(0.07)
                pygame.display.flip()
                pygame.event.pump()

                if self.control.control():
                    return

        finally:
            self.set_synchronous_mode(False)
            if self.main_agent is not None:
                self.main_agent.destroy()
                # self.bev_cam.destroy()
            for k, (cam_id, cam) in self.cam_list.items():
                cam.destroy()
            pygame.quit()
            # self.client.stop_recorder()


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(
        description='Multi_Agent_ObjectDetection')

    argparser.add_argument(
        '--config',
        default='config.yaml',
        metavar='FILE',
        type=str,
        help='config file path')

    argparser = argparser.parse_args()

    path = argparser.config

    config = get_config(path)

    try:
        client = CarlaClient(config)
        client.main()
    finally:
        print('EXIT')
