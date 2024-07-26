#!/usr/bin/env python
# Modified code from Carla Simulator
'''
Generation of images and extracting 3D bounding box information from CARLA Simulator for 3D Monocular Object Detection

Ego Vehicle can be controlled using keyboard commands:
Use WASD keys for control.
    W            : throttle
    S            : brake
    AD           : steer
    Space        : hand-brake
    P            : autopilot mode
    C            : Capture Data
    l            : Loop Capture Start
    L            : Loop Capture End
    ESC          : quit

'''

import glob
import os
import sys

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
import cv2
import time
import argparse
import textwrap
import math
import numpy as np

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_TAB
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_p
    from pygame.locals import K_c
    from pygame.locals import K_l
    from pygame.locals import K_t
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')


VIEW_WIDTH = 2484//2
VIEW_HEIGHT = 750//2
VIEW_FOV = 90
BB_COLOR = (248, 64, 24)
WBB_COLOR = (0, 0, 255)

vehicle_bbox_record = False
pedestrian_bbox_record = False
bbox_record = False
count = 0

rgb_info = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype='i')
seg_info = np.zeros((VIEW_HEIGHT, VIEW_WIDTH, 3), dtype='i')

# Creates Directory
def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    

    dir_rgb = '/export/amsvenkat/project/data/train_v5/image/'
    dir_seg = '/export/amsvenkat/project/data/train_v5/SegmentationImage/'
    dir_depth = '/export/amsvenkat/project/data/train_v5/DepthImage/'
    dir_rgb_bev = '/export/amsvenkat/project/data/train_v5/custom_data_bev/'
    dir_seg_bev = '/export/amsvenkat/project/data/train_v5/SegmentationImage_bev/'
    dir_pbbox = '/export/amsvenkat/project/data/train_v5/PedestrianBBox/'
    dir_vbbox = '/export/amsvenkat/project/data/train_v5/VehicleBBox/'
    dir_timestamp = '/export/amsvenkat/project/data/train_v5/info/'
    dir_wc = '/export/amsvenkat/project/data/train_v5/VehicleBBox_wc/'
    dir_wc_p = '/export/amsvenkat/project/data/train_v5/PedestrianBBox_wc/'
    
    if not os.path.exists(dir_rgb):
        os.makedirs(dir_rgb)
    if not os.path.exists(dir_seg):
        os.makedirs(dir_seg)
    if not os.path.exists(dir_depth):
        os.makedirs(dir_depth)
    if not os.path.exists(dir_rgb_bev):
        os.makedirs(dir_rgb_bev)
    if not os.path.exists(dir_seg_bev):
        os.makedirs(dir_seg_bev)
    if not os.path.exists(dir_pbbox):
        os.makedirs(dir_pbbox)
    if not os.path.exists(dir_vbbox):
        os.makedirs(dir_vbbox)
    if not os.path.exists(dir_timestamp):
        os.makedirs(dir_timestamp)
    if not os.path.exists(dir_wc):
        os.makedirs(dir_wc)
    if not os.path.exists(dir_wc_p):
        os.makedirs(dir_wc_p)



class ActorBoundingBoxes(object):
    '''
    This is a module responsible for extracting the 3D bounding box information of the actors in the scene in Carla
    And writing the information to a text file.

    '''
    def __init__(self, actors_list, camera, agent):
        self.camera = camera
        self.actor_list = actors_list
        self.agent = agent

    def get_loc_camcoord(self, actor ):
        '''
        Get the locations coordinates in the Kitti Camera coordinate system
        '''
        
        camera = self.camera
        l = actor.get_transform().location
        loc_world = np.array([l.x, l.y, l.z, 1 ])

        # gettign camera matrix
        world_to_camera = np.array(camera.get_transform().get_inverse_matrix())
        loc_cam = np.dot(world_to_camera,loc_world)
        
        # Unreal coordinate LHS to Kitti Camera coordinate RHS with axis = y , -z , x
        loc_cam = np.array([loc_cam[1],-loc_cam[2],loc_cam[0]])
        
        return loc_cam

    def relative_rot_y(self, actor, camera):
        c_rot = camera.get_transform().rotation.yaw
        v_rot = actor.get_transform().rotation.yaw
        
        #rot_y = math.pi * (v_rot- c_rot)/180
        
        # to get in radians for label format
        c_rot = math.radians(c_rot)
        v_rot = math.radians(v_rot)
        #period = 2 * math.pi
        
        # calculate angle difference, modulo to [0, 2*pi]
        # diff = (c_rot - v_rot + period / 2) % period - period / 2
        # if diff > np.pi:
        #     diff = diff - (2 * np.pi)  # shift (pi, 2*pi] to (-pi, 0]
        rot_y = (v_rot- c_rot)

        return rot_y

    def hwl(self, actor):
        '''
        Get the height, width and length of the actor from the extents of the bounding box
        Conversion of the bounding box extents in Unreal coordinate system LHS  Kitti format RHS
        '''
        extent = 2 * (actor.bounding_box.extent)
        l, w, h = extent.x, extent.y, extent.z

      # h, w, l = extent.z, extent.x, extent.y
        return [h, w, l]
    
    def get_alpha(self, actor):
        '''
        Get the alpha angle of the actor with respect to the camera'''

        player_transform =self.agent.get_transform()
        
        #heading direction of the agent
        forward_vector = player_transform.rotation.get_forward_vector()
        forward_vector_numpy = np.array([forward_vector.x, forward_vector.y, forward_vector.z])
        
        #location of agent
        vehicle_location = player_transform.location
        
        #location of actor
        actor_location = actor.get_transform().location
        
        #vector from ego agent to actor
        target_vector = actor_location - vehicle_location
        target_vector_numpy = np.array([target_vector.x, target_vector.y, target_vector.z])
        norm_target = np.linalg.norm(target_vector_numpy)

        #fix rounding errors of dot product (can only range between -1 and 1 for normalized vectors)
        dot_prod = np.dot(forward_vector_numpy, target_vector_numpy) / norm_target
        if dot_prod > 1:
            dot_prod = 1.0
        elif dot_prod < -1:
            dot_prod = -1.0

        # check https://github.com/traveller59/second.pytorch/issues/98
        # and https://arxiv.org/pdf/1904.12681.pdf (supplementary material), here theta = theta ray
        theta = math.degrees(math.acos(dot_prod))

        rot_agent = actor.get_transform().rotation.yaw
        rot_vehicle = player_transform.rotation.yaw
        # rotate by -90 to match kitti
        rel_angle = rot_agent - rot_vehicle - 90

        alpha = math.radians(rel_angle - theta)

        if alpha > math.pi:
            alpha = alpha - 2*math.pi
        elif alpha < - math.pi:
            alpha = alpha + 2*math.pi

        return alpha

    def get_kitti_data(self, cat):
        '''
        Get the kitti data format for the actor and write to a text file'''

        global bbox_record 
        
        if bbox_record == True:
            
            f = open(main_dir+str(cat)+'BBox/bbox'+str(count), 'w')
            f1 = open(main_dir+str(cat)+'BBox_wc/bbox'+str(count), 'w')
            
            for actor in self.actor_list:
                # filter out distance with 60m
                if self.agent.get_location().distance(actor.get_transform().location) < 60 :
                    
                    bb, wc = self.get_bounding_box(actor)

                    # filter objects behind camera , done in image space 
                    if all(bb[:, 2] > 0):

                        world_coords=wc

                        actor_transform = actor.get_transform()
                        cam_transform = self.camera.get_transform()
                        agent_transform = self.agent.get_transform()

                        # relative_rot_y.append(ActorBoundingBoxes.relative_rot_y(actor, camera))
                        obj_dims = self.hwl(actor)

                        actor_loc_camcord = self.get_loc_camcoord(actor)

                        # version1 
                        # Alpha in Carla = arctan(x/y) , Alpha in Kitti = arctan(z/x)

                        # version2
                        alpha = self.get_alpha(actor)

                        id= actor.id
                        dist =self.camera.get_location().distance(actor.get_transform().location)

                        bb_points_image = [(int(bb[i, 0]), int(bb[i, 1])) for i in range(8)]
                        bb_points_world = [(int(bb[i, 0]), int(bb[i, 1]),  int(bb[i, 2])) for i in range(8)]
                        
                        #[bounding_boxes, 1 world_coords, 2v_center, 3alpha, 4obj_3d, 5camloc_3d, 6relative_rot_y, 7id, 8dist, 9agent] 
                        f.write(str(bb_points_image)+'\n')

                        f1.write(str(bb_points_world) + '#' + str(actor_transform) + '#' + str(alpha) + '#'  #alpha
                        + str(obj_dims[0]) + ' ' + str(obj_dims[1]) + ' ' + str(obj_dims[2])  + '#'  #obj_3d
                        + str(actor_loc_camcord[0]) + ' ' + str(actor_loc_camcord[1]) + ' ' + str(actor_loc_camcord[2]) + '#'  #xyz
                        + str(cam_transform) + '#'  #camtrancform
                        + str(id) +  '#' + str(dist) +  '#' + str(agent_transform) +  '#' + '\n') #id, dist

            f.close()
            f1.close()

        if actor == 'Pedestrian':
            bbox_record = False

    def get_bounding_box(self, actor):
        '''
        Returns 3D bounding box points for a actor in the image plane.
        '''

        bb_cords = self._create_bb_points(actor)
        cords_x_y_z, world_cord = self._actor_to_sensor(bb_cords, actor)

        # chnaging from 4,8 to 3,8 - for 8 points
        cords_x_y_z =cords_x_y_z[:3, :]
        world_cord= world_cord[:3,:]

        # to get from the unreal system to standard for further process in camera
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        world_cord = np.transpose(world_cord)
        
        # project the 3D bounding box into the image plane
        bbox = np.transpose(np.dot(self.camera.calibration, cords_y_minus_z_x))
        img_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        
        return [img_bbox, world_cord]

    def _create_bb_points(self,actor):
        '''
        Returns 3D bounding box extents for a actor.
        '''
        #print(actor)
        cords = np.zeros((8, 4))
        extent = actor.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    def _actor_to_sensor(self, cords, actor):
        '''
        Transforms coordinates of a actor bounding box to sensor.
        '''
        sensor = self.camera
        world_cord = self._actor_to_world(cords, actor)
        sensor_cord = self._world_to_sensor(world_cord)

        return sensor_cord, world_cord

    def _actor_to_world(self, cords, actor):
        '''
        Transforms coordinates of a actor bounding box to world.
        '''

        bb_transform = carla.Transform(actor.bounding_box.location)
        bb_actor_matrix = self.get_matrix(bb_transform)
        actor_world_matrix = self.get_matrix(actor.get_transform())
        bb_world_matrix = np.dot(actor_world_matrix, bb_actor_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    def _world_to_sensor(self, cords):
        '''
        Transforms world coordinates to sensor.
        '''
        sensor = self.camera.get_transform()
        
        sensor_world_matrix = self.get_matrix(sensor)
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    def get_matrix(self, transform):
        '''
        Creates matrix from carla transform.
        '''
        # transform = self.camera.get_transform()
        
        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix

class BasicSynchronousClient(object):
    '''
    Basic implementation of a synchronous client to collect data from CARLA.'''

    def __init__(self):
        self.client = None
        self.world = None
        self.camera = None
        self.camera_bev = None
        self.camera_segmentation = None
        self.camera_depth = None
        self.camera_segmentation_bev = None
        self.car = None
        

        self.display = None
        self.image = None
        self.image_bev =None
        self.image_depth = None
        self.segmentation_image = None
        self.segmentation_image_bev = None

        self.capture = True
        self.capture_segmentation = True
        self.capture_bev = True
        self.capture_segmentation_bev = True
        self.capture_depth = True

        self.record = True
        self.seg_record = False
        self.rgb_record = False
        self.seg_record_bev = False
        self.rgb_record_bev = False
        self.depth_record = False

        self.screen_capture = 0 
        self.loop_state = False 
        self.bb = None

    def camera_blueprint(self, filter):
        '''
        Returns camera blueprint.
        '''
        camera_bp = self.world.get_blueprint_library().find(filter)
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        return camera_bp

    def set_synchronous_mode(self, synchronous_mode):
        '''
        Sets synchronous mode.'''

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

    def setup_car(self):
        '''
        Spawns ego-vehicle to be controlled.
        '''

        car_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        #location = carla.Transform(carla.Location(x=20, y=0.0, z=0), carla.Rotation(pitch=0, roll=0.0, yaw=0.0))
        #print(location)
        self.car = self.world.spawn_actor(car_bp, location)

    def setup_camera(self):
        '''
        Spawns RGB and Segementation Camera and sets the attributes.'''
    
        #camera_transform = carla.Transform(carla.Location(x=1.5, z=2.8), carla.Rotation(pitch=-15))
        camera_transform = carla.Transform(carla.Location(x=0, z=1.65), carla.Rotation(pitch=0))
        self.camera = self.world.spawn_actor(self.camera_blueprint('sensor.camera.rgb'), camera_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))

        seg_transform = carla.Transform(carla.Location(x=0, z=1.65), carla.Rotation(pitch=0))
        self.camera_segmentation = self.world.spawn_actor(self.camera_blueprint('sensor.camera.semantic_segmentation'), seg_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera_segmentation.listen(lambda image_seg: weak_self().set_segmentation(weak_self, image_seg))

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration
        self.camera_segmentation.calibration = calibration

    def setup_camera_bev(self):
        '''
        Spawns bev camera to render BEV space view.
        '''

        seg_transform_bev = carla.Transform(carla.Location(x=0, z=10), carla.Rotation(pitch=-90))
        self.camera_segmentation_bev = self.world.spawn_actor(self.camera_blueprint('sensor.camera.semantic_segmentation'), seg_transform_bev, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera_segmentation_bev.listen(lambda image_seg_bev: weak_self().set_segmentation_bev(weak_self, image_seg_bev))

        #camera_transform = carla.Transform(carla.Location(x=1.5, z=2.8), carla.Rotation(pitch=-15))
        camera_transform_bev = carla.Transform(carla.Location(x=0, z=10), carla.Rotation(pitch=-90))
        self.camera_bev = self.world.spawn_actor(self.camera_blueprint('sensor.camera.rgb'), camera_transform_bev, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera_bev.listen(lambda image_bev: weak_self().set_image_bev(weak_self, image_bev))

    def setup_camera_depth(self):
        '''
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        '''

        cam_depth_transform = carla.Transform(carla.Location(x=0, z=1.65), carla.Rotation(pitch=0))
        self.camera_depth= self.world.spawn_actor(self.camera_blueprint('sensor.camera.depth'), cam_depth_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera_depth.listen(lambda image_depth: weak_self().set_depth(weak_self, image_depth))

    def control(self, car):
        '''
        Applies control to main car based on pygame pressed keys.
        Will return True If ESCAPE is hit, otherwise False to end main loop.
        '''
        keys = pygame.key.get_pressed()

        if keys[K_ESCAPE]:
            return True

        control = car.get_control()
        control.throttle = 0
        if keys[K_w]:
            control.throttle = 1
            control.reverse = False
        elif keys[K_s]:
            control.throttle = 1
            control.reverse = True
        if keys[K_a]:
            control.steer = max(-1., min(control.steer - 0.05, 0))
        elif keys[K_d]:
            control.steer = min(1., max(control.steer + 0.05, 0))
        else:
            control.steer = 0
        if keys[K_p]:
            car.set_autopilot(True) 
            print('Auto pilot On')    
        if keys[K_t]:
            car.set_autopilot(False) 
            print('Auto pilot Off')      
        if keys[K_c]:
            self.screen_capture = self.screen_capture + 1
        else:
            self.screen_capture = 0

        if keys[K_l]:
            self.loop_state = True
            print('Loop Start') 

        if keys[K_l] and (pygame.key.get_mods() & pygame.KMOD_SHIFT):
            self.loop_state = False
            print('Loop End') 
            
        control.hand_brake = keys[K_SPACE]

        car.apply_control(control)
        return False

    @staticmethod
    def set_image(weak_self, img):
        '''
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        '''

        self = weak_self()
        if self.capture:
            self.image = img
            self.capture = False

        if self.rgb_record:
            
            # bounding_boxes, w_coords, v_center, relative_rot_y, obj_3d, loc_3d 
            vehicle_obj = ActorBoundingBoxes(self.world.get_actors().filter('vehicle.*'), self.camera, self.car)
            vehicle_obj.get_kitti_data('Vehicle')
            
            ped_obj = ActorBoundingBoxes(self.world.get_actors().filter('walker.pedestrian.*'), self.camera, self.car)
            ped_obj.get_kitti_data('Pedestrian')
            
            i = np.array(img.raw_data)
            i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
            i3 = i2[:, :, :3]
            cv2.imwrite(dir_rgb + str(self.image_count) + '.png', i3) 
            
            # holding general info of all data
            i4 = [img.timestamp, img.frame_number, img.width, img.height]
            i5 = self.world.get_snapshot().timestamp
            i6 = self.world.get_snapshot().platform_timestamp
            i7 = self.car.get_transform().location
            i8 = self.car.id
            
            print('RGB(custom)Image')
            self.rgb_record  = False

    @staticmethod
    def set_image_bev(weak_self, img):
        '''
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        '''
        self = weak_self()
        if self.capture_bev:
            self.image_bev = img
            self.capture_bev = False

        if self.rgb_record_bev:
            
            i = np.array(img.raw_data)
            i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
            i3 = i2[:, :, :3]

            cv2.imwrite(dir_rgb_bev + str(self.image_count) + '.png', i3) 
            
            print('RGB(bev)Image')
            self.rgb_record_bev  = False

    @staticmethod
    def set_segmentation(weak_self, img):
        '''
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        '''

        self = weak_self()
        if self.capture_segmentation:
            self.segmentation_image = img
            self.capture_segmentation = False

        if self.seg_record:
            img.convert(cc.CityScapesPalette)
            i = np.array(img.raw_data)
            i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
            i3 = i2[:, :, :3]
            cv2.imwrite(dir_seg + 'seg' + str(self.image_count) +'.png', i3)
            print('SegmentationImage')
            self.seg_record  = False

    @staticmethod
    def set_segmentation_bev(weak_self, img):
        '''
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        '''

        self = weak_self()
        if self.capture_segmentation_bev:
            self.segmentation_image_bev = img
            self.capture_segmentation_bev = False


        if self.seg_record_bev:
            img.convert(cc.CityScapesPalette)
            i = np.array(img.raw_data)
            i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
            i3 = i2[:, :, :3]
            cv2.imwrite(dir_seg_bev + 'seg_bev' + str(self.image_count) +'.png', i3)
            print('SegmentationBevImage')
            self.seg_record_bev  = False

    @staticmethod
    def set_depth(weak_self, img):
        '''
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        '''

        self = weak_self()
        if self.capture_depth:
            self.image_depth = img
            self.capture_depth = False

        if self.depth_record:
            
            #img.convert(cc.LogarithmicDepth)
            img.convert(carla.ColorConverter.Depth)
            # i = np.array(img.raw_data)
            # i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
            
            depth_meter = np.array(img.raw_data).reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))[:,:,0] * 1000 / 255

            cv2.imwrite(dir_depth + 'dm' + str(self.image_count) +'.png', depth_meter)
            np.save(dir_depth + 'dmgrey_{}'.format(str(self.image_count)), depth_meter)
            
            # x = np.load('/export/amsvenkat/project/data/train_v5/DepthImage/dm_{}'.format(str(self.image_count)))
            # print(x.shape)
            # f = open('/export/amsvenkat/project/data/train_v5/DepthImage/dm'+str(self.image_count) + '.txt', 'w')
            # f.write(str(i3_m))
            # f.close()

            print('DepthImage')

            self.depth_record  = False

    def render(self, display):
        '''
        Transforms image from camera sensor and blits it to main pygame display.
        '''

        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype('uint8'))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
      
            display.blit(surface, (0, 0))

    def change_traffic_light(self):
        # list_actor = self.world.get_actors()
        
        # if self.car.is_at_traffic_light():
        #    traffic_light = self.car.get_traffic_light()
           
        #    if traffic_light.get_state() == carla.TrafficLightState.Red:
        #        traffic_light.set_state(carla.TrafficLightState.Green)
        #        traffic_light.set_green_time(1000.0)
           
           # elif traffic_light.get_state() == carla.TrafficLightState.Green:
           #     traffic_light.set_state(carla.TrafficLightState.Red)
           #     traffic_light.set_red_time(1000.0)
            
        for actor_ in self.world.get_actors():   
            if isinstance(actor_, carla.TrafficLight):
                actor_.reset_group()
             # for any light, first set the light state, then set time. for yellow it is 
             # carla.TrafficLightState.Yellow and Red it is carla.TrafficLightState.Red
                # actor_.set_state(carla.TrafficLightState.Green) 
                # actor_.set_green_time(1000.0)
                # actor_.set_state(carla.TrafficLightState.Red) 
                # actor_.set_green_time(1000.0)
                # actor_.set_state(carla.TrafficLightState.Red) 
                # actor_.set_green_time(1000.0)
            # actor_.set_green_time(5000.0)
            # actor_.set_yellow_time(1000.0)

    def game_loop(self, args):
        '''
        Main program loop - handles user input events, gathers sensor data'''

        try:
            pygame.init()

            self.client = carla.Client('127.0.0.1', 2000)
            self.client.set_timeout(2.0)
            self.world = self.client.get_world()

            self.setup_car()
            self.setup_camera()
            #self.setup_camera_bev()
            self.setup_camera_depth()

            self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame_clock = pygame.time.Clock()

            self.set_synchronous_mode(True)
            self.image_count = 0
            self.time_interval = 0

            global vehicle_bbox_record
            global pedestrian_bbox_record
            global bbox_record
            global count

            # self.change_traffic_light()

            while True:
                self.world.tick()
                # self.change_traffic_light()

                self.capture = True
                pygame_clock.tick_busy_loop(30)

                self.render(self.display)
                self.car.set_autopilot(True)

                self.time_interval += 1
                if ((self.time_interval % args.CaptureLoop) == 0) and (self.loop_state):
                    self.image_count = self.image_count + 1 
                    self.rgb_record = True
                    self.seg_record = True
                    self.rgb_record_bev = True
                    self.seg_record_bev = True
                    self.depth_record = True
                    bbox_record = True
                    count = self.image_count
                    print('-------------------------------------------------')
                    print('ImageCount - %d' %self.image_count)

                if self.screen_capture == 1:
                    self.image_count = self.image_count + 1 
                    self.rgb_record = True
                    self.seg_record = True
                    self.rgb_record_bev = True
                    self.seg_record_bev = True
                    self.depth_record = True
                    bbox_record =True
                    count = self.image_count
                    print('-------------------------------------------------')
                    print('Captured! ImageCount - %d' %self.image_count)

                time.sleep(0.03)
                #self.rgb_record = False
                #self.seg_record = False
                pygame.display.flip()

                pygame.event.pump()
                if self.control(self.car):
                    return
                
                if self.image_count == 12000:
                    break

        finally:
            self.set_synchronous_mode(False)
            self.camera.destroy()
            self.camera_segmentation.destroy()
            self.car.destroy()
            # self.camera_bev.destroy()
            # self.camera_segmentation_bev.destroy()
            self.camera_depth.destroy()

            pygame.quit()

def main():

    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '-l', '--CaptureLoop',
        metavar='N',
        default=50,
        type=int,
        help='set Capture Cycle settings, Recommand : above 100')
    argparser.add_argument(
        '-p', '--path',
        default='./saved_data/',
        type=str,
        help='set path to save data')

    args = argparser.parse_args()

    print(__doc__)
    create_dir(args.path)

    try:
        client = BasicSynchronousClient()
        client.game_loop(args)
    finally:
        print('EXIT')

if __name__ == '__main__':
    
    main()
