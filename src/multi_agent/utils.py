# Modified code from Carla Simulator
import numpy as np


def convert_to_world(coords, cam_transform):
    '''
    args - coords: 3D coordinates in the camera frame - (3 + 1,n)
           cam_transform: Transform of the camera   (4 , 4)
    returns - 3D coordinates in the world frame ()
    in Carla Systems LHS, x - forward, y - right, z - up'''

    sensor_world_matrix = get_matrix(cam_transform)     
    world_coords = np.dot(sensor_world_matrix, coords)
    world_coords = np.array(world_coords)
    return world_coords

def get_matrix(transform):
    '''
    args - transform: Transform of the camera
    Returns a matrix from a transform
    shape: 4x4'''

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

def cam_calib(config):
    '''
    args - config: Config file
    Returns the camera calibration matrix'''

    calibration = np.identity(3)
    calibration[0, 2] = config['camera']['width'] / 2.0
    calibration[1, 2] = config['camera']['height'] / 2.0
    calibration[0, 0] = calibration[1, 1] = config['camera']['width']/ \
        (2.0 * np.tan(config['camera']['fov'] * np.pi / 360.0))
    return calibration





