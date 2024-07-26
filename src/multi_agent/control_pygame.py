#!/usr/bin/env python
# Modified code from Carla Simulator
import numpy as np
import pygame
from pygame import locals, surfarray
from pygame.locals import K_ESCAPE,K_SPACE, K_SPACE, K_a, K_d, K_s, K_w, K_TAB, K_BACKQUOTE, K_p, K_c, K_l, K_t

class PygameFunctions():
    def __init__(self, vehicle):
        self.vehicle = vehicle

    def control(self):
        keys = pygame.key.get_pressed()

        if keys[K_ESCAPE]:
            return True

        control = self.vehicle.get_control()
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
            self.vehicle.set_autopilot(True) 
            print("Auto pilot On")    
        
        if keys[K_t]:
            self.vehicle.set_autopilot(False) 
            print("Auto pilot Off")      
        
        if keys[K_c]:
            self.screen_capture = self.screen_capture + 1
        else:
            self.screen_capture = 0

        if keys[K_l]:
            self.loop_state = True
            print("Loop Start") 

        if keys[K_l] and (pygame.key.get_mods() & pygame.KMOD_SHIFT):
            self.loop_state = False
            print("Loop End") 
            
        control.hand_brake = keys[K_SPACE]
        self.vehicle.apply_control(control)
        
        return False

    def render(self, display, image):
        
        if image is not None:
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            surface = surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))

    def draw_bounding_boxes(self, display, vertices, bb_surface, colour):
        color= colour
        # bb_surface = pygame.Surface((width, height))
        # bb_surface.set_colorkey((0, 0, 0))

        for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i, j = k, (k + 1) % 4
            # use LINE_AA for opencv3
            p = (int(vertices[i][0]),  int(vertices[i][1]))
            q = (int(vertices[j][0]), int(vertices[j][1]))
            pygame.draw.line(bb_surface, color, p, q)
            i, j = k + 4, (k + 1) % 4 + 4
            pygame.draw.line(bb_surface, color , (int(vertices[i][0]),  int(vertices[i][1])), (int(vertices[j][0]), int(vertices[j][1])))
            i, j = k, k + 4
            pygame.draw.line(bb_surface,  color, (int(vertices[i][ 0]), int(vertices[i][1])), (int(vertices[j][0]), int(vertices[j][1])))

        display.blit(bb_surface, (0, 0))

