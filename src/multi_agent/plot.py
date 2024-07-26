#!/usr/bin/env python
# Code by Amrutha Venkatesan
import os
import random
import matplotlib.pyplot as plt
import numpy as np

def get_polygon(obj_box):
    '''
    Get the polygon of the object box for plotting
    args : object Coordinatesinates of shape (num_points, 3) - x,y,z
    returns : polygon of shape (5,2)
    '''
    
    polygon = np.zeros([5, 2], dtype=np.float32)

    for i in range(4):
        polygon[i, 0] = obj_box[i][0]
        polygon[i, 1] = obj_box[i][1]
    polygon[4, 0] = obj_box[0][0]
    polygon[4, 1] = obj_box[0][1]

    return polygon

def get_x_y(x, y, category):
    '''
    Get the max and min - x and y Coordinates for plotting
    args : list of x and y values
    returns : max and min values of x and y
    '''

    try:
        max_x = np.max(x)
        max_y = np.max(y)
        min_x = np.min(x)
        min_y = np.min(y)
    except ValueError:  #raised if `y` is empty.
        # print('x', x)
        # print('y', y)
        # returning default values for now
        return 0, 0, 5, 5

    
    del_y = max_y  - min_y
    del_x = max_x  - min_x
    del_s = np.max([del_y, del_x])
    
    gx = abs(del_s - del_x)
    gy = abs(del_s - del_y)

    if category == 'Pedestrian' :
    # For Pedestrian
        xmin = min_x - gx/2 - 4
        xmax = max_x + gx/2 + 4
        ymin = min_y - gy/2 - 4
        ymax = max_y + gy/2 + 4

    # For Car
    if category == 'Car' :
        xmin = min_x - gx/2 - 10
        xmax = max_x + gx/2 + 10
        ymin = min_y - gy/2 - 10
        ymax = max_y + gy/2 + 10

    # # TO have a minimum size of the plot
    # xmax_set = max(xmax, xmin + 20)
    # ymax_set = max(ymax, ymin + 20)

    return xmin, ymin, xmax, ymax

def get_max_min(actors):
    '''
    Get the max and min values of the x and y Coordinatesinates of all the actors in the scene
    args : list of actor bounding box information
    returns : max and min of x and y for the list of actors
    '''

    x = []
    y = []
    
    for actor in actors:
        
        data = actor['obj_data']
        
        bbox = data.bbox_4wc

        x1 = [i for i in bbox[:,0]]
        y1 = [i for i in bbox[:,1]]
        
        x.extend(x1)
        y.extend(y1)
    
    if len(x) == 0 or len(y) == 0:
        return 0, 0, 5, 5
   
    xy = get_x_y(x, y, data.obj_cat)

    return xy

class Plotter():
    
    def __init__(self, scene_actors, object_cat) -> None:
        '''
        Initialize the plotter with the scene actors and the object classes
        args : scene actors and the object classes
        '''
        
        self.scene_actors = scene_actors
        self.object_cat =object_cat
        self.plot_data = {}
        self.fig = None
        self.ax = None
        self.xy = {}
        self.get_plot_ready()
        
        self.label_agents = []
        self.label_dets = []
        self.label_gt_count = 0 
        self.label_merged = []
        self.label_tracklets =[]
        self.count =0
        self.plots ={}

        # TODO - verify color pattern
        color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(50)]
        cmap = plt.cm.get_cmap('hsv', 256)
        #print('cmap', len(cmap))
        cmap1 = np.array([cmap(i+5) for i in range(256)])[:, :3]
        cmap2 = np.array([cmap(i+5) for i in range(256)])[:, :3]
        np.random.shuffle(cmap1)
        np.random.shuffle(cmap2)
        self.color = color
        self.cmap1 = cmap1
        self.cmap2 = cmap2
        self.style = {'Car': '-','Pedestrian': '-',}
        # Upto 6 agents for now
        self.agent_colors = ['r' , 'c', 'm', 'y', 'k', 'w']

        
    def get_plot_ready(self): 
        '''
        Getting the figure and axes ready for plotting for every interval
        returns : figure and axes with the limits set
        '''
        
        # self.fig, self.ax = plt.subplots(2,2, sharex=True, sharey=True, figsize=(15, 15))
        
        # for i in self.object_classes:
            
        #     path = 'figures/{}'.format(i)
        #     if not os.path.exists(path):
        #         os.mkdir(path)

        #     # get xmin, ymin, xmax, ymax for the scene
        #     actors = [self.scene_actors.get(k) for k,v in self.scene_actors.items() if v['obj_data'].obj_cat == i]
        #     xy = get_max_min(actors)

        #     self.xy[i] = xy

        # for i in self.object_classes:
            
        #     path = 'figures/{}'.format(i)
        #     if not os.path.exists(path):
        #         os.mkdir(path)

        #     fig,  ax = plt.subplots(2,2, sharex=True, sharey=True, figsize=(15, 15))
            
        #     # get xmin, ymin, xmax, ymax for the scene
        #     actors = [self.scene_actors.get(k) for k,v in self.scene_actors.items() if v['obj_data'].obj_cat == i]
        #     xy = get_max_min(actors)

        #     ax[0, 0].set(xlim = (xy[0], xy[2]), ylim = (xy[1], xy[3]))
        #     ax[1, 0].set(xlim = (xy[0], xy[2]), ylim = (xy[1], xy[3]))
        
        #     ax[0, 0].set_aspect('equal')
        #     ax[1, 0].set_aspect('equal')

        #     self.plot_data[i] = {'fig': fig, 'ax': ax}
    
        path = '../../figures/{}'.format(self.object_cat)
        if not os.path.exists(path):
            os.mkdir(path)

        fig, ax = plt.subplots(1,3, sharex=True, sharey=True, figsize=(15, 10))
        
        # get xmin, ymin, xmax, ymax for the scene
        actors = [self.scene_actors.get(k) for k,v in self.scene_actors.items() if v['obj_data'].obj_cat == self.object_cat]
        xy = get_max_min(actors)

        # ax[0, 0].set(xlim = (xy[0], xy[2]), ylim = (xy[1], xy[3]))
        # ax[1, 0].set(xlim = (xy[0], xy[2]), ylim = (xy[1], xy[3]))
        # ax[1, 0].set(xlim = (xy[0], xy[2]), ylim = (xy[1], xy[3]))
        # ax[1, 1].set(xlim = (xy[0], xy[2]), ylim = (xy[1], xy[3]))

        ax[0].set(xlim = (xy[1], xy[3]), ylim = (xy[0], xy[2]))
        ax[1].set(xlim = (xy[1], xy[3]), ylim = (xy[0], xy[2]))
        ax[2].set(xlim = (xy[1], xy[3]), ylim = (xy[0], xy[2]))
        # ax[1, 1].set(xlim = (xy[1], xy[3]), ylim = (xy[0], xy[2]))
    
        ax[0].set_aspect('equal')
        ax[1].set_aspect('equal')
        ax[2].set_aspect('equal')
        # ax[1, 1].set_aspect('equal')

        self.fig = fig
        self.ax = ax  
        self.xy = xy  

    
    def plot_gt(self, gt_actors, agent_list):
        
        # ax1 = self.ax1[self.curr_cat][0]
        # ax1.set_title('GTs')
        # ax1 = self.plot_data[self.curr_cat]['ax'][0, 0]
        # ax1 = self.ax[0]
        self.ax[0].set_title('Ground Truth')
       
        self.label_gt_count = 0
        for k, v in gt_actors.items():
            s = self.style[v['obj_data'].obj_cat]
            polygon = get_polygon(v['obj_data'].bbox_4wc)
            #print('polu', polygon)

            if k in agent_list:
                color = self.agent_colors[self.agent_ind[k]]
                fill = 'full'
                label =  'Agent {}'.format(k) #if ( k not in self.label_agents) else '_'
                self.label_agents.append(k)

            else:
                color = 'g'
                fill = 'none'
                label = 'GT Actor' if self.label_gt_count == 0 else '_'
                self.label_gt_count += 1

            # Polygon x and y swappped as the coordinates are in (y, x) format in Carla System -
            self.plots[0] = self.ax[0].plot(polygon[0:5, 1], polygon[0:5, 0], color=color, linewidth=0.5, linestyle=s, label = label)

        self.ax[0].set_xlabel('World Coordinates - Y')
        self.ax[0].set_ylabel('World Coordinates - X')
        self.ax[0].legend()
    
    def plot_detections(self, detections):
        
        # ax2 = self.ax1[self.curr_cat][1]
        # ax2 = self.plot_data[self.curr_cat]['ax'][0, 1]
        # ax2 = self.ax[1]
        self.ax[1].set_title('Detections')
        self.label_dets = []

        for k, objs in detections.items():
            for obj in objs:
                if obj.obj_cat != self.object_cat:
                    continue
                
                s = self.style[obj.obj_cat]
                color = self.agent_colors[self.agent_ind[k]]
                obj_box = obj.bbox_4wc
                polygon = get_polygon(obj_box)
                
                # if k not in self.label_dets :
                #     label = 'Agent {} Detections'.format(k)
                #     self.label_dets.append(k)
                # else :
                #     label = '_'
                
                if k not in self.label_dets :
                    label = 'Agent {} Detections'.format(k)
                    self.label_dets.append(k)
                else :
                    label = '_'
                
                # Polygon x and y swappped as the coordinates are in (y, x) format in Carla System -
                self.plots[1] = self.ax[1].plot(polygon[0:5, 1], polygon[0:5, 0], color=color,linewidth=1,linestyle=s, label = label)

        self.ax[1].set_xlabel('World Coordinates - Y')
        self.ax[1].set_ylabel('World Coordinates - X')
        self.ax[1].legend()
    
    # def plot_merged(self, merged_boxes):
        
    #     # ax3 = self.ax2[self.curr_cat][0]
    #     # ax3 = self.plot_data[self.curr_cat]['ax'][1, 0]
    #     ax3 = self.ax[1]
    #     ax3.set_title('Merged')

    #     for k, obj in merged_boxes.items():
    #         s = self.style[obj.obj_cat]
    #         box = obj.bbox_4wc
    #         polygon = get_polygon(box)

    #         if k not in self.label_merged  :
    #             label = 'Merged {} '.format(k)
    #             self.label_merged.append(k)
    #         else:
    #             label =  '_'
    #         # Polygon x and y swappped as the coordinates are in (y, x) format in Carla System -
    #         self.plots[2] =ax3.plot( polygon[0:5, 1], polygon[0:5, 0], color=self.cmap2[k], linewidth=1,linestyle=s, label = label)
        
    #     ax3.set_xlabel('World Coordinates - Y')
    #     ax3.set_ylabel('World Coordinates - X')
    #     ax3.legend()

    def plot_tracked(self, tracked_boxes):

        #ax4 = self.ax2[self.curr_cat][1]
        # ax4 = self.plot_data[self.curr_cat]['ax'][1, 1]
        # ax4 = self.ax[2]
        self.ax[2].set_title('Tracked')
        self.label_tracklets =[]
        for k, obj in tracked_boxes.items():
            k = int(k)
            s = self.style[obj.obj_cat]
            polygon = get_polygon(obj.bbox_4wc)
            
            if k not in self.label_tracklets:
                label = 'Tracklet {} '.format(k) 
                self.label_tracklets.append(k)
            else :
                label = '_'
            
            # Polygon x and y swappped as the coordinates are in (y, x) format in Carla System - 
            self.plots[2] = self.ax[2].plot( polygon[0:5, 1], polygon[0:5, 0], color=self.cmap2[k],linewidth=1,linestyle=s, label = label)

        self.ax[2].set_xlabel('World Coordinates - Y')
        self.ax[2].set_ylabel('World Coordinates - X')
        self.ax[2].legend()

    def plot_boxes(self, gt_actors, agent_list, world_obj, detections, merged_boxes, tracked_boxes):
        '''
        Plotting the top view points of the bounding box for the BEV - GT, Predictions, Merged and Tracked boxes
        args : figure, axes *4, boxes, matched indices, agent list, world object, ground truth bounding boxes, merged boxes, tracked boxes
        returns : None
        '''
        
        frame_id = world_obj.get_snapshot().frame
        self.agent_ind = {x: k for k, x in enumerate(agent_list)}
        self.count += 1

        # for i in self.object_classes:
            # xy = self.xy[i]
            # #fig  = self.fig
            # self.ax[0, 0].set(xlim = (xy[0], xy[2]), ylim = (xy[1], xy[3]))
            # self.ax[1, 0].set(xlim = (xy[0], xy[2]), ylim = (xy[1], xy[3]))
            # self.ax[0, 0].set_aspect('equal')
            # self.ax[1, 0].set_aspect('equal')
            
        # if len(merged_boxes)==0:
            
        gt = {k : v for k,v in gt_actors.items() if v['obj_data'].obj_cat == self.object_cat or v['agent'] == True}
        #dets = {k : v for k,v in detections.items() if v.obj_cat == i}
        
        merges =  {k : v for k,v in merged_boxes.items() if v.obj_cat == self.object_cat}
        
        if merges == {}:
            return 0

        tracked = tracked_boxes[self.object_cat]
        
        self.plot_gt(gt, agent_list)
        self.plot_detections(detections)
        # self.plot_merged(merges)
        self.plot_tracked(tracked)
    
        self.fig.suptitle('BEV {} - {} agents, Frame {}'.format(self.object_cat, len(agent_list), frame_id))
        self.fig.savefig('../../figures/Plot/{}/{}.png'.format(self.object_cat, frame_id))

        for i in range(3):
            self.ax[i].clear()
            self.ax[i].set(xlim = (self.xy[1], self.xy[3]), ylim = (self.xy[0], self.xy[2]))
            self.ax[i].set_aspect('equal')

        # ax[1, 1].set(xlim = (xy[1], xy[3]), ylim = (xy[0], xy[2]))
        # self.fig.canvas.flush_evets()
        # if self.count == 4 :       
        #     plt.close(self.fig)


if __name__ == '__main__':
    X = 0
    #plot_test(merged)