#!/usr/bin/env python
# Code by Amrutha Venkatesan
import numpy as np
import torch
from smoke.config import cfg
from smoke.modeling.detector import build_detection_model
from smoke.utils.check_point import DetectronCheckpointer
from smoke.data.transforms import build_transforms
from smoke.structures.params_3d import ParamsList
from smoke.modeling.heatmap_coder import get_transfrom_matrix
from PIL import Image
from collections import defaultdict

class DetectionModel(object):

    def __init__(self, model_path, calib):
        self.model = build_detection_model(cfg)
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.cpu_device = torch.device('cpu')
        self.model.to(self.device)
        self.transforms = build_transforms(cfg)
        checkpointer = DetectronCheckpointer(cfg, self.model)
        _ = checkpointer.load(model_path)
        self.camera_calib = calib
        self.model.eval()
        print('Model loaded') 

    def main(self, img_list):
        preds = defaultdict(int)
        for i in range(len(img_list)):
            agent_id = img_list[i]['agent_id']
            img = img_list[i]['image']
            dets = self.network(img)
            if len(dets) > 0:    
                preds[agent_id] = dets
        # Preds shape = (num_agents, corresponding_num_detections, 13)


        return preds

    def get_3d_detections(self, prediction):
        ID_TYPE_CONVERSION = {
        0: 'Car',
        1: 'Cyclist',
        2: 'Pedestrian'
        }
        all_objs = []
        for p in prediction:
            if p[-1] < 0.3:
                continue
            p = p.numpy()
            p = p.round(4)
            type = ID_TYPE_CONVERSION[int(p[0])]
            row = [type, 0, 0] + p[1:].tolist()
            all_objs.append(row)

        # get detections order by score desc
        all_objs = sorted(all_objs, key=lambda x: x[15], reverse=True)
        return all_objs

    def network(self, img):
        
        img=Image.fromarray(np.uint8(img)).convert('RGB')
        center = np.array([i / 2 for i in img.size], dtype=np.float32)
        size = np.array([i for i in img.size], dtype=np.float32)
        center_size = [center, size]
        input_width = 1280
        input_height = 384
        output_width = 1280 //4
        output_height = 384 //4   
        
        trans_affine = get_transfrom_matrix(
            center_size,
            [input_width, input_height]
        )

        trans_affine_inv = np.linalg.inv(trans_affine)
        img = img.transform(
            (input_width, input_height),
            method=Image.AFFINE,
            data=trans_affine_inv.flatten()[:6],
            resample=Image.BILINEAR,
        )

        trans_mat = get_transfrom_matrix(
            center_size,
            [output_width, output_height]
        )

        # for inference we parametrize with original size
        target = ParamsList(image_size=center_size[1],is_train=False)
        target.add_field('trans_mat', trans_mat)
        target.add_field('K', self.camera_calib)
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        
        img = img.to(self.device)
        
        with torch.no_grad():
            output, _ = self.model(img, target)
            output = output.to(self.cpu_device)
          
        dets = self.get_3d_detections(output)
        return dets

