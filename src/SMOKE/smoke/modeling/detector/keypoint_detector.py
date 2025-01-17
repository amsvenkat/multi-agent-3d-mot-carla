import torch
from torch import nn

from smoke.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..heads.heads import build_heads


class KeypointDetector(nn.Module):
    '''
    Generalized structure for keypoint based object detector.
    main parts:
    - backbone
    - heads
    '''

    def __init__(self, cfg):
        super(KeypointDetector, self).__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.heads = build_heads(cfg, self.backbone.out_channels)

    def forward(self, images, targets=None):
        """
        Args:
            images:
            targets:

        Returns:

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        result, detector_losses = self.heads(features, targets)
        losses = {}
        losses.update(detector_losses)

        return result, losses
        
    # v3 - changes 

        # Added below afor validation script
        # if self.cfg.MODEL.VAL == True:
        #     losses = {}
        #     losses.update(detector_losses)
        #     return losses
    # v1 - original version  
        # if self.training or self.cfg.MODEL.VAL == True:
        #     losses = {}
        #     losses.update(detector_losses)
        #     return losses

        # return result