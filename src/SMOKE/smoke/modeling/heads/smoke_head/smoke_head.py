import torch
from torch import nn

from .smoke_predictor import make_smoke_predictor
from .loss import make_smoke_loss_evaluator
from .inference import make_smoke_post_processor


class SMOKEHead(nn.Module):
    def __init__(self, cfg, in_channels):
        super(SMOKEHead, self).__init__()

        self.cfg = cfg.clone()
        self.predictor = make_smoke_predictor(cfg, in_channels)
        self.loss_evaluator = make_smoke_loss_evaluator(cfg)
        self.post_processor = make_smoke_post_processor(cfg)

    def forward(self, features, targets=None):
        x = self.predictor(features)
        
        # changees for valiation with evaluation of metrics 
        # loss_heatmap, loss_regression = self.loss_evaluator(x, targets)
        # result = self.post_processor(x, targets)
        # return result, dict(hm_loss=loss_heatmap,reg_loss=loss_regression, )


    # Changes for validation script below if statement
    #v2 and v3 - with validation script
        # if not self.training #and self.cfg.MODEL.VAL == True:
            
        #     #loss_heatmap, loss_regression = self.loss_evaluator(x, targets)
        #     result = self.post_processor(x, targets)
        #     return result, {}
        
        if self.training:
            loss_heatmap, loss_regression = self.loss_evaluator(x, targets)

            return {}, dict(hm_loss=loss_heatmap,
                            reg_loss=loss_regression, )
        
        if not self.training: # and self.cfg.MODEL.VAL == False:
            result = self.post_processor(x, targets)

            return result, {}
    
    # v1 - original version
        # if self.training:
        #     loss_heatmap, loss_regression = self.loss_evaluator(x, targets)

        #     return {}, dict(hm_loss=loss_heatmap,
        #                     reg_loss=loss_regression, )
        # if not self.training:
        #     result = self.post_processor(x, targets)

        #     return result, {}


def build_smoke_head(cfg, in_channels):
    return SMOKEHead(cfg, in_channels)
