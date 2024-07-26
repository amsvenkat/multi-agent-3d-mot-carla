import os

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()
_C.MODEL = CN()
_C.MODEL.SMOKE_ON = True
_C.MODEL.DEVICE = "cuda"
#_C.MODEL.WEIGHT = "/export/amsvenkat/project/3d-multi-agent-tracking/src/SMOKE/logs/logs_scratch_carla/model_final.pth"
_C.MODEL.WEIGHT = "/export/amsvenkat/project/3d-multi-agent-tracking/src/SMOKE/logs/logs_v4.2/model_best.pth"
# _C.MODEL.WEIGHT = "catalog://ImageNetPretrained/DLA34"
# _C.MODEL.WEIGHT = "/export/amsvenkat/project/3d-multi-agent-tracking/src/SMOKE/logs/logs_v5.0/model_best_metric_049.pth"
# _C.MODEL.WEIGHT = "/export/amsvenkat/project/3d-multi-agent-tracking/src/SMOKE/logs/logs_v7.0/model_final_050.pth"
_C.MODEL.VAL = False

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------

_C.INPUT = CN()
# Size of the smallest side of the image during training
_C.INPUT.HEIGHT_TRAIN = 384
# Maximum size of the side of the image during training
_C.INPUT.WIDTH_TRAIN = 1280
# Size of the smallest side of the image during testing
_C.INPUT.HEIGHT_TEST = 384
# Maximum size of the side of the image during testing
_C.INPUT.WIDTH_TEST = 1280
# Values to be used for image normalization
#_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]  # kitti
# _C.INPUT.PIXEL_MEAN = [0.4850, 0.4738, 0.4247]# 7850 dataset carla
# #_C.INPUT.PIXEL_MEAN = [0.4430, 0.4117, 0.3845]
# #[0.554, 0.523, 0.488]  
# # Values to be used for image normalization
# #_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]  # kitti
# #_C.INPUT.PIXEL_STD =[0.2107, 0.1993, 0.1896]
# _C.INPUT.PIXEL_STD =[0.2495, 0.2475, 0.2349] # 7850 dataset carla
_C.INPUT.PIXEL_STD = [0.2718, 0.2570, 0.2486] # 7850 dataset carla
_C.INPUT.PIXEL_MEAN = [0.4361, 0.4203, 0.3811] # 7850 dataset 

# [0.223, 0.210, 0.191]
# Convert image to BGR format
_C.INPUT.TO_BGR = True
# Flip probability
_C.INPUT.FLIP_PROB_TRAIN = 0.5
# Shift and scale probability
_C.INPUT.SHIFT_SCALE_PROB_TRAIN = 0.3
_C.INPUT.SHIFT_SCALE_TRAIN = (0.2, 0.4)

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ()
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()
_C.DATASETS.VALIDATION = ()
# train split tor dataset

_C.DATASETS.TRAIN_SPLIT = ""
# test split for dataset
_C.DATASETS.TEST_SPLIT = ""
_C.DATASETS.VALIDATION_SPLIT = ""
_C.DATASETS.DETECT_CLASSES = ("Car", "Cyclist", "Pedestrian")
_C.DATASETS.MAX_OBJECTS = 40
_C.DATASETS.CUSTOM_DATA =False
# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 2
# If > 0, this enforces that each collated batch should have a size divisible
# by SIZE_DIVISIBILITY
_C.DATALOADER.SIZE_DIVISIBILITY = 0
# If True, each batch should contain only images for which the aspect ratio
# is compatible. This groups portrait images together, and landscape images
# are not batched with portrait images.
_C.DATALOADER.ASPECT_RATIO_GROUPING = False

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

# The backbone conv body to use
# The string must match a function that is imported in modeling.model_builder
_C.MODEL.BACKBONE.CONV_BODY = "DLA-34-DCN"

# Add StopGrad at a specified stage so the bottom layers are frozen
_C.MODEL.BACKBONE.FREEZE_CONV_BODY_AT = 0
# Normalization for backbone
_C.MODEL.BACKBONE.USE_NORMALIZATION = "GN"
_C.MODEL.BACKBONE.DOWN_RATIO = 4
_C.MODEL.BACKBONE.BACKBONE_OUT_CHANNELS = 64

# ---------------------------------------------------------------------------- #
# Group Norm options
# ---------------------------------------------------------------------------- #
_C.MODEL.GROUP_NORM = CN()
# Number of dimensions per group in GroupNorm (-1 if using NUM_GROUPS)
_C.MODEL.GROUP_NORM.DIM_PER_GP = -1
# Number of groups in GroupNorm (-1 if using DIM_PER_GP)
_C.MODEL.GROUP_NORM.NUM_GROUPS = 32
# GroupNorm's small constant in the denominator
_C.MODEL.GROUP_NORM.EPSILON = 1e-5

# ---------------------------------------------------------------------------- #
# Heatmap Head options
# ---------------------------------------------------------------------------- #

# --------------------------SMOKE Head--------------------------------
_C.MODEL.SMOKE_HEAD = CN()
_C.MODEL.SMOKE_HEAD.PREDICTOR = "SMOKEPredictor"
_C.MODEL.SMOKE_HEAD.LOSS_TYPE = ("FocalLoss", "DisL1")
_C.MODEL.SMOKE_HEAD.LOSS_ALPHA = 2
_C.MODEL.SMOKE_HEAD.LOSS_BETA = 4
# Channels for regression
_C.MODEL.SMOKE_HEAD.REGRESSION_HEADS = 8
# Specific channel for (depth_offset, keypoint_offset, dimension_offset, orientation)
_C.MODEL.SMOKE_HEAD.REGRESSION_CHANNEL = (1, 2, 3, 2)
_C.MODEL.SMOKE_HEAD.USE_NORMALIZATION = "GN"
_C.MODEL.SMOKE_HEAD.NUM_CHANNEL = 256
# Loss weight for hm and reg loss
_C.MODEL.SMOKE_HEAD.LOSS_WEIGHT = (1., 10.)
# Reference car size in (length, height, width)
# for (car, cyclist, pedestrian)
# kitti
# _C.MODEL.SMOKE_HEAD.DIMENSION_REFERENCE = ((3.88, 1.63, 1.53),
#                                            (1.78, 1.70, 0.58),
#                                            (0.88, 1.73, 0.67))
# _C.MODEL.SMOKE_HEAD.DIMENSION_REFERENCE = ((4.49, 1.60, 1.99),
#                                           (0.39, 1.67, 0.39),
#                                           (0.88, 1.73, 0.67),)
# avg_car:  [ 4.58550587  1.55275123  2.00934705 19.91398062]
# avg_ped: [ 0.38817947  1.72871441  0.38817947 20.51485764]
# avgdep:  20.13723872622355
# stdz: 11.10141000398532
_C.MODEL.SMOKE_HEAD.DIMENSION_REFERENCE = (( 4.64, 1.60, 2.03),(1.78, 1.70, 0.58),(0.39,  1.73,  0.39),)
# Reference depth
_C.MODEL.SMOKE_HEAD.DEPTH_REFERENCE = ( 20.13723872622355,  11.10141000398532) # train v5
# train v3 - (27.15, 14.62) # (28.01, 16.32)  
#_C.MODEL.SMOKE_HEAD.DEPTH_REFERENCE = (41.77, 33.90)
_C.MODEL.SMOKE_HEAD.USE_NMS = False

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER = "Adam"
_C.SOLVER.MAX_ITERATION = 14500
_C.SOLVER.STEPS = (5850, 9350)

_C.SOLVER.BASE_LR = 0.00025
_C.SOLVER.BIAS_LR_FACTOR = 2
_C.SOLVER.LOAD_OPTIMIZER_SCHEDULER = True

_C.SOLVER.CHECKPOINT_PERIOD = 20
_C.SOLVER.EVALUATE_PERIOD = 20

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 32
_C.SOLVER.MASTER_BATCH = -1

# ---------------------------------------------------------------------------- #
# Test
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST.SINGLE_GPU_TEST = True
_C.TEST.IMS_PER_BATCH = 1
_C.TEST.PRED_2D = True

# Number of detections per image
_C.TEST.DETECTIONS_PER_IMG = 50
_C.TEST.DETECTIONS_THRESHOLD = 0.25


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Directory where output files are written
_C.OUTPUT_DIR = "./tools/logs_v3.1"
# Set seed to negative to fully randomize everything.
# Set seed to positive to use a fixed seed. Note that a fixed seed does not
# guarantee fully deterministic behavior.
_C.SEED = -1
# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
_C.CUDNN_BENCHMARK = True

_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
