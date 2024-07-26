import os


class DatasetCatalog():
    DATA_DIR = "datasets"
    DATASETS = {
        # chanegs for validation
        "kitti_train": {
            "root": "/export/amsvenkat/project/3d-multi-agent-tracking/data/train_v3/",
            #"root": "kitti/training/",
        },
        "kitti_test": {
            "root": "/export/amsvenkat/project/3d-multi-agent-tracking/data/train_v3/",
            #"root": "kitti/testing/",
        },
        "kitti_val": {
            "root": "/export/amsvenkat/project/3d-multi-agent-tracking/data/train_v3/",
           
            #"root": "kitti/testing/",
        },
        # chanegs for validation
        "carla_train": {
            "root": "/export/amsvenkat/project/3d-multi-agent-tracking/data/train_v3/"
            # "root": "/work/dlclarge1/venkatea-objdet/data/",
            #"root": "kitti/training/",
        },
        "carla_test": {
            # "root": "/work/dlclarge1/venkatea-objdet/data/",
            # "root": "/work/dlclarge2/venkatea-obj-detection-workspace/tool_objdetection/data/",
            "root": "/export/amsvenkat/project/3d-multi-agent-tracking/data/train_v3/",
            #"root": "kitti/testing/",
        },
        "carla_val": {
        #    "root": "/work/dlclarge2/venkatea-obj-detection-workspace/tool_objdetection/data/",
            # "root" :"/work/dlclarge1/venkatea-objdet/data/"
            "root": "/export/amsvenkat/project/3d-multi-agent-tracking/data/train_v3/",
            #"root": "kitti/testing/",
        },
        

    }

    @staticmethod
    def get(name):
        # chnaged to carla from kitti for carla testing
        if "carla" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["root"]),
            )
            return dict(
                factory="KITTIDataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog():
    IMAGENET_MODELS = {
        "DLA34": "http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth"
    }

    @staticmethod
    def get(name):
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_imagenet_pretrained(name)

    @staticmethod
    def get_imagenet_pretrained(name):
        name = name[len("ImageNetPretrained/"):]
        url = ModelCatalog.IMAGENET_MODELS[name]
        return url
