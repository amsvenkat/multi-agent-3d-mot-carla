## Multi Agent 3D Multi-Object Tracking: A Baseline Framework

## Abstract
Over the last few years, single agent visual scene understanding has significantly progressed. Nonetheless, single-agent perception suffers from significant occlusion given complex scenes. At the same time, bandwidth-related barriers are lowering in recent years. In this work, we present a 3D multi-object tracking framework that fuses monocular 3D object detections produced by multiple agents in the bird's-eye-view. In order to demonstrate how an increased effective field of view can contribute to improved scene understanding, we employ the Carla simulator to retrieve complex, occluded scenes.

## Prerequisites
### Dataset Generation
Dataset of images and labels generated in Carla follow the Kitti Format. 

Create separate environment based on the environment.yaml file and follow the steps to generate the dataset:

1. Get images with the N argument for the capture interval
 
 - Terminal 1
    ```
    cd carla/carla_9.11/
    ./CarlaUE4.sh -quality-level=Epic -world-port=2000 -resx=800 -resy=600
    ```

 - Terminal 2
    ```
    cd src/data_generation/
    conda activate smoke_det
    python spawn.py -n 40 -w 30
    ```
- Terminal 3
    ```
    cd src/data_generation/
    conda activate smoke_det
    python get_images.py -N 50

    ```

2. Get corresponding labels in Kitti format, once required number of images are generated

   ```
   python get_labels.py
   ```

3. Create Imagesets as per the count in the images and labels generated

### Object Detector
Off-the-shelf 3D object detector-SMOKE is used. For more information on SMOKE, check the official SMOKE page (https://github.com/lzccccc/SMOKE)

1. To train the model on the above data, edit the config file as per your directory structure:
    ```
    cd src/SMOKE
    python tools/plain_train_net_validation.py --num-gpus 2 --config-file "configs/carla_train.yaml"

    ```

2. To evaluate model, edit the config.json to include the directory paths, and give the path of the config file as argument. Evaluation metrics are based on Nuscenes metrics-NDS,mAP and mTPs.

    ```
    cd src/SMOKE/evaluation
    python eval.py config.json

    ``` 

## Multi Object Detection and Tracking
To run the multi-object detector and tracker,

1. Edit the config file to add your path for the tracking and gt sequences and the model weights.

2. Run below:

 - Terminal 1
    ```
    cd carla/carla_9.11/
    ./CarlaUE4.sh -quality-level=Epic -world-port=2000 -resx=800 -resy=600
    ```

 - Terminal 2
    ```
    cd src/data_generation/
    conda activate smoke_det
    python spawn.py -n 40 -w 30
    ```

 - Terminal 3
    ```
    cd src/multi_agent/
    conda activate smoke_det
    python multiagent_objdet.py
    ```

   Class related tracks can be visualized in the figures directory.

3. To run the evaluation after taking sufficient frames of data, edit seqmap file in the evaluation directory to include the sequences and frame numbers.
Arguments required: results directory, number of hypothesis, 2D or 3D metric, threshold value for data association (eg. 0.25), single or multi.
Also, add gt_path as per your results path in evaluate.py

   ```
   cd src/multi_agent
   conda activate smoke_det
   python evaluation/evaluate.py tracking_1 1 3D 0.25 single
   ```

## Acknowledgements
Code snippets borrowed from multiple repositories - [Carla](https://github.com/carla-simulator/carla), [AB3DMOT](https://github.com/xinshuoweng/AB3DMOT), [SMOKE](https://github.com/lzccccc/SMOKE)
My sincere thanks to Github community.