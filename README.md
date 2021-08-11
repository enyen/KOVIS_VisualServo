# KOVIS: Keypoint-based Visual Servoing with Zero-Shot Sim-to-Real Transfer for Robotics Manipulation

This repo covers the data generation, training and inference of kovis visual servos introduced in 
[KOVIS: Keypoint-based Visual Servoing with Zero-Shot Sim-to-Real Transfer for Robotics Manipulation](https://arxiv.org/abs/2007.13960) (IROS 2020) 
by [En Yen Puang](https://scholar.google.com/citations?user=jePmCqAAAAAJ), 
[Keng Peng Tee](https://scholar.google.com/citations?user=xO_OuigAAAAJ) and 
[Wei Jing](https://scholar.google.com/citations?user=12L_IZMAAAAJ).

Inference uses UR5, realsense D435 camera, ROS and urx.

[ [Paper](https://arxiv.org/abs/2007.13960) ] [ [Video](https://www.youtube.com/watch?v=gfBJBR2tDzA) ]

![demo](demo.gif)

# Installation
We recommend using virtual python environment like [Conda](https://docs.conda.io/en/latest/miniconda.html) with python3.
1. Install [PyTorch](https://pytorch.org/)
2. Install other packages
```bash
pip install -r requirements.txt 
```

3. (Optinal) Install [ROS](http://wiki.ros.org/ROS/Installation), [rospkg](http://wiki.ros.org/rospkg) and [RealSense](https://github.com/IntelRealSense/realsense-ros) camera driver
3. (Optinal) install [urx](https://github.com/SintefManufacturing/python-urx) for UR robot
```bash
pip install git+https://github.com/enyen/python-urx
```

# Usage

1. Generate training data in pyBullet:

```bash
cd KOVIS_VisualServo
# example for generating dataset for pick-mug task
python oja_pick.py cfg/dataset_pick_mug.yaml
```

If no monitor is connected, render without shadow by replacing line 19 with
```python
pyb.connect(pyb.DIRECT)  # pyb.connect(pyb.GUI)
```

2. Training servo in pyTorch:

```bash
# example for training for pick-mug task
python train_servo.py cfg/train_pick_mug.yaml
```

3. Running on robot:
    -   launch realsense camera with both infra cameras enabled
    -   turn off realsense laser using rqt_reconfigure
```python
from inference_oja import Interface
rob = Interface()

# reach
# TODO: move tcp to close to object

# pick
rob.servo('pick_mug', 10, [0.01, 0.01, 0.01, 0.05, 0, 0], [0.1, 5])
rob.set_gripper(1)

# continue
# TODO: move object away
```

# Citation
Please cite our paper if you use this code.

      @inproceedings{puang2020kovis,
        title={KOVIS: Keypoint-based Visual Servoing with Zero-Shot Sim-to-Real Transfer for Robotics Manipulation},
        author={Puang, En Yen and Tee, Keng Peng and Jing, Wei},
        booktitle={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
        pages={7527--7533},
        year={2020},
        organization={IEEE}
      }

# License
This code is under GPL-3.0 License. Contact

puangenyen at gmail . com 

to discuss other license agreement.