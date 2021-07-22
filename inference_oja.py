import urx
import time
import math3d as m3d
from threading import Thread
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper

import rospy
from sensor_msgs.msg import Image

import os
import sys
import yaml
import copy
import numpy as np
from collections import namedtuple

import torch
import train_model as model
import torch.backends.cudnn as cudnn
from skimage import transform as sktrans

ACCE = 0.1
VELO = 0.2
###############  UPDATE ME  ###############
IP_ROBOT = "192.168.1.5"
TRANS_TCP = (0, 0, 0.19, 1.2092, -1.2092, 1.2092)
TRANS_BASE = [0, 0, 0, -0.61394313, 1.48218982, 0.61394313]
TOPIC_LEFT = '/realsense_wrist/infra1/image_rect_raw'
TOPIC_RIGHT = '/realsense_wrist/infra2/image_rect_raw'
###########################################


class Interface:
    def __init__(self):
        # ROS
        rospy.init_node('ur_interface')

        # robot setup
        print('Connecting robot...')
        self.rob = urx.Robot(IP_ROBOT, True)
        self.rob.set_tcp(TRANS_TCP)
        self.rob.set_csys(m3d.Transform(TRANS_BASE))
        time.sleep(0.5)
        self.gripper = Robotiq_Two_Finger_Gripper(self.rob)

        # placeholder
        self.sub_img_infraL, self.sub_img_infraR = None, None
        self.data_img_infraL, self.data_img_infraR = None, None
        self.acquire_img_infraL, self.acquire_img_infraR = False, False

        # network
        self.servo_net_pick_mug = None
        self.servo_arg_pick_mug = None
        self.init_network('pick_mug')
        # self.servo_net_insert_plug = None
        # self.servo_arg_insert_plug = None
        # self.init_network('insert_plug')
        # self.servo_net_insert_shaft = None
        # self.servo_arg_insert_shaft = None
        # self.init_network('insert_shaft')

    def disconnect(self):
        """
        terminate connection with robot
        :return: None
        """
        self.rob.close()

    def move_tcp_absolute(self, pose, wait=True):
        """
        move eff to absolute pose in robot base frame with position control
        :param pose: list [x y z R P Y] (meter, radian)
        :param wait: blocking wait
        :return: None
        """
        self.rob.set_pose(m3d.Transform(pose), ACCE, VELO, wait)

    def move_tcp_relative(self, pose, wait=True):
        """
        move eff to relative pose in tool frame with position control
        :param pose: relative differences in [x y z R P Y] (meter, radian)
        :param wait: blocking wait
        :return: None
        """
        self.rob.add_pose_tool(m3d.Transform(pose), ACCE, VELO, wait)

    def move_tcp_perpendicular(self, wait=True):
        """
        move eff perpendicular to xy-plane
        :param wait: blocking wait
        :return: None
        """
        tcp_pose = self.rob.getl()
        self.move_tcp_absolute([tcp_pose[i] if i < 4 else 0 for i in range(6)], wait)

    def set_gripper(self, val):
        """
        gripper position control
        :param val: boolean (False:open, True:close)
        :return: None
        """
        if val:
            self.gripper.close_gripper()
        else:
            self.gripper.open_gripper()

    def _cb_img_infraL(self, img):
        if self.acquire_img_infraL:
            self.data_img_infraL = copy.deepcopy(img)
            self.acquire_img_infraL = False

    def _cb_img_infraR(self, img):
        if self.acquire_img_infraR:
            self.data_img_infraR = copy.deepcopy(img)
            self.acquire_img_infraR = False

    def init_network(self, net):
        """
        initialize servo network
        :param net: name of the servo
        :return: None
        """
        dir_path = os.path.dirname(os.path.realpath(__file__))
        cudnn.enabled = True
        cudnn.benchmark = True
        cudnn.deterministic = True
        arg = yaml.load(open(os.path.join(dir_path, 'result/{}/servo.yaml'.format(net)), 'r'), yaml.Loader)
        arg = namedtuple('Arg', arg.keys())(**arg)
        ckpt = torch.load(os.path.join(dir_path, 'result/{}/ckpt.pth'.format(net)), map_location=torch.device('cpu'))
        kper = model.KeyPointGaussian(arg.sigma_kp[1], (arg.num_keypoint, *arg.im_size[1]))
        enc = model.Encoder(arg.num_input, arg.num_keypoint, arg.growth_rate[0], arg.blk_cfg_enc, arg.drop_rate,
                            kper, True).cuda()
        cvt = model.ConverterServo(arg.num_keypoint * 6, arg.growth_rate[2], arg.blk_cfg_cvt,
                                   [sum(arg.motion_vec), 1]).cuda()
        enc.load_state_dict(ckpt['enc_state_dict'])
        cvt.load_state_dict(ckpt['cvt_state_dict'])
        setattr(self, 'net_servo_{}'.format(net), torch.nn.Sequential(enc, cvt))
        getattr(self, 'net_servo_{}'.format(net)).eval()
        setattr(self, 'arg_servo_{}'.format(net), arg)

    def servo(self, action, duration, speed, thresholds, post_job=None):
        """
        start servo routine
        :param action: servo name
        :param duration: number of second for the servo to run
        :param speed: list of 6 float as tcp's speed in m/s
        :param thresholds: list of a float and an int as low speed (completion) and count threshold
        :param post_job: list of 6 float to move tcp relatively after servo completed
        :return: None
        """
        # subscribe to images
        if self.sub_img_infraL is None:
            self.sub_img_infraL = rospy.Subscriber(TOPIC_LEFT, Image, self._cb_img_infraL, queue_size=1)
        if self.sub_img_infraR is None:
            self.sub_img_infraR = rospy.Subscriber(TOPIC_RIGHT, Image, self._cb_img_infraR, queue_size=1)

        # align tcp
        self.move_tcp_perpendicular()

        self.process_servo = Thread(target=self._watch_servo,
                                    args=(action, duration,
                                          getattr(self, 'arg_servo_{}'.format(action)).im_size[0],
                                          speed, thresholds, post_job))
        self.process_servo.start()

    def _watch_servo(self, action, duration, sizes, speeds, thresholds, post_job):
        t = 0.5
        time0 = time.time()
        cnt_end = 0
        vec_idx = np.where(getattr(self, 'arg_servo_{}'.format(action)).motion_vec)[0].tolist()

        # start servoing
        while time.time() - time0 < duration:
            # acquire images
            self.acquire_img_infraL = True
            self.acquire_img_infraR = True
            cnt_no_img = 0
            while self.acquire_img_infraL or self.acquire_img_infraR:
                time.sleep(0.005)
                cnt_no_img += 1
                if cnt_no_img > 100:
                    break
            if cnt_no_img > 100:
                rospy.logwarn('no camera data received!')
                # break
            infraL = self._img_preproc(self.data_img_infraL, sizes)
            infraR = self._img_preproc(self.data_img_infraR, sizes)

            # forward-pass
            with torch.no_grad():
                vec, speed = getattr(self, 'net_servo_{}'.format(action))([infraL.cuda(), infraR.cuda()])

            # tcp motion
            speed = torch.sigmoid(speed).detach().cpu().item()
            speed = max(0, speed - thresholds[0])
            vec = (vec / torch.norm(vec)).detach().cpu()
            vec = [vec[vec_idx.index(i)] * speed * speeds[i] if i in vec_idx else 0 for i in range(6)]
            self.rob.speedl_tool(vec, ACCE, t)
            if speed > thresholds[0]:
                cnt_end = 0
            elif speed <= thresholds[0]:
                cnt_end += 1
                if cnt_end > thresholds[1]:
                    rospy.logdebug('servo done')
                    if post_job is not None:
                        self.move_tcp_relative(post_job)
                        rospy.logdebug('post servo job done')
                    self.servo_done = action
                    break

        # unsubscribe to images
        self.sub_img_infraL.unregister()
        self.sub_img_infraR.unregister()
        self.sub_img_infraL = None
        self.sub_img_infraR = None
        self.acquire_img = 0

    @staticmethod
    def _img_preproc(msg, sizes):
        dtype = np.dtype(np.uint8).newbyteorder('>' if msg.is_bigendian else '<')
        img = np.ndarray(shape=(msg.height, msg.width), dtype=dtype, buffer=msg.data)
        if msg.is_bigendian == (sys.byteorder == 'little'):
            img = img.byteswap().newbyteorder()

        # crop - scale - normalize
        ratios = img.shape[1] * 1. / img.shape[0], sizes[1] * 1. / sizes[0]
        if not np.isclose(*ratios):
            if ratios[0] > ratios[1]:
                l = int(np.round(img.shape[0] * ratios[1]))
                a = (img.shape[1] - l) // 2
                img = img[:, a:a + l]
            else:
                l = int(np.round(img.shape[1] / ratios[1]))
                a = (img.shape[0] - l) // 2
                img = img[a:a + l, :]
        img = sktrans.resize(img, sizes, order=1, preserve_range=True, clip=False, anti_aliasing=False)
        img = torch.from_numpy(img[None, ...]).float().div(255).sub_(0.44).div_(0.26)
        return img.unsqueeze(0)
