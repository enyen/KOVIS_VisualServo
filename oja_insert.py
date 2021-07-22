import os
import sys

import yaml
import numpy as np
from PIL import Image
import pybullet as pyb
from tqdm import trange
from collections import namedtuple

from oja import Oja

arg = yaml.load(open(sys.argv[1], 'r'), yaml.Loader)
arg = namedtuple('arg', arg.keys())(**arg)


class OjaInsert:
    def __init__(self, dir_dataset, data_size):
        pyb.connect(pyb.GUI)
        pyb.configureDebugVisualizer(shadowMapResolution=16384)
        pyb.resetSimulation()
        pyb.setTimeStep(sum(arg.time_step) / 2.)

        # object spawning sequence
        self.robot = Oja(arg.time_step)
        self.obj = pyb.loadURDF("model/object/{}.urdf".format(arg.object), useFixedBase=True)
        self.case = pyb.loadURDF("model/object/case.urdf", useFixedBase=True)

        self.dir_dataset = dir_dataset
        self.data_size = data_size
        self.pose_tcp, self.tcp_obj = None, None
        self.label_tmp = []
        self.i_sample = 0
        with open(os.path.join(self.dir_dataset, 'gt.yaml'), 'w') as f:
            yaml.dump([], f, Dumper=yaml.Dumper)

    def __del__(self):
        pyb.disconnect()

    def reset(self):
        while True:
            # case
            pos_case, rot_case = arg.case_pos, pyb.getQuaternionFromEuler(arg.case_rot)
            pyb.resetBasePositionAndOrientation(self.case, pos_case, rot_case)
            # object
            pos_obj, rot_obj = (np.random.rand(3) - 0.5) * arg.obj_pos_rag + arg.obj_pos_ctr, \
                               pyb.getQuaternionFromEuler((np.random.rand(3) - 0.5) * arg.obj_rot_rag + arg.obj_rot_ctr)
            pos_obj, rot_obj = pyb.multiplyTransforms(pos_case, rot_case, pos_obj, rot_obj)
            pyb.resetBasePositionAndOrientation(self.obj, pos_obj, rot_obj)
            # tcp
            pos_obj_, rot_obj_ = pyb.multiplyTransforms(pos_case, rot_case, arg.obj_pos_ctr,
                                                        pyb.getQuaternionFromEuler(arg.obj_rot_ctr))
            pos_tcp, rot_tcp = (np.random.rand(3) - 0.5) * arg.tcp_pos_rag + arg.tcp_pos_ctr, \
                               pyb.getQuaternionFromEuler((np.random.rand(3) - 0.5) * arg.tcp_rot_rag + arg.tcp_rot_ctr)
            pos_tcp, rot_tcp = pyb.multiplyTransforms(pos_obj_, rot_obj_, pos_tcp, rot_tcp)
            self.robot.home()
            if self.robot.set_tcp_pose(pos_tcp, rot_tcp, True):
                break
        self.robot.set_gripper_joint(arg.gripper_joint, True)
        self.pose_tcp = (pos_tcp, rot_tcp)
        self.tcp_obj = pyb.multiplyTransforms(*pyb.invertTransform(pos_tcp, rot_tcp), pos_obj, rot_obj)

    def rollout(self):
        timestep = np.random.uniform(*arg.time_step)
        while True:
            vel = (np.random.rand(6) - 0.5) * (arg.rollout_vel_lin_rag + arg.rollout_vel_rot_rag) \
                  + (arg.rollout_vel_lin_ctr + arg.rollout_vel_rot_ctr)
            vel = vel / np.linalg.norm(vel)
            vel_ = vel[:3].tolist() + (arg.rot_mul * vel[3:]).tolist()

            for i in range(arg.rollout_step):
                pyb.stepSimulation()
                if len(pyb.getClosestPoints(self.obj, self.case, 0)) > 0:
                    break
                self.save_images(i, self.get_images())
                self.pose_tcp = self.robot.apply_speed_tcp(vel_[:3], vel_[3:], self.pose_tcp, True, True, timestep)
                pos_shaft, rot_shaft = pyb.multiplyTransforms(self.pose_tcp[0], self.pose_tcp[1],
                                                              self.tcp_obj[0], self.tcp_obj[1])
                pyb.resetBasePositionAndOrientation(self.obj, pos_shaft, rot_shaft)
                self.robot.set_gripper_joint(arg.gripper_joint, True)
                num_roll = i

            if num_roll >= 2:
                break
            self.reset()

        self.save_label((vel * -1.).tolist() + [timestep])

    def get_images(self):
        shift_xyz = (np.random.rand(3) - 0.5) * arg.cam_pos_rag + arg.cam_pos_ctr
        shift_rpy = (np.random.rand(3) - 0.5) * arg.cam_rot_rag + arg.cam_rot_ctr

        # get complete images - Left
        fov = np.random.uniform(*arg.fov)
        color_l, depth_l, segme_l = self.robot.get_image(shift_xyz=shift_xyz, shift_rpy=shift_rpy, fov=fov,
                                                         size=arg.img_size, clip=arg.cam_clip, random_lighting=True)

        # get complete images - Right
        color_r, depth_r, segme_r = self.robot.get_image(shift_xyz=shift_xyz, shift_rpy=shift_rpy, fov=fov,
                                                         baseline=[(0, -np.random.uniform(*arg.baseline), 0), (0, 0, 0)],
                                                         size=arg.img_size, clip=arg.cam_clip, random_lighting=True)
        return color_l, depth_l, segme_l, color_r, depth_r, segme_r

    def save_images(self, i_rollout, img):
        Image.fromarray(img[0].astype(np.uint8)).save(os.path.join(
            self.dir_dataset, 'left/color/{:05d}_{:02d}.png'.format(self.i_sample - 1, i_rollout)))
        Image.fromarray(img[3].astype(np.uint8)).save(os.path.join(
            self.dir_dataset, 'right/color/{:05d}_{:02d}.png'.format(self.i_sample - 1, i_rollout)))
        Image.fromarray((img[1] * 255).astype(np.uint8)).save(os.path.join(
            self.dir_dataset, 'left/depth/{:05d}_{:02d}.png'.format(self.i_sample - 1, i_rollout)))
        Image.fromarray((img[2] + 1).astype(np.uint8)).save(os.path.join(
            self.dir_dataset, 'left/segme/{:05d}_{:02d}.png'.format(self.i_sample - 1, i_rollout)))
        Image.fromarray((img[4] * 255).astype(np.uint8)).save(os.path.join(
            self.dir_dataset, 'right/depth/{:05d}_{:02d}.png'.format(self.i_sample - 1, i_rollout)))
        Image.fromarray((img[5] + 1).astype(np.uint8)).save(os.path.join(
            self.dir_dataset, 'right/segme/{:05d}_{:02d}.png'.format(self.i_sample - 1, i_rollout)))

    def save_label(self, label=None):
        if label is not None:
            self.label_tmp.append(label)
        if len(self.label_tmp) > 500 or \
                ((label is None) and (len(self.label_tmp) > 0)):
            with open(os.path.join(self.dir_dataset, 'gt.yaml'), 'r') as f:
                labels = yaml.load(f, Loader=yaml.Loader)
            labels.extend(self.label_tmp)
            with open(os.path.join(self.dir_dataset, 'gt.yaml'), 'w') as f:
                yaml.dump(labels, f, Dumper=yaml.Dumper)
            self.label_tmp = []

    def run(self):
        print('collecting {} data...'.format(self.data_size))
        for _ in trange(self.data_size):
            self.i_sample += 1
            self.reset()
            self.rollout()
        self.save_label()


if __name__ == '__main__':
    dir_base = os.path.dirname(os.path.realpath(__file__))
    # training and testing data
    for i in range(2):
        dir_dataset = os.path.join(dir_base, arg.dir_dataset[i])
        if os.path.exists(dir_dataset):
            os.system('rm -r {}'.format(dir_dataset))
        os.makedirs(os.path.join(dir_dataset, 'left/color'))
        os.makedirs(os.path.join(dir_dataset, 'left/depth'))
        os.makedirs(os.path.join(dir_dataset, 'left/segme'))
        os.makedirs(os.path.join(dir_dataset, 'right/color'))
        os.makedirs(os.path.join(dir_dataset, 'right/depth'))
        os.makedirs(os.path.join(dir_dataset, 'right/segme'))
        OjaInsert(dir_dataset, arg.data_size[i]).run()
