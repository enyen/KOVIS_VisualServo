import pybullet as pyb
import numpy as np

PI = 3.14159265359
ARM_LOWER = [-2*PI, -2*PI, -2*PI, -2*PI, -2*PI, -2*PI]
ARM_UPPER = [2*PI, 2*PI, 2*PI, 2*PI, 2*PI, 2*PI]
ARM_RANGE = [4*PI, 4*PI, 4*PI, 4*PI, 4*PI, 4*PI]
ARM_HOME = [PI/4, -PI/2, -PI/2, -PI/2, PI/2, 0]
ARM_DAMP = [0.01, 0.01, 0.001, 0.001, 0.001, 0.001]

GRIPPER_LOWER = [0, -0.8, 0, 0, -0.8, 0]
GRIPPER_UPPER = [0.8, 0, 0.8, 0.8, 0, 0.8]
GRIPPER_RANGE = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
GRIPPER_HOME = [0, 0, 0, 0, 0, 0]
GRIPPER_DAMP = [0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001]

IDX_ARM = [1, 2, 3, 4, 5, 6]
IDX_GRIPPER = [10, 12, 14, 15, 17, 19]
IDX_TCP = 20
IDX_CAMERA = 23


class Oja:
    def __init__(self, timestep, robot_xyz=(0, 0, 0), robot_rpy=(0, 0, 0)):
        self.robotUid = None
        self.timestep = timestep
        self.load_robot(robot_xyz, robot_rpy)

    def load_robot(self, robot_xyz=(0, 0, 0), robot_rpy=(0, 0, 0), joint_info=True):
        self.robotUid = pyb.loadURDF("model/oja/oja.urdf.xacro", robot_xyz,
                                     pyb.getQuaternionFromEuler(robot_rpy), useFixedBase=True)
        if joint_info:
            print('================================== Joint Info ==================================')
            for i in range(pyb.getNumJoints(self.robotUid)):
                info = pyb.getJointInfo(self.robotUid, i)
                if info[2] == 0:
                    jtype = 'revolute'
                elif info[2] == 1:
                    jtype = 'prismatic'
                elif info[2] == 2:
                    jtype = 'spherical'
                elif info[2] == 3:
                    jtype = 'planner'
                elif info[2] == 4:
                    jtype = 'fixed'
                else:
                    jtype = 'unknown'
                print('{:02d}'.format(info[0]), '  ', jtype, '\t', info[1].decode("utf-8"), ' ', info[12].decode("utf-8"))
            print('================================================================================')

    def home(self):
        self._set_joint(IDX_ARM, ARM_HOME, True)
        self._set_joint(IDX_GRIPPER, GRIPPER_HOME, True)

    def get_tcp_pose(self):
        return pyb.getLinkState(self.robotUid, IDX_TCP, computeForwardKinematics=1)[4:6]

    def set_gripper_joint(self, pos, reset=False):
        self._set_joint(IDX_GRIPPER, [pos, -pos, pos, pos, -pos, pos], reset)

    def set_tcp_pose(self, pos, rot, reset=True):
        for _ in range(5):
            tcp_pos, tcp_rot = self.get_tcp_pose()
            if (np.abs(np.array(tcp_pos) - pos).sum() > 0.004) or \
                    (np.abs(np.array(tcp_rot) - rot).sum() > 0.015):
                jpos = pyb.calculateInverseKinematics(self.robotUid, IDX_TCP, pos, rot,
                                                      ARM_LOWER+GRIPPER_LOWER, ARM_UPPER+GRIPPER_UPPER,
                                                      ARM_RANGE+GRIPPER_RANGE, ARM_HOME+GRIPPER_HOME,
                                                      ARM_DAMP+GRIPPER_DAMP, pyb.IK_SDLS)
                jpos = jpos[:6]
                self._set_joint(IDX_ARM, jpos, reset)
            else:
                return True
        # return False
        return True

    def apply_speed_tcp(self, vel_xyz, vel_rpy, pose_tcp,
                        relative=False, reset=False, timestep=None):
        timestep = timestep if timestep is not None else np.random.uniform(self.timestep[0], self.timestep[1])
        if len(pose_tcp[1]) == 3:
            euler = True
        else:
            euler = False
        if relative:
            tcp_r = pyb.getQuaternionFromEuler(pose_tcp[1]) if euler else pose_tcp[1]
            pos_goal = [timestep * vel_xyz[i] for i in range(3)]
            rot_goal = [timestep * vel_rpy[i] for i in range(3)]
            pos, rot = pyb.multiplyTransforms(pose_tcp[0], tcp_r,
                                              pos_goal, pyb.getQuaternionFromEuler(rot_goal))
            self.set_tcp_pose(pos, rot, reset)
        else:
            tcp_r = pyb.getEulerFromQuaternion(pose_tcp[1]) if not euler else pose_tcp[1]
            pos = [pose_tcp[0][i] + timestep * vel_xyz[i] for i in range(3)]
            rot = [tcp_r[i] + timestep * vel_rpy[i] for i in range(3)]
            rot = pyb.getQuaternionFromEuler(rot)
            self.set_tcp_pose(pos, rot, reset)
        return pos, \
               pyb.getEulerFromQuaternion(rot) if euler else rot

    def _set_joint(self, idx, pos, reset=False):
        if reset:
            pyb.resetJointStatesMultiDof(self.robotUid, idx, [[i] for i in pos], [[0] for _ in range(len(pos))])
        else:
            pyb.setJointMotorControlMultiDofArray(self.robotUid, idx, pyb.POSITION_CONTROL, [[i] for i in pos])

    def get_image(self, size=(240, 240), clip=(0.1, 4.0), fov=58, baseline=None,
                  shift_xyz=(0, 0, 0), shift_rpy=(0, 0, 0), random_lighting=False):
        # camera pose
        cam_pos, cam_rot = pyb.getLinkState(self.robotUid, IDX_CAMERA, computeForwardKinematics=1)[4:6]
        cam_pos, cam_rot = pyb.multiplyTransforms(cam_pos, cam_rot,
                                                  shift_xyz, pyb.getQuaternionFromEuler(shift_rpy))
        if baseline is not None:
            cam_pos, cam_rot = pyb.multiplyTransforms(cam_pos, cam_rot,
                                                      baseline[0], pyb.getQuaternionFromEuler(baseline[1]))
        cam_rot = np.array(pyb.getMatrixFromQuaternion(cam_rot)).reshape(3, 3)

        # rendering
        viewMat = pyb.computeViewMatrix(cam_pos, cam_pos+cam_rot[:, 0], cam_rot[:, 2].tolist())
        projMat = pyb.computeProjectionMatrixFOV(fov, size[0]/size[1], clip[0], clip[1])

        if random_lighting:
            pyb.configureDebugVisualizer(lightPosition=[np.random.uniform(-3, 3),
                                                        np.random.uniform(-3, 3),
                                                        np.random.uniform(1, 5)])

        color, depth, segmask = pyb.getCameraImage(width=size[0],
                                                   height=size[1],
                                                   viewMatrix=viewMat,
                                                   projectionMatrix=projMat,
                                                   renderer=pyb.ER_BULLET_HARDWARE_OPENGL)[2:5]

        return np.reshape(color, (size[1], size[0], 4))[..., :3], \
               np.reshape(depth, (size[1], size[0])), \
               np.reshape(segmask, (size[1], size[0]))
