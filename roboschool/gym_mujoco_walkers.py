from roboschool.scene_abstract import cpp_household
#from roboschool.scene_stadium import SinglePlayerStadiumScene
from roboschool.gym_forward_walker import RoboschoolForwardWalker
from roboschool.gym_mujoco_xml_env import RoboschoolMujocoXmlEnv
import gym, gym.spaces, gym.utils, gym.utils.seeding
import numpy as np
import os, sys

from roboschool.dm_control_utils import rewards

class RoboschoolForwardWalkerMujocoXML(RoboschoolForwardWalker, RoboschoolMujocoXmlEnv):
    def __init__(self, fn, robot_name, action_dim, obs_dim, power):
        RoboschoolMujocoXmlEnv.__init__(self, fn, robot_name, action_dim, obs_dim)
        RoboschoolForwardWalker.__init__(self, power)

class RoboschoolHopper(RoboschoolForwardWalkerMujocoXML):
    foot_list = ["foot"]
    def __init__(self):
        RoboschoolForwardWalkerMujocoXML.__init__(self, "hopper.xml", "torso", action_dim=3, obs_dim=15, power=0.75)
    def alive_bonus(self, z, pitch):
        return +1 if z > 0.8 and abs(pitch) < 1.0 else -1

class RoboschoolWalker2d(RoboschoolForwardWalkerMujocoXML):
    foot_list = ["foot", "foot_left"]
    def __init__(self):
        RoboschoolForwardWalkerMujocoXML.__init__(self, "walker2d.xml", "torso", action_dim=6, obs_dim=22, power=0.40)
    def alive_bonus(self, z, pitch):
        return +1 if z > 0.8 and abs(pitch) < 1.0 else -1
    def robot_specific_reset(self):
        RoboschoolForwardWalkerMujocoXML.robot_specific_reset(self)
        for n in ["foot_joint", "foot_left_joint"]:
            self.jdict[n].power_coef = 30.0

class RoboschoolHalfCheetah(RoboschoolForwardWalkerMujocoXML):
    foot_list = ["ffoot", "fshin", "fthigh",  "bfoot", "bshin", "bthigh"]  # track these contacts with ground
    def __init__(self):
        RoboschoolForwardWalkerMujocoXML.__init__(self, "half_cheetah.xml", "torso", action_dim=6, obs_dim=26, power=0.90)
    def alive_bonus(self, z, pitch):
        # Use contact other than feet to terminate episode: due to a lot of strange walks using knees
        return +1 if np.abs(pitch) < 1.0 and not self.feet_contact[1] and not self.feet_contact[2] and not self.feet_contact[4] and not self.feet_contact[5] else -1
    def robot_specific_reset(self):
        RoboschoolForwardWalkerMujocoXML.robot_specific_reset(self)
        self.jdict["bthigh"].power_coef = 120.0
        self.jdict["bshin"].power_coef  = 90.0
        self.jdict["bfoot"].power_coef  = 60.0
        self.jdict["fthigh"].power_coef = 140.0
        self.jdict["fshin"].power_coef  = 60.0
        self.jdict["ffoot"].power_coef  = 30.0

class RoboschoolAnt(RoboschoolForwardWalkerMujocoXML):
    foot_list = ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']
    def __init__(self):
        RoboschoolForwardWalkerMujocoXML.__init__(self, "ant.xml", "torso", action_dim=8, obs_dim=28, power=2.5)
    def alive_bonus(self, z, pitch):
        return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground


## 3d Humanoid ##

class RoboschoolHumanoid(RoboschoolForwardWalkerMujocoXML):
    foot_list = ["right_foot", "left_foot"]
    TASK_WALK, TASK_STAND_UP, TASK_ROLL_OVER, TASKS = range(4)

    def __init__(self, model_xml='humanoid_symmetric.xml'):
        RoboschoolForwardWalkerMujocoXML.__init__(self, model_xml, 'torso', action_dim=17, obs_dim=44, power=0.41)
        # 17 joints, 4 of them important for walking (hip, knee), others may as well be turned off, 17/4 = 4.25
        self.electricity_cost  = 4.25*RoboschoolForwardWalkerMujocoXML.electricity_cost
        self.stall_torque_cost = 4.25*RoboschoolForwardWalkerMujocoXML.stall_torque_cost
        self.initial_z = 0.8

    def robot_specific_reset(self):
        RoboschoolForwardWalkerMujocoXML.robot_specific_reset(self)
        self.motor_names  = ["abdomen_z", "abdomen_y", "abdomen_x"]
        self.motor_power  = [100, 100, 100]
        self.motor_names += ["right_hip_x", "right_hip_z", "right_hip_y", "right_knee"]
        self.motor_power += [100, 100, 300, 200]
        self.motor_names += ["left_hip_x", "left_hip_z", "left_hip_y", "left_knee"]
        self.motor_power += [100, 100, 300, 200]
        self.motor_names += ["right_shoulder1", "right_shoulder2", "right_elbow"]
        self.motor_power += [75, 75, 75]
        self.motor_names += ["left_shoulder1", "left_shoulder2", "left_elbow"]
        self.motor_power += [75, 75, 75]
        self.motors = [self.jdict[n] for n in self.motor_names]
        self.humanoid_task()

    def humanoid_task(self):
        self.set_initial_orientation(self.TASK_WALK, yaw_center=0, yaw_random_spread=np.pi/16)

    def set_initial_orientation(self, task, yaw_center, yaw_random_spread):
        self.task = task
        cpose = cpp_household.Pose()
        yaw = yaw_center + self.np_random.uniform(low=-yaw_random_spread, high=yaw_random_spread)
        if task==self.TASK_WALK:
            pitch = 0
            roll = 0
            cpose.set_xyz(self.start_pos_x, self.start_pos_y, self.start_pos_z + 1.4)
        elif task==self.TASK_STAND_UP:
            pitch = np.pi/2
            roll = 0
            cpose.set_xyz(self.start_pos_x, self.start_pos_y, self.start_pos_z + 0.45)
        elif task==self.TASK_ROLL_OVER:
            pitch = np.pi*3/2 - 0.15
            roll = 0
            cpose.set_xyz(self.start_pos_x, self.start_pos_y, self.start_pos_z + 0.22)
        else:
            assert False
        cpose.set_rpy(roll, pitch, yaw)
        self.cpp_robot.set_pose_and_speed(cpose, 0,0,0)
        self.initial_z = 0.8

    def apply_action(self, a):
        assert( np.isfinite(a).all() )
        for i, m, power in zip(range(len(self.motors)), self.motors, self.motor_power):
            m.set_motor_torque( float(power*self.power*np.clip(a[i], -1, +1)) )

    def alive_bonus(self, z, pitch):
        return +2 if z > 0.78 else -1   # 2 here because 17 joints produce a lot of electricity cost just from policy noise, living must be better than dying

class RoboschoolHumanoidBullet3(RoboschoolForwardWalkerMujocoXML):
    foot_list = ["right_foot", "left_foot"]

    def __init__(self, model_xml='humanoid.xml'):
        RoboschoolForwardWalkerMujocoXML.__init__(self, model_xml, 'torso', action_dim=21, obs_dim=52, power=0.41)
        # 21 joints, 6 of them important for walking (hip, knee, ankle), others may as well be turned off, 21/6 = 3.5
        self.electricity_cost  = 3.5*RoboschoolForwardWalkerMujocoXML.electricity_cost
        self.stall_torque_cost = 3.5*RoboschoolForwardWalkerMujocoXML.stall_torque_cost
        self.initial_z = 0.8

    def robot_specific_reset(self):
        RoboschoolForwardWalkerMujocoXML.robot_specific_reset(self)
        self.motor_names  = ["abdomen_z", "abdomen_y", "abdomen_x"]
        self.motor_power  = [100, 100, 100]
        self.motor_names += ["right_hip_x", "right_hip_z", "right_hip_y", "right_knee"]
        self.motor_power += [100, 100, 300, 200]
        self.motor_names += ["right_ankle_x", "right_ankle_y"]
        self.motor_power += [50, 50]
        self.motor_names += ["left_hip_x", "left_hip_z", "left_hip_y", "left_knee"]
        self.motor_power += [100, 100, 300, 200]
        self.motor_names += ["left_ankle_x", "left_ankle_y"]
        self.motor_power += [50, 50]
        self.motor_names += ["right_shoulder1", "right_shoulder2", "right_elbow"]
        self.motor_power += [75, 75, 75]
        self.motor_names += ["left_shoulder1", "left_shoulder2", "left_elbow"]
        self.motor_power += [75, 75, 75]
        self.motors = [self.jdict[n] for n in self.motor_names]
        self.humanoid_task()

    def humanoid_task(self):
        self.set_initial_orientation(yaw_center=0, yaw_random_spread=np.pi/16)

    def set_initial_orientation(self, yaw_center, yaw_random_spread):
        cpose = cpp_household.Pose()
        yaw = yaw_center + self.np_random.uniform(low=-yaw_random_spread, high=yaw_random_spread)
        pitch = 0
        roll = 0
        cpose.set_xyz(self.start_pos_x, self.start_pos_y, self.start_pos_z + 1.4)
        cpose.set_rpy(roll, pitch, yaw)
        self.cpp_robot.set_pose_and_speed(cpose, 0,0,0)
        self.initial_z = 0.8

    def apply_action(self, a):
        assert( np.isfinite(a).all() )
        for i, m, power in zip(range(len(self.motors)), self.motors, self.motor_power):
            m.set_motor_torque( float(power*self.power*np.clip(a[i], -1, +1)) )

    def alive_bonus(self, z, pitch):
        return +2 if z > 0.78 else -1   # 2 here because 21 joints produce a lot of electricity cost just from policy noise, living must be better than dying

class RoboschoolHumanoidBullet3Experimental(RoboschoolHumanoidBullet3):
    def __init__(self, model_xml='humanoid.xml', reward_type='walk'):
        RoboschoolHumanoidBullet3.__init__(self, model_xml)
        self.reward_type = reward_type

        if self.reward_type == "dm_control":
            self.stand_height = 1.4
            self.move_speed = 10  # Run task

        if self.reward_type in ("walk", "walk_slow", "walk_target", "walk_slow_target"):
            data = np.load(os.path.join(os.path.dirname(__file__), "data/cmu_mocap_walk.npz"))
            self.qpos = data['qpos']
            self.obs = data['obs']
            if self.reward_type in ("walk", "walk_target"):
                ind = 0
            if self.reward_type in ("walk_slow", "walk_slow_target"):
                ind = 3
            self.rstep = data['rstep'][data['rstep'][:,0] == ind]
            self.lstep = data['lstep'][data['lstep'][:,0] == ind]

        if self.reward_type in ("turn_left", "turn_right", "turn_left_from_walk_slow_target", "turn_right_from_walk_slow_target"):
            data = np.load(os.path.join(os.path.dirname(__file__), "data/cmu_mocap_turn.npz"))
            self.qpos = data['qpos']
            self.obs = data['obs']
            if self.reward_type in ("turn_left", "turn_left_from_walk_slow_target"):
                self.turn = data['lturn']
            if self.reward_type in ("turn_right", "turn_right_from_walk_slow_target"):
                self.turn = data['rturn']
            if self.reward_type in ("turn_left_from_walk_slow_target", "turn_right_from_walk_slow_target"):
                # TODO: commit sample file
                idata = np.load(os.path.join(os.path.dirname(__file__), "data/sample_walk_slow_target.npz"))
                self.iqpos = idata['qpos']

        if self.reward_type in ("sit", "sit_from_turn"):
            data = np.load(os.path.join(os.path.dirname(__file__), "data/cmu_mocap_sit.npz"))
            self.qpos = data['qpos']
            self.obs = data['obs']
            self.sitd = data['sitd']
            self.offsetz = data['offsetz']
            self.offsetx = data['offsetx']
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(57,))
            if self.reward_type == "sit_from_turn":
                # TODO: commit sample file
                idata1 = np.load(os.path.join(os.path.dirname(__file__), "data/sample_turn_left_from_walk_slow_target.npz"))
                idata2 = np.load(os.path.join(os.path.dirname(__file__), "data/sample_turn_right_from_walk_slow_target.npz"))
                self.iqpos = np.concatenate((idata1['qpos'],idata2['qpos']))

        if self.reward_type == "holistic":
            data1 = np.load(os.path.join(os.path.dirname(__file__), "data/cmu_mocap_holistic.npz"))
            self.qpos1 = data1['qpos']
            self.obs1 = data1['obs']
            self.holist = data1['holist']
            data2 = np.load(os.path.join(os.path.dirname(__file__), "data/cmu_mocap_sit.npz"))
            self.qpos2 = data2['qpos']
            self.obs2 = data2['obs']
            self.sitd = data2['sitd']
            self.offsetz = data2['offsetz']
            self.offsetx = data2['offsetx']

    def create_single_player_scene(self):
        scene = super().create_single_player_scene()

        if self.reward_type in ("walk_target", "walk_slow_target"):
            scene.zero_at_running_strip_start_line = False

        return scene

    def humanoid_task(self):
        self.scene.stadium = self.scene.cpp_world.load_thingy(
            os.path.join(os.path.dirname(__file__), "models_outdoor/stadium/plane100.obj"),
            self.scene.stadium.pose(), 1.0, 0, 0xFFFFFF, True)
        self.pre_joint_pos = None
        self.pre_torso_pos = None
        self.initial_z = 0.8

        if self.reward_type in ("walk", "walk_slow", "walk_target", "walk_slow_target"):
            self._reset_expert(foot='r', ind=0)
            self._reset_robot_pose_and_speed(self.expert_qpos[0])
        elif self.reward_type in ("turn_left", "turn_right", "turn_left_from_walk_slow_target", "turn_right_from_walk_slow_target"):
            self._reset_expert()
            self._reset_robot_pose_and_speed(self.expert_qpos[0])
        elif self.reward_type in ("sit", "sit_from_turn"):
            self._reset_expert()
            self._reset_robot_pose_and_speed(self.expert_qpos[0])
        elif self.reward_type == "holistic":
            self._reset_expert()
            self._reset_robot_pose_and_speed(self.expert_qpos[0])
        else:
            super().humanoid_task()

        if self.reward_type in ("walk_target", "walk_slow_target"):
            self.target_reposition()

        if self.reward_type in ("sit", "sit_from_turn", "holistic"):
            self._reset_chair()

    def _reset_expert(self, **kwargs):
        if self.reward_type in ("walk", "walk_slow", "walk_target", "walk_slow_target"):
            assert kwargs['foot'] == 'r' or kwargs['foot'] == 'l'
            if kwargs['foot'] == 'r':
                s = self.np_random.randint(len(self.rstep)) if 'ind' not in kwargs else kwargs['ind']
                s = self.rstep[s]
            if kwargs['foot'] == 'l':
                s = self.np_random.randint(len(self.lstep)) if 'ind' not in kwargs else kwargs['ind']
                s = self.lstep[s]
            self.cur_foot = kwargs['foot']
            qpos = self.qpos[s[0]][s[1]:s[1] + s[2] + 1].copy()
            # Move root of the first frame to the 2D origin
            qpos[:, -9:-7] -= qpos[0, -9:-7]

        if self.reward_type in ("turn_left", "turn_right", "turn_left_from_walk_slow_target", "turn_right_from_walk_slow_target"):
            s = self.turn[2]
            qpos = self.qpos[s[0]][s[1]:s[1] + s[2] + 1].copy()
            # Move root of the first frame to the 2D origin and set yaw randomly
            yaw = self.np_random.uniform(-np.pi, np.pi)
            R = self._rpy2xmat(0, 0, yaw)
            qpos[:, -4] += yaw
            qpos[:, -9:-7] = (qpos[:, -9:-7] - qpos[0, -9:-7]).dot(R[:2,:2])
            qpos[:, -3:-1] = qpos[:, -3:-1].dot(R[:2,:2])
            if self.reward_type in ("turn_left_from_walk_slow_target", "turn_right_from_walk_slow_target"):
                # Sample an ending pose from walk slow target replace the first pose
                i = [self.np_random.randint(x) for x in self.iqpos.shape[:2]]
                iqpos = self.iqpos[i[0]][i[1]].copy()
                yaw = qpos[0, -4] - iqpos[-4]
                R = self._rpy2xmat(0, 0, yaw)
                iqpos[-4] += yaw
                iqpos[-9:-7] = 0
                iqpos[-3:-1] = iqpos[-3:-1].dot(R[:2,:2])
                qpos[0] = iqpos

        if self.reward_type in ("sit", "sit_from_turn", "holistic"):
            s = self.sitd[0]
            if self.reward_type in ("sit", "sit_from_turn"):
                qpos = self.qpos[s[0]][s[1]:s[1] + s[2] + 1].copy()
            if self.reward_type == "holistic":
                qpos = self.qpos2[s[0]][s[1]:s[1] + s[2] + 1].copy()
            # Move root of the first frame to the 2D origin
            qpos[:, -4] += np.pi/2
            R = self._rpy2xmat(0, 0, np.pi/2)
            qpos[:, -9:-7] = (qpos[:, -9:-7] - qpos[0, -9:-7]).dot(R[:2,:2])
            qpos[:, -3:-1] = qpos[:, -3:-1].dot(R[:2,:2])
            # Apply offsets to align foot position
            qpos[:, -9] -= self.offsetx[0][0]
            qpos[:, -7] -= self.offsetz[0][0]
            qpos[1:, -3] = (qpos[1:, -9] - qpos[:-1, -9]) / 0.0165
            qpos[1:, -1] = (qpos[1:, -7] - qpos[:-1, -7]) / 0.0165
            # Set position in chair centered coordinates
            if self.reward_type in ("sit", "holistic"):
                qpos[:, -9] += 0.4000
            if self.reward_type == "sit_from_turn":
                # Add small variation to starting 2D position and yaw
                dx = self.np_random.normal(0.5000, 1/30)
                dy = self.np_random.normal(0, 1/10)
                dr = np.arctan2(dy, dx) + self.np_random.normal(0, np.pi/27)
                qpos[:, -4] += dr
                R = self._rpy2xmat(0, 0, dr)
                qpos[:, -9:-7] = qpos[:, -9:-7].dot(R[:2,:2])
                qpos[:, -3:-1] = qpos[:, -3:-1].dot(R[:2,:2])
                qpos[:, -9] += dx
                qpos[:, -8] += dy
                # Sample an ending pose from turn left and append to start
                i = [self.np_random.randint(x) for x in self.iqpos.shape[:2]]
                iqpos = self.iqpos[i[0]][i[1]].copy()
                yaw = qpos[0, -4] - iqpos[-4]
                R = self._rpy2xmat(0, 0, yaw)
                iqpos[-4] += yaw
                iqpos[-9:-7] = [dx, dy]
                iqpos[-3:-1] = iqpos[-3:-1].dot(R[:2,:2])
                qpos = np.concatenate((iqpos[None], qpos))

        if self.reward_type == "holistic":
            s = self.holist[0]
            qpos1 = self.qpos1[s[0]][s[1]:s[1] + s[2] + 1].copy()
            # Move root of the first frame to the 2D origin
            yaw = -qpos1[-1, -4]
            qpos1[:, -4] += yaw
            R = self._rpy2xmat(0, 0, yaw)
            qpos1[:, -9:-7] = (qpos1[:, -9:-7] - qpos1[0, -9:-7]).dot(R[:2,:2])
            qpos1[:, -3:-1] = qpos1[:, -3:-1].dot(R[:2,:2])
            # Move root so that it will be just in front of the chair before sitting down
            qpos1[:, -9:-7] -= qpos1[180, -9:-7] - [0.4000, -0.1000]
            # Extract relevant segment for holistic
            qpos1 = qpos1[80:160]
            # Extract relevant segment for sit
            qpos2 = qpos[35:]
            # Concatenate holistic and sit
            qpos = np.concatenate((qpos1, qpos2))

        self.expert_qpos = qpos
        self.expert_step = 0

    def _reset_robot_pose_and_speed(self, qpos):
        for j, joint in enumerate(self.ordered_joints):
            joint.reset_current_position(qpos[2*j], qpos[2*j+1])
        cpose = cpp_household.Pose()
        cpose.set_xyz(*qpos[-9:-6])
        cpose.set_rpy(*qpos[-6:-3])
        self.cpp_robot.set_pose_and_speed(cpose, *qpos[-3:])

    def _reset_chair(self):
        pose = cpp_household.Pose()
        pose.set_xyz(0, 0, 0.4105)
        pose.set_rpy(np.pi/2, 0, 0)
        self.urdf = self.scene.cpp_world.load_urdf(
            os.path.join(os.path.dirname(__file__), "models_household/chair/chair.urdf"),
            pose, False, True)
        self.urdf.set_pose_and_speed(pose, 0, 0, 0)
        self.urdf.query_position()
        self.sit_target_pos = np.array([0, 0, 0.4015], dtype=np.float32)

    def target_reposition(self):
        theta = self.np_random.uniform(-np.pi/4, np.pi/4)
        dist = self.np_random.uniform(1, 5)
        px, py = self.robot_body.pose().xyz()[:2]
        vx, vy = self._rpy2xmat(*self.robot_body.pose().rpy())[0, :2]
        dx = (np.cos(theta) * vx - np.sin(theta) * vy) * dist
        dy = (np.sin(theta) * vx + np.cos(theta) * vy) * dist
        self.walk_target_x = px + dx
        self.walk_target_y = py + dy

        self.target = None
        self.target = self.scene.cpp_world.debug_sphere(self.walk_target_x, self.walk_target_y, 0.2, 0.1, 0xFF8080)
        self.target_timeout = 151

    def calc_state(self):
        state = super().calc_state()

        if self.pre_joint_pos is None:
            self.pre_joint_pos = np.array([j.current_position()[0] for j in self.ordered_joints], dtype=np.float32)
        if self.pre_torso_pos is None:
            self.pre_torso_pos = np.array(self.robot_body.pose().xyz(), dtype=np.float32)

        if self.reward_type == "dm_control":
            self.head_height = self.robot_body.pose().xyz()[2] + 0.28
            self.torso_xmat = self._rpy2xmat(*self.robot_body.pose().rpy())

        if self.reward_type in ("walk_target", "walk_slow_target"):
            self.target_timeout -= 1
            if self.walk_target_dist < 0.2 or self.target_timeout <= 0:
                self.target_reposition()
                state = super().calc_state()
                self.potential = self.calc_potential()       # avoid reward jump

        j = np.array([j.current_relative_position() for j in self.ordered_joints], dtype=np.float32).flatten()
        z = self.body_xyz[2] - self.initial_z
        r, p, _ = self.body_rpy
        v = 0.3 * np.dot(self.rot_minus_yaw, self.robot_body.speed())
        more = np.array([z, r, p, *v], dtype=np.float32)

        if self.reward_type in ("walk", "walk_slow", "walk_target", "walk_slow_target"):
            goal = np.array([np.sin(self.angle_to_target), np.cos(self.angle_to_target)], dtype=np.float32)
            state = np.clip( np.concatenate([more] + [j] + [self.feet_contact] + [goal]), -5, +5)

        if self.reward_type in ("turn_left", "turn_right", "turn_left_from_walk_slow_target", "turn_right_from_walk_slow_target"):
            goal = np.array([0.0, 0.0], dtype=np.float32)
            state = np.clip( np.concatenate([more] + [j] + [self.feet_contact] + [goal]), -5, +5)

        if self.reward_type in ("sit", "sit_from_turn"):
            self.sit_target_dist = np.linalg.norm( self.parts['pelvis'].pose().xyz() - self.sit_target_pos )

            R = self._rpy2xmat(*self.robot_body.pose().rpy())
            p = R.dot(self.sit_target_pos - self.parts['pelvis'].pose().xyz())
            q = self.robot_body.pose().quatertion()
            q = (q * np.array([[-1, -1, -1, 1]]))[0]
            state = np.clip( np.concatenate([more] + [j] + [self.feet_contact] + [p] + [q]), -5, +5)

        if self.reward_type == "holistic":
            self.sit_target_dist = np.linalg.norm( self.parts['pelvis'].pose().xyz() - self.sit_target_pos )

            goal = np.array([0.0, 0.0], dtype=np.float32)
            state = np.clip( np.concatenate([more] + [j] + [self.feet_contact] + [goal]), -5, +5)

        return state

    def _rpy2xmat(self, r, p, y):
        Rr = np.array(
            [[1,          0,           0],
             [0, np.cos(-r), -np.sin(-r)],
             [0, np.sin(-r),  np.cos(-r)]]
            )
        Rp = np.array(
            [[ np.cos(-p), 0, np.sin(-p)],
             [          0, 1,          0],
             [-np.sin(-p), 0, np.cos(-p)]]
            )
        Ry = np.array(
            [[np.cos(-y), -np.sin(-y), 0],
             [np.sin(-y),  np.cos(-y), 0],
             [         0,           0, 1]]
            )
        return Rr.dot(Rp.dot(Ry))

    def calc_potential(self):
        if self.reward_type in ("sit", "sit_from_turn", "holistic"):
            return - self.sit_target_dist / self.scene.dt
        else:
            return super().calc_potential()

    def _step(self, a):
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.apply_action(a)
            self.scene.global_step()

        state = self.calc_state()

        alive = float(self.alive_bonus(state[0]+self.initial_z, self.body_rpy[1]))   # state[0] is body height above ground, body_rpy[1] is pitch
        done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        for i,f in enumerate(self.feet):
            contact_names = set(x.name for x in f.contact_list())
            self.feet_contact[i] = 1.0 if (self.foot_ground_object_names & contact_names) else 0.0

        cur_joint_pos = np.array([j.current_position()[0] for j in self.ordered_joints], dtype=np.float32)
        cur_torso_pos = np.array(self.robot_body.pose().xyz(), dtype=np.float32)
        cur_torso_rot = np.array(self.robot_body.pose().rpy(), dtype=np.float32)

        if self.reward_type == "dm_control":
            standing = rewards.tolerance(self.head_height,
                                         bounds=(self.stand_height, float('inf')),
                                         margin=self.stand_height/4)
            upright = rewards.tolerance(self.torso_xmat[2,2],
                                        bounds=(0.9, float('inf')), sigmoid='linear',
                                        margin=1.9, value_at_margin=0)
            stand_reward = standing * upright
            small_control = rewards.tolerance(a, margin=1,
                                              value_at_margin=0,
                                              sigmoid='quadratic').mean()
            small_control = (4 + small_control) / 5
            # Run task
            torso_velocity = np.linalg.norm(np.array(self.robot_body.speed())[[0,1]])
            move = rewards.tolerance(torso_velocity,
                                     bounds=(self.move_speed, float('inf')),
                                     margin=self.move_speed, value_at_margin=0,
                                     sigmoid='linear')
            move = (5*move + 1) / 6
            self.rewards = [small_control * stand_reward * move]

        if self.reward_type in ("walk", "walk_slow", "walk_target", "walk_slow_target"):
            self.expert_step += 1
            r_joint_pos = self._reward_joint_pos(cur_joint_pos, 1.0000)
            r_joint_vel = self._reward_joint_vel(cur_joint_pos, 0.0100)

            if self.reward_type in ("walk", "walk_slow"):
                if self.reward_type == "walk":
                    r_torso_vel = self._reward_torso_vel(cur_torso_pos, 1.0000)
                if self.reward_type == "walk_slow":
                    r_torso_vel = self._reward_torso_vel(cur_torso_pos, 10.0000)
                self.rewards = [0.5000 * r_joint_pos, 0.0500 * r_joint_vel, 0.1000 * r_torso_vel]

            if self.reward_type in ("walk_target", "walk_slow_target"):
                potential_old = self.potential
                self.potential = self.calc_potential()
                r_target = self._reward_target(potential_old, 10.0000)
                self.rewards = [0.5000 * r_joint_pos, 0.0500 * r_joint_vel, 0.5000 * r_target]

            if self.expert_step == len(self.expert_qpos) - 1:
                if self.cur_foot == 'r':
                    self._reset_expert(foot='l')
                else:
                    self._reset_expert(foot='r')

        if self.reward_type in ("turn_left", "turn_right", "turn_left_from_walk_slow_target", "turn_right_from_walk_slow_target"):
            if self.expert_step == len(self.expert_qpos) - 1:
                r_joint_pos = self._reward_joint_pos(cur_joint_pos,  1.0000)
                r_torso_rot = self._reward_torso_rot(cur_torso_rot, 10.0000)
                r_ecost = self._reward_ecost(a)
                self.rewards = [0.5000 * r_joint_pos, 0.1000 * r_torso_rot, 0.5000 * r_ecost]
            else:
                self.expert_step += 1
                r_joint_pos = self._reward_joint_pos(cur_joint_pos,  1.0000)
                r_joint_vel = self._reward_joint_vel(cur_joint_pos,  0.0100)
                r_torso_vel = self._reward_torso_vel(cur_torso_pos, 10.0000)
                r_torso_rot = self._reward_torso_rot(cur_torso_rot, 10.0000)
                self.rewards = [0.5000 * r_joint_pos, 0.0500 * r_joint_vel, 0.1000 * r_torso_vel, 0.1000 * r_torso_rot]

            done = done or self._done_torso_rot(cur_torso_rot, np.pi / 4)

        if self.reward_type in ("sit", "sit_from_turn", "holistic"):
            if self.expert_step == len(self.expert_qpos) - 1:
                r_joint_pos = self._reward_joint_pos(cur_joint_pos, 1.0000)
                r_ecost = self._reward_ecost(a)
                self.rewards = [0.5000 * r_joint_pos, 0.1000 * r_ecost]
            else:
                self.expert_step += 1
                r_joint_pos = self._reward_joint_pos(cur_joint_pos, 1.0000)
                r_joint_vel = self._reward_joint_vel(cur_joint_pos, 0.0100)
                potential_old = self.potential
                self.potential = self.calc_potential()
                r_target = float(self.potential - potential_old)
                if self.reward_type in ("sit", "sit_from_turn"):
                    self.rewards = [0.5000 * r_joint_pos, 0.0500 * r_joint_vel, 0.5000 * r_target]
                if self.reward_type == "holistic":
                    self.rewards = [0.5000 * r_joint_pos, 0.0500 * r_joint_vel, 0.1000 * r_target]

            done = state[0]+self.initial_z < 0.54 or not np.isfinite(state).all()
            if self.reward_type == "holistic":
                done = done or self._done_torso_rot(cur_torso_rot, np.pi / 4)

        self.pre_joint_pos = cur_joint_pos
        self.pre_torso_pos = cur_torso_pos

        self.frame += 1
        if (done and not self.done) or self.frame==self.spec.timestep_limit:
            self.episode_over(self.frame)
        self.done += done   # 2 == 1+True
        self.reward += sum(self.rewards)
        self.HUD(state, a, done)

        return state, sum(self.rewards), bool(done), {}

    def _reward_joint_pos(self, cur_joint_pos, w):
        act_joint_pos = cur_joint_pos
        ref_joint_pos = self.expert_qpos[self.expert_step, 0:2*len(self.ordered_joints):2]
        return np.exp(-w * np.sum((act_joint_pos - ref_joint_pos)**2))

    def _reward_joint_vel(self, cur_joint_pos, w):
        act_joint_vel = (cur_joint_pos - self.pre_joint_pos) / 0.0165
        ref_joint_vel = (self.expert_qpos[self.expert_step, 0:2*len(self.ordered_joints):2] -
                         self.expert_qpos[self.expert_step - 1, 0:2*len(self.ordered_joints):2]) / 0.0165
        return np.exp(-w * np.sum((act_joint_vel - ref_joint_vel)**2))

    def _reward_torso_vel(self, cur_torso_pos, w):
        act_torso_vel = (cur_torso_pos - self.pre_torso_pos) / 0.0165
        ref_torso_vel = (self.expert_qpos[self.expert_step, -9:-6] -
                         self.expert_qpos[self.expert_step - 1, -9:-6]) / 0.0165
        return np.exp(-w * np.sum((act_torso_vel - ref_torso_vel)**2))

    def _reward_torso_rot(self, cur_torso_rot, w):
        act_torso_rot = cur_torso_rot
        ref_torso_rot = self.expert_qpos[self.expert_step, -6:-3]
        return np.exp(-w * np.sum(self._ang_diff(act_torso_rot, ref_torso_rot)**2))

    def _reward_target(self, potential_old, w):
        progress = float(self.potential - potential_old)
        return 1 / (1 + np.exp(-w * progress))

    def _reward_ecost(self, a):
        electricity_cost  = self.electricity_cost  * float(np.abs(a*self.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())
        return electricity_cost

    def _done_torso_rot(self, cur_torso_rot, thresh):
        act_torso_rot = cur_torso_rot
        ref_torso_rot = self.expert_qpos[self.expert_step, -6:-3]
        return np.abs(self._ang_diff(cur_torso_rot, ref_torso_rot)[2]) > thresh

    def _ang_diff(self, a, b):
        return (a - b) - (((a - b) - np.pi) // (2 * np.pi) + 1) * (2 * np.pi)

class RoboschoolHumanoidBullet3ExperimentalTrainingWrapper(RoboschoolHumanoidBullet3Experimental):
    def __init__(self, model_xml='humanoid.xml', reward_type='walk'):
        RoboschoolHumanoidBullet3Experimental.__init__(self, model_xml, reward_type)

    def humanoid_task(self):
        self.scene.stadium = self.scene.cpp_world.load_thingy(
            os.path.join(os.path.dirname(__file__), "models_outdoor/stadium/plane100.obj"),
            self.scene.stadium.pose(), 1.0, 0, 0xFFFFFF, True)
        self.pre_joint_pos = None
        self.pre_torso_pos = None
        self.initial_z = 0.8

        # Enables a different iniitialization strategy during training
        if self.reward_type in ("sit", "sit_from_turn", "holistic"):
            self._reset_expert()
            # Randomly sample a starting frame
            if self.reward_type == "sit":
                s = self.np_random.randint(0, 83)
            if self.reward_type in ("sit_from_turn", "holistic"):
                if self.np_random.uniform() < 0.50:
                    s = 0
                else:
                    if self.reward_type == "sit_from_turn":
                        s = self.np_random.randint(1, 84)
                    if self.reward_type == "holistic":
                        s = self.np_random.randint(80, 128)
            self.expert_qpos = self.expert_qpos[s:]
            # Ignore joint velocity
            if self.reward_type == "sit" or s > 0:
                self.expert_qpos[:, 1:42:2] = 0
            self._reset_robot_pose_and_speed(self.expert_qpos[0])
            self._reset_chair()
        else:
            super().humanoid_task()

class RoboschoolHumanoidBullet3HighLevelExperimental(RoboschoolHumanoidBullet3Experimental):
    def __init__(self, model_xml='humanoid.xml', reward_type='walk_slow_easy'):
        RoboschoolHumanoidBullet3Experimental.__init__(self, model_xml, reward_type)

        self.qpos = {}
        data = np.load(os.path.join(os.path.dirname(__file__), "data/cmu_mocap_walk.npz"))
        self.qpos['walk'] = data['qpos']
        ind = 3
        self.rstep = data['rstep'][data['rstep'][:,0] == ind]
        self.lstep = data['lstep'][data['lstep'][:,0] == ind]

        # Should set max_episode_steps in __init__.py to a mulitple of timestep,
        # otherwise the reward will be nan at max_episode_steps.
        self.timestep = 30

        if self.reward_type in ("walk_slow_easy", "walk_slow_hard"):
            self.action_space = gym.spaces.Dict(dict(
                meta=gym.spaces.Dict(dict(
                    switch=gym.spaces.Discrete(1),
                    walk=gym.spaces.Box(-np.inf, np.inf, shape=(2,)),
                )),
                walk=gym.spaces.Box(-1, 1, shape=(21,)),
            ))
            self.observation_space = gym.spaces.Dict(dict(
                switch=gym.spaces.Box(-1, 0, shape=(), dtype=np.int32),
                meta=gym.spaces.Box(-np.inf, np.inf, shape=(57,)),
                walk=gym.spaces.Box(-np.inf, np.inf, shape=(52,)),
            ))

    def humanoid_task(self):
        self.scene.stadium = self.scene.cpp_world.load_thingy(
            os.path.join(os.path.dirname(__file__), "models_outdoor/stadium/plane100.obj"),
            self.scene.stadium.pose(), 1.0, 0, 0xFFFFFF, True)
        self.walk_target_x = 0
        self.walk_target_y = 0
        self.initial_z = 0.8
        self.on_support = False

        if self.reward_type in ("walk_slow_easy", "walk_slow_hard"):
            s = self.rstep[0]
            qpos = self.qpos['walk'][s[0]][s[1]:s[1] + s[2] + 1][0].copy()
            theta = self.np_random.uniform(-np.pi, np.pi)
            rad = self.np_random.uniform(2, 5)
            px = np.cos(theta) * rad
            py = np.sin(theta) * rad
            if self.reward_type == "walk_slow_easy":
                yaw = theta + np.pi + self.np_random.uniform(low=-np.pi/4, high=np.pi/4)
            if self.reward_type == "walk_slow_hard":
                yaw = theta + np.pi + self.np_random.uniform(low=-np.pi, high=np.pi)
            R = self._rpy2xmat(0, 0, yaw)
            qpos[-4] += yaw
            qpos[-9:-7] = [px, py]
            qpos[-3:-1] = qpos[-3:-1].dot(R[:2,:2])

        self._reset_robot_pose_and_speed(qpos)
        self._reset_chair()

    def calc_state(self, step=False):
        state = RoboschoolHumanoidBullet3.calc_state(self)

        j = np.array([j.current_relative_position() for j in self.ordered_joints], dtype=np.float32).flatten()
        z = self.body_xyz[2] - self.initial_z
        r, p, _ = self.body_rpy
        v = 0.3 * np.dot(self.rot_minus_yaw, self.robot_body.speed())
        more = np.array([z, r, p, *v], dtype=np.float32)

        self.sit_target_dist = np.linalg.norm( self.parts['pelvis'].pose().xyz() - self.sit_target_pos )

        R = self._rpy2xmat(*self.robot_body.pose().rpy())
        p = R.dot(self.sit_target_pos - self.parts['pelvis'].pose().xyz())
        q = self.robot_body.pose().quatertion()
        q = (q * np.array([[-1, -1, -1, 1]]))[0]

        if self.reward_type in ("walk_slow_easy", "walk_slow_hard"):
            goal = np.array([np.sin(self.angle_to_target), np.cos(self.angle_to_target)], dtype=np.float32)
            state = {
                'meta': np.clip( np.concatenate([more] + [j] + [self.feet_contact] + [p] + [q]), -5, +5),
                'walk': np.clip( np.concatenate([more] + [j] + [self.feet_contact] + [goal]), -5, +5),
            }

        # Currently step == False is expected to occurs only during _reset(),
        # but this should also handle more general cases (i.e. self.frame != 0).
        if step:
            state['switch'] = -1 if (self.frame + 1) % self.timestep == 0 else self.cur_switch
        else:
            state['switch'] = -1 if self.frame % self.timestep == 0 else self.cur_switch

        return state

    def calc_potential(self):
        return - self.sit_target_dist / (self.scene.dt * self.timestep)

    def _step(self, a):
        self.cur_switch, a, goal = a

        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.apply_action(a)
            self.scene.global_step()

        if self.frame % self.timestep == 0:
            theta, dist = goal
            px, py = self.robot_body.pose().xyz()[:2]
            vx, vy = self._rpy2xmat(*self.robot_body.pose().rpy())[0, :2]
            dx = (np.cos(theta) * vx - np.sin(theta) * vy) * dist
            dy = (np.sin(theta) * vx + np.cos(theta) * vy) * dist
            self.walk_target_x = px + dx
            self.walk_target_y = py + dy

        state = self.calc_state(step=True)

        alive = float(self.alive_bonus(self.body_xyz[2], self.body_rpy[1]))   # state[0] is body height above ground, body_rpy[1] is pitch
        done = alive < 0
        if not all([np.isfinite(state[k]).all() for k in state]):
            print("~INF~")
            done = True

        for i,f in enumerate(self.feet):
            contact_names = set(x.name for x in f.contact_list())
            self.feet_contact[i] = 1.0 if (self.foot_ground_object_names & contact_names) else 0.0

        self.on_support = self.parts['pelvis'].pose().xyz()[2] < 0.54 and "chair" in [x.name for x in self.parts['pelvis'].contact_list()]

        if (self.frame + 1) % self.timestep == 0 or done:
            if self.on_support:
                self.rewards = [1.0]
            else:
                potential_old = self.potential
                self.potential = self.calc_potential()
                r_target = float(self.potential - potential_old)
                self.rewards = [0.5000 * r_target]
        else:
            self.rewards = [0.0]

        reward = sum(self.rewards) if (self.frame + 1) % self.timestep == 0 or done else float('nan')

        self.frame += 1
        if (done and not self.done) or self.frame==self.spec.timestep_limit:
            self.episode_over(self.frame)
        self.done += done   # 2 == 1+True
        self.reward += sum(self.rewards)
        self.HUD(state['meta'], a, done)

        return state, reward, bool(done), {}

class RoboschoolHumanoidBullet3HighLevelExperimentalTrainingWrapper(RoboschoolHumanoidBullet3HighLevelExperimental):
    def __init__(self, model_xml='humanoid.xml', reward_type='walk_slow_easy'):
        RoboschoolHumanoidBullet3HighLevelExperimental.__init__(self, model_xml, reward_type)

    def humanoid_task(self):
        super().humanoid_task()
