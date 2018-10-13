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

    def __init__(self, model_xml='humanoid.xml', obs_dim=52):
        RoboschoolForwardWalkerMujocoXML.__init__(self, model_xml, 'torso', action_dim=21, obs_dim=obs_dim, power=0.41)
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
    def __init__(self, model_xml='humanoid.xml', obs_dim=52, reward_type='walk'):
        RoboschoolHumanoidBullet3.__init__(self, model_xml, obs_dim)
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

        if self.reward_type == "turn":
            data = np.load(os.path.join(os.path.dirname(__file__), "data/cmu_mocap_turn.npz"))
            self.qpos = data['qpos']
            self.obs = data['obs']
            self.turn = data['lturn']

    def create_single_player_scene(self):
        scene = super().create_single_player_scene()

        if self.reward_type in ("walk_target", "walk_slow_target"):
            scene.zero_at_running_strip_start_line = False

        return scene

    def humanoid_task(self):
        self.pre_joint_pos = None
        self.pre_torso_pos = None

        self.initial_z = 0.8

        if self.reward_type in ("walk", "walk_slow", "walk_target", "walk_slow_target"):
            self._reset_expert(foot='r', ind=0)
            self._reset_robot_pose_and_speed(self.expert_qpos[0])
        elif self.reward_type == "turn":
            self._reset_expert()
            self._reset_robot_pose_and_speed(self.expert_qpos[0])
        else:
            super().humanoid_task()

        if self.reward_type in ("walk_target", "walk_slow_target"):
            self.target_reposition()

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
        elif self.reward_type == "turn":
            s = self.turn[2]
            qpos = self.qpos[s[0]][s[1]:s[1] + s[2] + 1].copy()
            # Move root of the first frame to the 2D origin and set yaw randomly
            yaw = self.np_random.uniform(-np.pi, np.pi)
            R = self._rpy2xmat(0, 0, yaw)
            qpos[:, -4] += yaw
            qpos[:, -9:-7] = (qpos[:, -9:-7] - qpos[0, -9:-7]).dot(R[:2,:2])
            qpos[:, -3:-1] = qpos[:, -3:-1].dot(R[:2,:2])
        else:
            assert False

        self.expert_qpos = qpos
        self.expert_step = 0

    def _reset_robot_pose_and_speed(self, qpos):
        for j, joint in enumerate(self.ordered_joints):
            joint.reset_current_position(qpos[2*j], qpos[2*j+1])
        cpose = cpp_household.Pose()
        cpose.set_xyz(*qpos[-9:-6])
        cpose.set_rpy(*qpos[-6:-3])
        self.cpp_robot.set_pose_and_speed(cpose, *qpos[-3:])

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

        if self.reward_type == "turn":
            state = np.hstack((state[[0]], state[3:]))
            done = 1.0 if self.expert_step == len(self.expert_qpos) - 1 else 0.0
            state = np.hstack((state, [done]))

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

        if self.reward_type == "turn":
            r_joint_pos = self._reward_joint_pos(cur_joint_pos,  1.0000)
            r_torso_rot = self._reward_torso_rot(cur_torso_rot, 10.0000)

            if self.expert_step == len(self.expert_qpos) - 1:
                r_ecost = self._reward_ecost(a)
                self.rewards = [0.5000 * r_joint_pos, 0.1000 * r_torso_rot, 0.5000 * r_ecost]
            else:
                self.expert_step += 1
                r_joint_vel = self._reward_joint_vel(cur_joint_pos,  0.0100)
                r_torso_vel = self._reward_torso_vel(cur_torso_pos, 10.0000)
                self.rewards = [0.5000 * r_joint_pos, 0.0500 * r_joint_vel, 0.1000 * r_torso_vel, 0.1000 * r_torso_rot]

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
    def __init__(self, model_xml='humanoid.xml', obs_dim=52, reward_type='walk'):
        RoboschoolHumanoidBullet3Experimental.__init__(self, model_xml, obs_dim, reward_type)

    def humanoid_task(self):
        # Enables a different iniitialization strategy during training
        super().humanoid_task()
