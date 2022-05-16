"""
Robomimic Wrapper
TODO: render with nvidia-gpu
"""
import os
import math
from pathlib import Path
import h5py
import numpy as np
from filelock import FileLock
from copy import deepcopy

import gym
from gym.envs.registration import register

"""
NOTE:
Don't use this environmnet to train a policy online. This is for (1) get_dataset interface for offline RL and (2) only for offline RL evaluation.
  - the horizon is set differently to the original env
  - `done` is always false
Therefore, consider using the raw RoboSuiteEnv to train a policy online RL algorithms, such as SAC.
"""

"""
For max_episode_steps, please check
cat robomimic_exps/core/tool_hang/ph/low_dim/bc.json | python3 -c "import sys, json; print(json.load(sys.stdin)['experiment']['rollout'])"
"""

register(
    id='lift-low-mg-v0',
    entry_point='arq.modules.robomimic_env:RobomimicLowDimEnv',
    max_episode_steps=400,
    kwargs={'task':'lift','dataset_type':'mg','hdf5_type':'low_dim_dense'}
)

register(
    id='lift-low-mg-sparse-v0',
    entry_point='arq.modules.robomimic_env:RobomimicLowDimEnv',
    max_episode_steps=400,
    kwargs={'task':'lift','dataset_type':'mg','hdf5_type':'low_dim_sparse'}
)

register(
    id='lift-low-ph-v0',
    entry_point='arq.modules.robomimic_env:RobomimicLowDimEnv',
    max_episode_steps=400,
    kwargs={'task':'lift','dataset_type':'ph','hdf5_type':'low_dim'}
)

register(
    id='lift-low-mh-v0',
    entry_point='arq.modules.robomimic_env:RobomimicLowDimEnv',
    max_episode_steps=500,
    kwargs={'task':'lift','dataset_type':'mh','hdf5_type':'low_dim'}
)

register(
    id='can-low-mg-v0',
    entry_point='arq.modules.robomimic_env:RobomimicLowDimEnv',
    max_episode_steps=400,
    kwargs={'task':'can','dataset_type':'mg','hdf5_type':'low_dim_dense'}
)

register(
    id='can-low-mg-sparse-v0',
    entry_point='arq.modules.robomimic_env:RobomimicLowDimEnv',
    max_episode_steps=400,
    kwargs={'task':'can','dataset_type':'mg','hdf5_type':'low_dim_sparse'}
)

register(
    id='can-low-ph-v0',
    entry_point='arq.modules.robomimic_env:RobomimicLowDimEnv',
    max_episode_steps=400,
    kwargs={'task':'can','dataset_type':'ph','hdf5_type':'low_dim'}
)

register(
    id='can-low-mh-v0',
    entry_point='arq.modules.robomimic_env:RobomimicLowDimEnv',
    max_episode_steps=500,
    kwargs={'task':'can','dataset_type':'mh','hdf5_type':'low_dim'}
)

register(
    id='square-low-ph-v0',
    entry_point='arq.modules.robomimic_env:RobomimicLowDimEnv',
    max_episode_steps=400,
    kwargs={'task':'square','dataset_type':'ph','hdf5_type':'low_dim'}
)

register(
    id='square-low-mh-v0',
    entry_point='arq.modules.robomimic_env:RobomimicLowDimEnv',
    max_episode_steps=500,
    kwargs={'task':'square','dataset_type':'mh','hdf5_type':'low_dim'}
)

register(
    id='transport-low-ph-v0',
    entry_point='arq.modules.robomimic_env:RobomimicLowDimEnv',
    max_episode_steps=700,
    kwargs={'task':'transport','dataset_type':'ph','hdf5_type':'low_dim'}
)

register(
    id='transport-low-mh-v0',
    entry_point='arq.modules.robomimic_env:RobomimicLowDimEnv',
    max_episode_steps=1100,
    kwargs={'task':'transport','dataset_type':'mh','hdf5_type':'low_dim'}
)

register(
    id='toolhang-low-ph-v0',
    entry_point='arq.modules.robomimic_env:RobomimicLowDimEnv',
    max_episode_steps=700,
    kwargs={'task':'tool_hang','dataset_type':'ph','hdf5_type':'low_dim'}
)

##############################################
##############################################
##############################################
##############################################
##############################################
##############################################
import robosuite
assert robosuite.__version__.split(".")[0] == "1" and  (int(robosuite.__version__.split(".")[1]) >= 2), "only support robosuite v1.2+"

from robomimic import DATASET_REGISTRY
import robomimic.utils.file_utils as F
from arq.modules.env_utils import Trajectory

DEFAULT_BASE_DIR = os.environ.get('ROBOMIMIC_DATASET_DIR', os.path.expanduser('~/.robomimic/datasets'))

# Checkout https://github.com/ARISE-Initiative/robomimic/blob/490dd4d4f758c47357d314d47ffa5bb4aefa8d65/robomimic/scripts/download_datasets.py#L53
ALL_TASKS = ["lift", "can", "square", "transport", "tool_hang", "lift_real", "can_real", "tool_hang_real"] # robosuite env name: [Lift, PickPlaceCan, NutAssemblySquare, TwoArmTransport]
ALL_DATASET_TYPES = ["ph", "mh", "mg", "paired"]
ALL_HDF5_TYPES = ["raw", "low_dim", "image", "low_dim_sparse", "low_dim_dense", "image_sparse", "image_dense"] #_sparse / _dense is about reward in machine-generated type

# Modalities
# You can confirm the modality used by WhatMatters paper by executing the following scripts
# python -m robomimic.scripts.generate_paper_configs --config_dir ./tmp
# cat ./tmp/core/{tasks}/{dataset_type}/{hdf5_type}/bc.json | python3 -c "import sys, json; print(json.load(sys.stdin)['observation']['modalities'])"

# for general low_dim
## Also checkout https://github.com/ARISE-Initiative/robomimic/blob/490dd4d4f758c47357d314d47ffa5bb4aefa8d65/robomimic/scripts/generate_paper_configs.py#L67
DEFAULT_LOW_DIM_MODAL = {
    "obs": {"low_dim": ["robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos","object"],"rgb": []},
    "goal": {"low_dim": [], "rgb": []} }

# for transport low_dim environment
## Also checkout https://github.com/ARISE-Initiative/robomimic/blob/490dd4d4f758c47357d314d47ffa5bb4aefa8d65/robomimic/scripts/generate_paper_configs.py#L291
TRANSPORT_LOW_DIM_MODAL= {
    "obs": {
        "low_dim": ["robot0_eef_pos","robot0_eef_quat","robot0_gripper_qpos","robot1_eef_pos","robot1_eef_quat","robot1_gripper_qpos","object"],
        "rgb": []},
    "goal": {"low_dim": [],"rgb": []}}

def _get_modal(task,dataset_type,hdf5_type):
    if 'low_dim' in hdf5_type:
        if task == 'transport': return TRANSPORT_LOW_DIM_MODAL
        else: return DEFAULT_LOW_DIM_MODAL
    elif 'image' in hdf5_type:
        raise NotImplementedError
    else:
        assert hdf5_type == 'raw'
        raise NotImplementedError

class RobomimicLowDimEnv(gym.Env):
    @staticmethod
    def initialize(task,dataset_type,hdf5_type):
        download_dir = os.path.abspath(os.path.join(DEFAULT_BASE_DIR, task, dataset_type))
        url = DATASET_REGISTRY[task][dataset_type][hdf5_type]["url"]
        fname = url.split("/")[-1]

        Path(download_dir).mkdir(parents=True,exist_ok=True)

        dataset_path = Path(download_dir)/fname

        with FileLock(str(dataset_path) + '.lock') as lock:
            if not dataset_path.is_file():
                F.download_url(url=url,download_dir=download_dir)

        env_meta = F.get_env_metadata_from_dataset(dataset_path=dataset_path)
        shape_meta = {}

        with h5py.File(dataset_path, 'r') as f:
            ## Get official train-val split.
            try:
                mask_keys = f['mask'].keys()
            except KeyError:
                mask_keys = []

            env_splits = {
                key: f['mask'][key][...].astype(str)
                for key in mask_keys
            }
            assert 'all' not in env_splits.keys()
            env_splits['all'] = list(f['data'].keys())

            ## Get shape information
            ## https://github.com/ARISE-Initiative/robomimic/blob/490dd4d4f758c47357d314d47ffa5bb4aefa8d65/robomimic/utils/file_utils.py#L87
            demo_id = sorted(list(f["data"].keys()))[0]
            demo = f[f"data/{demo_id}"]

            # action dimension
            shape_meta['ac_dim'] = demo["actions"].shape[1:]
            shape_meta['ob_dims'] = {
                key: demo[f"obs/{key}"].shape[1:]
                for key in sorted([key for key in demo["obs"]])
            }       

        # Get appropriate data modality
        env_modal = _get_modal(task,dataset_type,hdf5_type)
        ob_dims = np.sum([math.prod(shape_meta['ob_dims'][key]) for key in env_modal['obs']['low_dim']])

        return dataset_path, env_meta, env_splits, env_modal, ob_dims

    def __init__(self,task,dataset_type,hdf5_type,render=False,render_offscreen=False,use_image_obs=False,reward_as_success=True,terminate_on_success=True):
        ## Checkout https://github.com/ARISE-Initiative/robomimic/blob/490dd4d4f758c47357d314d47ffa5bb4aefa8d65/robomimic/envs/env_robosuite.py#L20
        """
        input:
            reward_as_success --> for the easy of evalation.
            terminate_on_success --> done in the original robomimic impl. 
                ## Checkout cat robomimic_exps/core/tool_hang/ph/low_dim/bc.json | python3 -c "import sys, json; print(json.load(sys.stdin)['experiment']['rollout'])"
        """
        self.dataset_path, self.env_meta, self.env_splits, self.env_modal, ob_dims = self.initialize(task,dataset_type,hdf5_type)

        # Create an Env
        env_kwargs = deepcopy(self.env_meta['env_kwargs'])
        env_kwargs.update(dict(
            has_renderer=render,
            has_offscreen_renderer=(render_offscreen or use_image_obs),
            ignore_done=True,
            use_object_obs=True,
            use_camera_obs=use_image_obs,
            camera_depths=False,
        ))

        if env_kwargs['has_offscreen_renderer']:
            import egl_probe
            valid_gpu_devices = egl_probe.get_available_devices()
            if len(valid_gpu_devices) > 0:
                env_kwargs["render_gpu_device_id"] = valid_gpu_devices[0]

        self.env = robosuite.make(
            self.env_meta['env_name'],
            **env_kwargs
        )

        # Make sure joint position observations and eef vel observations are active
        for ob_name in self.env.observation_names:
            if ("joint_pos" in ob_name) or ("eef_vel" in ob_name):
                self.env.modify_observable(observable_name=ob_name, attribute="active", modifier=True)

        self.observation_space = gym.spaces.Box(
            np.full((ob_dims,), -float("inf"), dtype=np.float32),
            np.full((ob_dims,), float("inf"), dtype=np.float32),
            dtype=np.float32)
        self.action_space = gym.spaces.Box(
            self.env.action_spec[0],
            self.env.action_spec[1],
            dtype=np.float32)

        self.reward_as_success = reward_as_success
        self.terminate_on_success = terminate_on_success

    def _flatten(self,ob_dict):
        return np.concatenate([ob_dict[key] for key in self.env_modal['obs']['low_dim']],axis=-1)

    def reset(self):
        ob_dict = self.env.reset()
        ob_dict['object'] = ob_dict['object-state']

        return self._flatten(ob_dict)

    def step(self,action):
        ob_dict, r, done, info = self.env.step(action)
        ob_dict['object'] = ob_dict['object-state']

        assert done == False

        success = self.env._check_success()

        if self.reward_as_success:
            r = float(success)

        if self.terminate_on_success:
            done = done or success

        return self._flatten(ob_dict), r, done, info

    def render(self, mode="human", height=None, width=None, camera_name="agentview"):
        """
        Render from simulation to either an on-screen window or off-screen to RGB array.
        Args:
            mode (str): pass "human" for on-screen rendering or "rgb_array" for off-screen rendering
            height (int): height of image to render - only used if mode is "rgb_array"
            width (int): width of image to render - only used if mode is "rgb_array"
            camera_name (str): camera name to use for rendering
        """
        if mode == "human":
            cam_id = self.env.sim.model.camera_name2id(camera_name)
            self.env.viewer.set_camera(cam_id)
            return self.env.render()
        elif mode == "rgb_array":
            return self.env.sim.render(height=height, width=width, camera_name=camera_name)[::-1]
        else:
            raise NotImplementedError("mode={} is not implemented".format(mode))

    def get_dataset(self,split='all',sanity_check=False,ignore_done=False,truncate_if_done=False,shape_reward=True):
        """
        Parameters:
            ignore_done: the original done signal is ignored. (set to all zero)
                Following the white-paper, 'done' is set when
                    (1) 'sparse_reward' is 1, or 
                    (2) end of the trajectory (See Appendix C. Reward and Dones part)
                (Actually, there is only a single case where end-of-trajectory is activated, namely transport-mh-low-dim; in all other cases, sparse_reward == dones. Mostly, trajectory terminates due to time-out)
            truncate_if_done: truncate the trajectory when 'done' is set for the first time
                'done' can be set multiple times because 'done' is just used as a task-success signal.
            shape reward: if set, -1 everywhere except goal
        Return:
            a list of Trajectory
        """
        assert split in self.env_splits.keys(), f'{split} not in {self.env_splits.keys()}'

        with h5py.File(self.dataset_path, 'r') as f:
            trajs = []
            demo_keys = self.env_splits[split]

            demo_keys = sorted(demo_keys,key = lambda x: int(x.split('_')[1]))

            for demo_key in demo_keys:
                demo = f['data'][demo_key]

                obs = self._flatten(demo['obs'])
                actions = demo['actions'][...]
                rewards = demo['rewards'][...]
                dones = demo['dones'][...]
                next_obs = self._flatten(demo['next_obs'])

                if truncate_if_done and np.any(dones):
                    traj_len = np.argmax(dones) + 1
                else:
                    traj_len = len(obs)

                traj = Trajectory(
                    states = np.concatenate([obs,next_obs[-1:]],axis=0).astype(np.float32)[:traj_len+1],
                    actions = actions.astype(np.float32)[:traj_len],
                    rewards = rewards.astype(np.float32)[:traj_len] - 1. if shape_reward else rewards.astype(np.float32)[:traj_len],
                    dones = np.zeros_like(dones.astype(bool))[:traj_len] if ignore_done else dones[:traj_len],
                    frames = None,
                )
                trajs.append(traj)

                if sanity_check:
                    assert len(obs) == len(actions)
                    assert len(obs) == len(rewards)
                    assert len(obs) == len(dones)
                    assert len(obs) == len(next_obs)

                    assert np.allclose(obs[1:],next_obs[:-1])
                    #assert np.sum(dones) <= 1, f'done count is weird. ...{dones[-10:]} {np.sum(dones)}'
                    #assert np.sum(dones) == 0 or (np.sum(dones) == 1 and dones[-1] == True), f'done count is weird. {dones} {np.sum(dones)}'

                    if truncate_if_done:
                        assert np.sum(traj.rewards) <= 1

        return trajs

if __name__ == "__main__":
    """
    # Check whether mg low-dim-dense and low-dim-sparse is the same dataset except reward.

    env = RobomimicLowDimEnv('can','mg','low_dim_dense')
    env_sparse = RobomimicLowDimEnv('can','mg','low_dim_sparse')

    trajs = env.get_dataset(ignore_done=False)
    trajs_sparse = env_sparse.get_dataset(ignore_done=False)

    assert len(trajs) == len(trajs_sparse)

    for t, t_s in zip(trajs,trajs_sparse):
        assert np.allclose(t.states, t_s.states)
        assert np.allclose(t.actions, t_s.actions)
        assert np.allclose(t.dones, t_s.dones)
    """

    """
    # Check 'done' of the sparse-reward environments, which is default.
    # Following the white-paper, 'done' is whether 'sparse_reward' is 1 or end of the trajectory (See Appendix C. Reward and Dones part)
    # (Actually, there is only a single case where end-of-trajectory is activated, namely transport-mh-low-dim; in all other cases, reward == dones in full.)
    for env_id in [
        ('lift','mg','low_dim_sparse'),
        ('lift','ph','low_dim'),
        ('lift','mh','low_dim'),
        ('can','mg','low_dim_sparse'),
        ('can','ph','low_dim'),
        ('can','mh','low_dim'),
        ('square','ph','low_dim'),
        ('square','mh','low_dim'),
        ('transport','ph','low_dim'),
        ('transport','mh','low_dim'),
    ]:
        env = RobomimicLowDimEnv(*env_id)
        trajs = env.get_dataset(ignore_done=False)

        print('---------------------------')
        print(env_id)

        for t in trajs:
            assert np.allclose(t.rewards[:-1], t.dones[:-1]), f'{np.where(t.rewards != t.dones)} / {len(t.rewards)}'
    """

    """
    # Sanity Check (truncate if done works okay?)
    for env_id in [
        ('lift','mg','low_dim_sparse'),
        ('lift','ph','low_dim'),
        ('lift','mh','low_dim'),
        ('can','mg','low_dim_sparse'),
        ('can','ph','low_dim'),
        ('can','mh','low_dim'),
        ('square','ph','low_dim'),
        ('square','mh','low_dim'),
        ('transport','ph','low_dim'),
        ('transport','mh','low_dim'),
    ]:
        env = RobomimicLowDimEnv(*env_id)
        trajs = env.get_dataset(ignore_done=True,truncate_if_done=True,sanity_check=True)
        print([np.sum(traj.rewards) for traj in trajs])
    """