import mimicgen
import numpy as np
from robomimic.envs.env_robosuite import EnvRobosuite
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import collections
import os
import os
from tqdm import tqdm
import time
import pathlib
from termcolor import cprint
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from equi_diffpo.env_runner.base_image_runner import BaseImageRunner
from equi_diffpo.common.pytorch_util import dict_apply
from equi_diffpo.policy.base_image_policy import BaseImagePolicy
from equi_diffpo.model.common.rotation_transformer import RotationTransformer
from equi_diffpo.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from equi_diffpo.gym_util.sync_vector_env import SyncVectorEnv
from equi_diffpo.gym_util.async_vector_env import AsyncVectorEnv
import dill
import collections
import numpy as np
from copy import deepcopy
import h5py
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from zero.config.default import get_config
import json
import mimicgen
from robosuite.controllers import ALL_CONTROLLERS
from copy import deepcopy, copy
from codebase.z_utils.Rotation import *
from equi_diffpo.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
from equi_diffpo.gym_util.multistep_wrapper import MultiStepWrapper
import wandb.sdk.data_types.video as wv
np.set_printoptions(precision=4, suppress=True)


def dummy_data_generator():
    """
    Generate dummy data for testing.
    """
    # Generate random data
    data = np.random.rand(100, 7)  # 100 samples, 10 features
    return data


def create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=enable_render,
        use_image_obs=enable_render,
    )
    return env


def check_and_make(path):
    if not os.path.exists(path):
        os.makedirs(path)


def direct_test(traj: np.ndarray = None):
    options = {}
    eval_config = {
        'env_name': 'Stack_D0',
        'robots': 'Panda',
        'controller': 'JOINT_POSITION',
        'has_renderer': True,
        'record_video': False,
        'has_offscreen_renderer': False,
    }

    print(eval_config)

    env_name = eval_config['env_name']
    robot = eval_config['robots']
    has_renderer = eval_config['has_renderer']
    has_offscreen_renderer = eval_config['has_offscreen_renderer']

    options['env_name'] = env_name
    options["robots"] = robot
    options['controller_configs'] = load_controller_config(default_controller=eval_config['controller'])
    options['controller_configs']['control_delta'] = False
    print("options", options)

    env = suite.make(
        **options,
        has_renderer=has_renderer,
        has_offscreen_renderer=has_offscreen_renderer,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
    )

    obs = env.reset()
    env.viewer.set_camera(camera_id=0)
    low, high = env.action_spec
    # do visualization
    action_set = None

    for i in range(10000):
        PosEuler_offset_action2obs = np.array([0, 0, 0, 0, 0, 90])
        ee_pos = obs['robot0_eef_pos']
        ee_rot = obs['robot0_eef_quat']
        joint_pos_cos = obs['robot0_joint_pos_cos']
        joint_pos_sin = obs['robot0_joint_pos_sin']
        JP = np.arctan2(joint_pos_sin, joint_pos_cos)

        print("JP", JP)

        ee_open = [1]

        JP = np.concatenate((JP, ee_open), axis=-1)

        obs, reward, done, _ = env.step(JP)

        env.render()


def test1():
    env_meta = FileUtils.get_env_metadata_from_dataset("data/robomimic/datasets/stack_d1/stack_d1_abs_traj_eePose.hdf5")


direct_test()
