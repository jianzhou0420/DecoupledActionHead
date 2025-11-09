import pickle
from termcolor import cprint
from typing import Dict, List
import torch
import numpy as np
import h5py
from tqdm import tqdm
import zarr
import os
import shutil
import copy
import json
import hashlib
from filelock import FileLock
from threadpoolctl import threadpool_limits
import concurrent.futures
import multiprocessing
from omegaconf import OmegaConf
from jiandecouple.common.pytorch_util import dict_apply
from jiandecouple.dataset.base_dataset import BaseImageDataset, LinearNormalizer
from jiandecouple.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from jiandecouple.model.common.rotation_transformer import RotationTransformer
from jiandecouple.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from jiandecouple.common.replay_buffer import ReplayBuffer
from jiandecouple.common.sampler import SequenceSampler, get_val_mask
from jiandecouple.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)
from jiandecouple.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
register_codecs()


def _convert_robomimic_to_replay(store, shape_meta, dataset_path, abs_action, rotation_transformer,
                                 n_workers=None, max_inflight_tasks=None, n_demo=100):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        shape = attr['shape']
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            rgb_keys.append(key)
        elif type == 'low_dim':
            lowdim_keys.append(key)

    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    with h5py.File(dataset_path) as file:
        # count total steps
        demos = file['data']
        episode_ends = list()
        prev_end = 0
        for i in range(n_demo):
            demo = demos[f'demo_{i}']
            episode_length = demo['actions'].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
        n_steps = episode_ends[-1]
        episode_starts = [0] + episode_ends[:-1]
        _ = meta_group.array('episode_ends', episode_ends,
                             dtype=np.int64, compressor=None, overwrite=True)

        # save lowdim data
        for key in tqdm(lowdim_keys + ['action'], desc="Loading lowdim data"):
            data_key = 'obs/' + key
            if key == 'action':
                data_key = 'actions'
            this_data = list()
            for i in range(n_demo):
                demo = demos[f'demo_{i}']
                this_data.append(demo[data_key][:].astype(np.float32))
            this_data = np.concatenate(this_data, axis=0)
            if key == 'action':
                this_data = _convert_actions(
                    raw_actions=this_data,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer
                )
                assert this_data.shape == (n_steps,) + tuple(shape_meta['action']['shape'])
            else:
                assert this_data.shape == (n_steps,) + tuple(shape_meta['obs'][key]['shape'])
            _ = data_group.array(
                name=key,
                data=this_data,
                shape=this_data.shape,
                chunks=this_data.shape,
                compressor=None,
                dtype=this_data.dtype
            )

        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                # make sure we can successfully decode
                _ = zarr_arr[zarr_idx]
                return True
            except Exception as e:
                return False

        with tqdm(total=n_steps * len(rgb_keys), desc="Loading image data", mininterval=1.0) as pbar:
            # one chunk per thread, therefore no synchronization needed
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = set()
                for key in rgb_keys:
                    data_key = 'obs/' + key
                    shape = tuple(shape_meta['obs'][key]['shape'])
                    c, h, w = shape
                    this_compressor = Jpeg2k(level=50)
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps, h, w, c),
                        chunks=(1, h, w, c),
                        compressor=this_compressor,
                        dtype=np.uint8
                    )
                    for episode_idx in range(n_demo):
                        demo = demos[f'demo_{episode_idx}']
                        hdf5_arr = demo['obs'][key]
                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                # limit number of inflight tasks
                                completed, futures = concurrent.futures.wait(futures,
                                                                             return_when=concurrent.futures.FIRST_COMPLETED)
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError('Failed to encode image!')
                                pbar.update(len(completed))

                            zarr_idx = episode_starts[episode_idx] + hdf5_idx
                            futures.add(
                                executor.submit(img_copy,
                                                img_arr, zarr_idx, hdf5_arr, hdf5_idx))
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError('Failed to encode image!')
                pbar.update(len(completed))

    replay_buffer = ReplayBuffer(root)
    return replay_buffer


def _convert_actions(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1, 2, 7)
            is_dual_arm = True

        pos = raw_actions[..., :3]
        rot = raw_actions[..., 3:6]
        gripper = raw_actions[..., 6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([
            pos, rot, gripper
        ], axis=-1).astype(np.float32)

        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1, 20)
        actions = raw_actions
    return actions


if __name__ == '__main__':
    from jiandecouple.z_utils.h5_utils import HDF5Inspector

    shape_meta = {
        'obs': {
            'agentview_image': {'shape': [3, 84, 84], 'type': 'rgb'},
            'robot0_eye_in_hand_image': {'shape': [3, 84, 84], 'type': 'rgb'},
            'robot0_eef_pos': {'shape': [3]},
            'robot0_eef_quat': {'shape': [4]},
            'robot0_gripper_qpos': {'shape': [2]}
        },
        'action': {'shape': [10]}
    }
    dataset_path1 = "/media/jian/ssd4t/DP/first/data/robomimic/datasets/stack_d1/stack_d1_abs_traj_eePose.hdf5"
    dataset_path2 = "/media/jian/ssd4t/DP/first/data/robomimic/datasets/coffee_d2/coffee_d2_abs_traj_eePose.hdf5"

    # HDF5Inspector.inspect_hdf5(dataset_path1)

    rotation_rep = 'rotation_6d'
    rotation_transformer = RotationTransformer(
        from_rep='axis_angle', to_rep=rotation_rep)

    replay_buffer1 = _convert_robomimic_to_replay(
        store=zarr.NestedDirectoryStore('test/test.zarr'),
        shape_meta=shape_meta,
        dataset_path=dataset_path1,
        abs_action=True,
        rotation_transformer=rotation_transformer,
        n_demo=10
    )
    replay_buffer2 = _convert_robomimic_to_replay(
        store=zarr.NestedDirectoryStore('test/test2.zarr'),
        shape_meta=shape_meta,
        dataset_path=dataset_path2,
        abs_action=True,
        rotation_transformer=rotation_transformer,
        n_demo=10
    )
