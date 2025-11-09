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
register_codecs()


class RobomimicReplayImageDataset(BaseImageDataset):
    def __init__(self,
                 shape_meta: dict,
                 dataset_path: list,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 n_obs_steps=None,
                 abs_action=False,
                 rotation_rep='rotation_6d',  # ignored when abs_action=False
                 use_legacy_normalizer=False,
                 use_cache=False,
                 seed=42,
                 val_ratio=0.0,
                 n_demo=100
                 ):

        self.n_demo = n_demo
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep=rotation_rep)
        dataset_path = list(dataset_path)

        dataset_name = []
        for dataset in dataset_path:
            dataset_name.append(dataset.split('/')[-2])

        zarr_path = 'data/robomimic/zarr/'
        dataset_name = '-'.join(dataset_name)

        replay_buffer = None
        if use_cache:
            cache_zarr_path = zarr_path + dataset_name + f'.{n_demo}.' + '.zarr.zip'
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        # store = zarr.DirectoryStore(cache_zarr_path)
                        replay_buffer = _convert_robomimic_to_replay_multitask(
                            store=zarr.MemoryStore(),
                            shape_meta=shape_meta,
                            dataset_paths=dataset_path,
                            abs_action=abs_action,
                            rotation_transformer=rotation_transformer,
                            n_demo=n_demo)
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _convert_robomimic_to_replay_multitask(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                dataset_paths=dataset_path,
                abs_action=abs_action,
                rotation_transformer=rotation_transformer,
                n_demo=n_demo)

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)

        # for key in rgb_keys:
        #     replay_buffer[key].compressor.numthreads=1

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask

        # debug
        # cprint('Debugging validation mask', 'yellow')
        # val_list = []
        # for i in range(len(val_mask)):
        #     print(f'Episode {i} - train: {train_mask[i]}, val: {val_mask[i]}')
        #     if val_mask[i]:
        #         val_list.append(i)
        # print(f'Validation episodes: {val_list}')
        # /debug

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        with open("equi_diffpo/constant/ABCDEFGHIJKL_normalizer.pkl", "rb") as f:
            this_normalizer = pickle.load(f)
            print("Using precomputed normalizer for actions.")

        normalizer['action'] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('JPOpen'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    # region getitem
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key][T_slice], -1, 1
                                        ).astype(np.float32) / 255.
            # T,C,H,W
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        return torch_data


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


def _convert_robomimic_to_replay_multitask(
    store,
    shape_meta,
    dataset_paths,
    abs_action,
    rotation_transformer,
    n_workers=None,
    max_inflight_tasks=None,
    n_demo=100
):
    """
    Converts multiple Robomimic datasets into a single Zarr Replay Buffer store.
    Handles missing keys by filling with zeros.

    Args:
        store: Zarr store object (e.g., zarr.DirectoryStore('my_replay.zarr')).
        shape_meta (dict): Metadata describing the shapes and types of observations and actions.
        dataset_paths (list[str]): A list of file paths to the HDF5 datasets to be combined.
        abs_action (bool): Flag for action space conversion.
        rotation_transformer: Transformer for rotation data.
        n_workers (int, optional): Number of parallel workers for data processing. Defaults to CPU count.
        max_inflight_tasks (int, optional): Max number of tasks to queue for workers. Defaults to n_workers * 5.
        n_demo_per_dataset (int, optional): Maximum number of demonstrations to load from each dataset. Defaults to 100.
    """

    # ----------------------------------------------
    # region 0.0 setup zarr store and meta groups
    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    # ---------------------------------------------
    # region 0.1 meta_data
    if not isinstance(dataset_paths, list):
        raise TypeError("dataset_paths must be a list of file paths.")

    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    rgb_keys = []
    lowdim_keys = []
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            rgb_keys.append(key)
        elif type == 'low_dim':
            lowdim_keys.append(key)

    # 有用信息
    # 1. lowdim_keys
    # 2. rgb_keys
    # endregion

    # ---------------------------------------------
    # 0.2 figure out the episodes data
    episode_ends = []
    total_steps = 0
    demo_counts = []
    for dataset_path in dataset_paths:
        with h5py.File(dataset_path, 'r') as file:
            demos = file['data']
            num_demos_in_file = min(n_demo, len(demos.keys()))
            demo_counts.append(num_demos_in_file)
            for i in range(num_demos_in_file):
                demo_key = f'demo_{i}'
                if demo_key not in demos:
                    print(f"Warning: {demo_key} not found in {dataset_path}. Skipping.")
                    continue
                episode_length = demos[demo_key]['actions'].shape[0]
                total_steps += episode_length
                episode_ends.append(total_steps)

    n_steps = total_steps
    if not episode_ends:
        print("No demonstrations found across all datasets. Exiting.")
        return None

    episode_starts = [0] + episode_ends[:-1]
    meta_group.array('episode_ends', episode_ends, dtype=np.int64, compressor=None, overwrite=True)
    print(f"Found a total of {n_steps} steps across {len(episode_ends)} episodes from {len(dataset_paths)} datasets.")

    # 有用信息
    # 1. episode_starts
    # 2. episode_ends
    # 3. n_steps
    # 4. demo_counts

    # endregion
    # ---------------------------------------------
    # region 1.1 load lowdim data
    for key in tqdm(lowdim_keys + ['action'], desc="Loading all lowdim data"):
        all_datasets_data = []
        data_key_source = 'obs/' + key if key != 'action' else 'actions'

        for dataset_path in dataset_paths:
            with h5py.File(dataset_path, 'r') as file:
                demos = file['data']
                this_dataset_data = []
                num_demos_in_file = min(n_demo, len(demos.keys()))
                for i in range(num_demos_in_file):
                    demo_key = f'demo_{i}'
                    if demo_key not in demos:
                        continue

                    demo = demos[demo_key]
                    episode_length = demo['actions'].shape[0]

                    # MODIFIED: Check if key exists, otherwise fill with zeros
                    data_exists = (key == 'action' and 'actions' in demo) or \
                                  (key != 'action' and 'obs' in demo and key in demo['obs'])

                    if data_exists:
                        data = demo[data_key_source][:].astype(np.float32)
                    else:
                        print(f"Key '{data_key_source}' not found in {dataset_path} demo_{i}. Filling with zeros.")
                        shape = shape_meta['action']['shape'] if key == 'action' else shape_meta['obs'][key]['shape']
                        data = np.zeros((episode_length,) + tuple(shape), dtype=np.float32)

                    this_dataset_data.append(data)

                if this_dataset_data:
                    all_datasets_data.append(np.concatenate(this_dataset_data, axis=0))

        if not all_datasets_data:
            print(f"Warning: No data found for key '{key}'. Skipping.")
            continue

        final_data = np.concatenate(all_datasets_data, axis=0)

        if key == 'action':
            final_data = _convert_actions(final_data, abs_action, rotation_transformer)

        data_group.array(name=key, data=final_data, compressor=None)

    def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
        try:
            zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
            # make sure we can successfully decode
            _ = zarr_arr[zarr_idx]
            return True
        except Exception as e:
            return False

        # except Exception as e:
        #     print(f"Error copying image at zarr_idx {zarr_idx} from hdf5_idx {hdf5_idx}: {e}")
        #     return False

    with tqdm(total=n_steps * len(rgb_keys), desc="Loading all image data", mininterval=1.0) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = set()
            for key in rgb_keys:
                shape = tuple(shape_meta['obs'][key]['shape'])
                c, h, w = shape
                img_arr = data_group.require_dataset(
                    name=key,
                    shape=(n_steps, h, w, c),
                    chunks=(1, h, w, c),
                    compressor=Jpeg2k(level=50),
                    dtype=np.uint8)

                global_episode_idx = 0
                for i, dataset_path in enumerate(dataset_paths):
                    with h5py.File(dataset_path, 'r') as file:
                        demos = file['data']
                        num_demos_in_file = demo_counts[i]

                        for episode_idx in range(num_demos_in_file):
                            demo = demos[f'demo_{episode_idx}']
                            episode_length = demo['actions'].shape[0]

                            hdf5_arr = demo['obs'][key]
                            for hdf5_idx in range(hdf5_arr.shape[0]):
                                if len(futures) >= max_inflight_tasks:
                                    completed, futures = concurrent.futures.wait(futures,
                                                                                 return_when=concurrent.futures.FIRST_COMPLETED)
                                    for f in completed:
                                        if not f.result():
                                            raise RuntimeError('Failed to encode image!')
                                    pbar.update(len(completed))

                                zarr_idx = episode_starts[global_episode_idx] + hdf5_idx
                                futures.add(executor.submit(img_copy, img_arr, zarr_idx, hdf5_arr, hdf5_idx))

                            global_episode_idx += 1

                        completed, futures = concurrent.futures.wait(futures)
                        for f in completed:
                            if not f.result():
                                raise RuntimeError('Failed to encode/copy a final image!')
                        pbar.update(len(completed))

    replay_buffer = ReplayBuffer(root)
    return replay_buffer


def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1 / max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )
