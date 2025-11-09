import multiprocessing
import concurrent.futures
import h5py
import zarr
import numpy as np
from tqdm import tqdm
from jiandecouple.common.replay_buffer import ReplayBuffer
# Use the user-provided import and register codecs
# register_codecs() # Call this if your custom codec library requires it.
from jiandecouple.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
# Assume ReplayBuffer and _convert_actions are defined elsewhere
# For demonstration purposes, let's create dummy versions.
register_codecs()  # Register custom codecs if necessary


def _convert_actions(raw_actions, abs_action, rotation_transformer):
    """Dummy function for action conversion."""
    # In a real scenario, this would apply transformations.
    # print(f"Converting actions... (abs_action: {abs_action})")
    return raw_actions


def _convert_robomimic_to_replay_multitask(
    store,
    shape_meta,
    dataset_paths,
    abs_action,
    rotation_transformer,
    n_workers=None,
    max_inflight_tasks=None,
    n_demo_per_dataset=100
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
            num_demos_in_file = min(n_demo_per_dataset, len(demos.keys()))
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
                num_demos_in_file = min(n_demo_per_dataset, len(demos.keys()))
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


# ================== Example Usage from User ==================
if __name__ == '__main__':
    # NOTE: This section uses local file paths provided by the user and will only
    # run in an environment with access to these files.

    # Register custom codecs if necessary
    # from equi_diffpo.codecs.imagecodecs_numcodecs import register_codecs
    # register_codecs()

    # 1. Define dataset paths
    dataset_paths = [
        # Example paths, replace with your actual file paths
        '/media/jian/ssd4t/DP/first/data/robomimic/datasets/coffee_d2/coffee_d2_abs_traj_eePose.hdf5',
        '/media/jian/ssd4t/DP/first/data/robomimic/datasets/stack_d1/stack_d1_abs_traj_eePose.hdf5'
    ]

    if not dataset_paths:
        print("Please provide valid dataset paths in the `if __name__ == '__main__':` block.")
    else:
        # 2. Define shape metadata
        shape_meta = {
            'obs': {
                'agentview_image': {'shape': [3, 84, 84], 'type': 'rgb'},
                'robot0_eye_in_hand_image': {'shape': [3, 84, 84], 'type': 'rgb'},
                'robot0_eef_pos': {'shape': [3]},  # type default: low_dim
                'robot0_eef_quat': {'shape': [4]},
                'robot0_gripper_qpos': {'shape': [2]}
            },
            'action': {'shape': [10]}
        }

        # 3. Define Zarr store
        zarr_store_path = 'test/my_combined_replay.zarr'
        store = zarr.DirectoryStore(zarr_store_path)

        # 4. Run the conversion function
        print("\nStarting conversion of multiple datasets...")
        replay_buffer = _convert_robomimic_to_replay_multitask(
            store=store,
            shape_meta=shape_meta,
            dataset_paths=dataset_paths,
            abs_action=False,
            rotation_transformer=None,
            n_demo_per_dataset=100,  # Process up to 10 demos from each file
            n_workers=18,
        )
        if replay_buffer:
            print(f"\nConversion complete. Combined data saved to: {zarr_store_path}")
