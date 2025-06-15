import sys
import argparse
import torch

"""
original data is equidiff 
0. stack_d1_abs.hdf5

we convert it to three versions:
1. stack_d1_traj_eePose.hdf5
2. stack_d1_traj_JP.hdf5
3. stack_d1_traj_JP_eeloss.hdf5
"""
import h5py
import numpy as np
from copy import deepcopy
from codebase.z_utils.Rotation import PosQuat2HT, HT2PosAxis, PosEuler2HT, inv, axis2quat
from termcolor import cprint
import json
from tqdm import tqdm
import os
from natsort import natsorted


class HDF5Inspector:
    '''
    Print the structure of an HDF5 file in a tree format,
    but limit each group to displaying only 10 child keys.
    Usage:
    HDF5Inspector.inspect_hdf5('path/to/your/file.hdf5')
    '''
    MAX_KEYS = 15  # Maximum number of child items to display per group # TODO： make this a parameter

    @staticmethod
    def print_tree(name, obj, prefix='', is_last=True):
        connector = '└── ' if is_last else '├── '
        # Print either the root or a group/dataset name
        if isinstance(obj, h5py.Group):
            if name == '/':
                print(name)
            else:
                print(f"{prefix}{connector}{name.split('/')[-1]}")
            # Print any attributes on the group
            if obj.attrs:
                for i, attr in enumerate(obj.attrs):
                    is_last_attr = (i == len(obj.attrs) - 1 and len(obj) == 0)
                    attr_connector = '└── ' if is_last_attr else '├── '
                    attr_prefix = prefix + ('    ' if is_last else '│   ')
                    print(f"{attr_prefix}{attr_connector}@{attr}  ⟵ Attribute on group")
            # Get child items, but limit to MAX_KEYS
            items = list(obj.items())
            total_children = len(items)
            items_to_print = items[:HDF5Inspector.MAX_KEYS]
            for idx, (child_name, child_obj) in enumerate(items_to_print):
                last = (idx == len(items_to_print) - 1) and (total_children <= HDF5Inspector.MAX_KEYS)
                new_prefix = prefix + ('    ' if is_last else '│   ')
                HDF5Inspector.print_tree(child_name, child_obj, new_prefix, last)
            # If there are more than MAX_KEYS children, indicate truncation
            if total_children > HDF5Inspector.MAX_KEYS:
                trunc_prefix = prefix + ('    ' if is_last else '│   ')
                print(f"{trunc_prefix}└── ... and {total_children - HDF5Inspector.MAX_KEYS} more items")
        elif isinstance(obj, h5py.Dataset):
            shape = obj.shape
            dtype = obj.dtype
            print(f"{prefix}{connector}{name.split('/')[-1]}  ⟵ Dataset (shape: {shape}, dtype: {dtype})")
            # Print any attributes on the dataset
            if obj.attrs:
                for i, attr in enumerate(obj.attrs):
                    is_last_attr = (i == len(obj.attrs) - 1)
                    attr_connector = '└── ' if is_last_attr else '├── '
                    attr_prefix = prefix + ('    ' if is_last else '│   ')
                    print(f"{attr_prefix}{attr_connector}@{attr}  ⟵ Attribute on dataset")

    @staticmethod
    def inspect_hdf5(file_path):
        """Inspect and print the structure of the given HDF5 file."""
        try:
            with h5py.File(file_path, 'r') as f:
                HDF5Inspector.print_tree('/', f['/'], '', True)
        except Exception as e:
            print(f"Error opening file '{file_path}': {e}")


class DatasetConvertor:
    '''
    receive original data stack_d1_abs.hdf5
    '''
    PosEuler_offset_action2obs = np.array([0, 0, 0, 0, 0, 90])
    PosEuler_base_mimicgen = np.array([-0.561, 0., 0.925, 0., 0., 0.])
    PosEuler_offset_JP2eePose = np.array([0., 0., 0., 0., 0., - 180.])

    def traj_eePose(self, original_path: str):
        traj_eePose_path = original_path.replace("/datasets_abs/", "/datasets/").replace('.hdf5', '_traj_eePose.hdf5')
        os.makedirs(os.path.dirname(traj_eePose_path), exist_ok=True)

        self._copy2new_h5py_file(original_path, traj_eePose_path)
        cprint(f"Converting\n{original_path}\nto{traj_eePose_path}\n", 'blue')

        with h5py.File(traj_eePose_path, 'r+') as f:
            data = f['data']
            for i, key in enumerate(data.keys()):
                # 1. get a numpy copy of the dataset
                demo_data = data[key]
                PosAxisOpen_old = deepcopy(demo_data['actions'][...])  # PosAxis
                robot_ee_pos = deepcopy(demo_data['obs']['robot0_eef_pos'][...])
                robot_ee_quat = deepcopy(demo_data['obs']['robot0_eef_quat'][...])
                isOpen = PosAxisOpen_old[:, -1:]  # open action at t

                # 2. convert to eePose-PosAxis
                PosQuat_curr = np.concatenate([robot_ee_pos, robot_ee_quat], axis=-1)  # eePose at t
                PosAxis_curr = self._PosQuat2PosAxis(PosQuat_curr)

                # 3. convert to new PosAxisOpen
                PosAxis_new = np.concatenate((PosAxis_curr[1:, :], PosAxisOpen_old[-1:, :-1]), axis=0)
                PosAxisOpen_new = np.concatenate((PosAxis_new, isOpen), axis=-1)

                assert PosAxisOpen_new.shape == PosAxisOpen_old.shape, "PosAxisOpen_new shape is not equal to PosAxisOpen_old shape"

                demo_data['actions'][...] = PosAxisOpen_new
                assert np.all(demo_data['actions'][...] == PosAxisOpen_new), "demo_data['actions'] is not equal to PosAxisOpen_new"
        cprint(f"Convertion has been done\n You should find{traj_eePose_path}", 'green')

    def traj_JP(self, original_path: str):
        '''
        1. copy original_path to traj_JP_path
        2. change controller type to JOINT_POSITION
        3. convert actions from PosAxisOpen to JP
        '''

        traj_JP_path = original_path.replace("/datasets_abs/", "/datasets/").replace('.hdf5', '_traj_JP.hdf5')
        os.makedirs(os.path.dirname(traj_JP_path), exist_ok=True)
        self._copy2new_h5py_file(original_path, traj_JP_path)

        cprint(f"Converting\n{original_path}\nto{traj_JP_path}\n", 'blue')

        self._controller_type_to_JP(traj_JP_path)

        # change_controller_type(JP_h5py_file)
        with h5py.File(traj_JP_path, 'r+') as f:
            data = f['data']
            for i, key in enumerate(data.keys()):
                # 1. get a numpy copy of the dataset
                demo_data = data[key]
                action_data = deepcopy(demo_data['actions'][...])
                JP_all = deepcopy(demo_data['obs']['robot0_joint_pos'][...])

                open_action = action_data[:, -1:]  # open action at t
                JP_all_new = np.concatenate((JP_all[1:, :], JP_all[-1:, :]), axis=0)
                JPOpen_all_new = np.concatenate((JP_all_new, open_action), axis=-1)
                del demo_data['actions']
                demo_data['actions'] = JPOpen_all_new
        cprint(f"Convertion has been done\n You should find{traj_JP_path}", 'green')

    def traj_JP_eeloss(self, original_path: str):
        '''
        1. copy original_path to traj_JP_path
        2. change controller type to JOINT_POSITION
        3. convert actions from PosAxisOpen to JP
        4. add x0loss group with eePose
        '''
        traj_JP_eeloss_path = original_path.replace("/datasets_abs/", "/datasets/").replace('.hdf5', '_traj_JP_eeloss.hdf5')
        os.makedirs(os.path.dirname(traj_JP_eeloss_path), exist_ok=True)
        self._copy2new_h5py_file(original_path, traj_JP_eeloss_path)
        cprint(f"Converting\n{original_path}\nto{traj_JP_eeloss_path}\n", 'blue')
        self._controller_type_to_JP(traj_JP_eeloss_path)

        # change_controller_type(JP_h5py_file)
        with h5py.File(traj_JP_eeloss_path, 'r+') as f:
            data = f['data']
            for i, key in enumerate(data.keys()):
                # 1. get a numpy copy of the dataset
                demo_data = data[key]
                action_data = deepcopy(demo_data['actions'][...])
                JP_all = deepcopy(demo_data['obs']['robot0_joint_pos'][...])

                open_action = action_data[:, -1:]  # open action at t
                JP_all_new = np.concatenate((JP_all[1:, :], JP_all[-1:, :]), axis=0)
                JPOpen_all_new = np.concatenate((JP_all_new, open_action), axis=-1)
                del demo_data['actions']
                demo_data['actions'] = JPOpen_all_new

        # add x0lo
        with h5py.File(traj_JP_eeloss_path, 'r+') as f:
            data = f['data']
            for i, key in tqdm(enumerate(data.keys())):
                demo_i_group = data[key]
                ee_pos = deepcopy(demo_i_group['obs']['robot0_eef_pos'][...])
                ee_quat = deepcopy(demo_i_group['obs']['robot0_eef_quat'][...])
                open_ = deepcopy(demo_i_group['actions'][..., -1:])

                eePose = np.concatenate((ee_pos, ee_quat), axis=-1)
                eePose = np.concatenate((eePose[1:, :], eePose[-1:, :]), axis=0)
                eePose_with_open = np.concatenate((eePose, open_), axis=-1)

                x0loss_group = demo_i_group.require_group('x0loss')
                x0loss_group.create_dataset('eePose', data=eePose_with_open, dtype='f')

        cprint(f"Convertion has been done\n You should find{traj_JP_eeloss_path}", 'green')

    def pure_lowdim_eePose(self, original_path: str):
        pure_lowdim_path = original_path.replace("/datasets_abs/", "/datasets/").replace('.hdf5', '_pure_lowdim_traj_eePose.hdf5')
        os.makedirs(os.path.dirname(pure_lowdim_path), exist_ok=True)
        self._copy2new_h5py_file(original_path, pure_lowdim_path)
        cprint(f"Converting\n{original_path}\nto{pure_lowdim_path}\n", 'blue')

        self._controller_type_to_JP(pure_lowdim_path)
        # change_controller_type(JP_h5py_file)
        with h5py.File(pure_lowdim_path, 'r+') as f:
            data = f['data']
            for i, key in enumerate(data.keys()):
                # 1. get a numpy copy of the dataset
                demo_data = data[key]
                PosAxisOpen_old = deepcopy(demo_data['actions'][...])  # PosAxis
                robot_ee_pos = deepcopy(demo_data['obs']['robot0_eef_pos'][...])
                robot_ee_quat = deepcopy(demo_data['obs']['robot0_eef_quat'][...])
                isOpen = PosAxisOpen_old[:, -1:]  # open action at t

                # 2. convert to eePose-PosAxis
                PosQuat_curr = np.concatenate([robot_ee_pos, robot_ee_quat], axis=-1)  # eePose at t
                T_obs_curr = PosQuat2HT(PosQuat_curr)
                T_action_curr = T_obs_curr @ inv(PosEuler2HT(self.PosEuler_offset_action2obs[None, ...]))  # offset between action and obs
                PosAxis_curr = HT2PosAxis(T_action_curr)

                # 3. convert to new PosAxisOpen
                PosAxis_new = np.concatenate((PosAxis_curr[1:, :], PosAxisOpen_old[-1:, :-1]), axis=0)
                PosAxisOpen_new = np.concatenate((PosAxis_new, isOpen), axis=-1)

                assert PosAxisOpen_new.shape == PosAxisOpen_old.shape, "PosAxisOpen_new shape is not equal to PosAxisOpen_old shape"

                demo_data['actions'][...] = PosAxisOpen_new
                assert np.all(demo_data['actions'][...] == PosAxisOpen_new), "demo_data['actions'] is not equal to PosAxisOpen_new"

            for i, key in enumerate(data.keys()):
                # delete all obs
                # make states to be the only obs
                demo_data = data[key]
                states = deepcopy(demo_data['states'][...])
                obs_group = demo_data['obs']

                for name, obj in list(obs_group.items()):
                    if isinstance(obj, h5py.Dataset) and name != 'agentview_image':
                        del obs_group[name]

                obs_group.create_dataset('states', data=states, dtype='f8')

    def pure_lowdim_JP(self, original_path: str):
        pure_lowdim_path = original_path.replace("/datasets_abs/", "/datasets/").replace('.hdf5', '_pure_lowdim_traj_JP.hdf5')
        os.makedirs(os.path.dirname(pure_lowdim_path), exist_ok=True)
        self._copy2new_h5py_file(original_path, pure_lowdim_path)
        cprint(f"Converting\n{original_path}\nto{pure_lowdim_path}\n", 'blue')

        # change_controller_type(JP_h5py_file)
        with h5py.File(pure_lowdim_path, 'r+') as f:
            data = f['data']
            for i, key in enumerate(data.keys()):
                # 1. get a numpy copy of the dataset
                demo_data = data[key]
                action_data = deepcopy(demo_data['actions'][...])
                JP_all = deepcopy(demo_data['obs']['robot0_joint_pos'][...])

                open_action = action_data[:, -1:]  # open action at t
                JP_all_new = np.concatenate((JP_all[1:, :], JP_all[-1:, :]), axis=0)
                JPOpen_all_new = np.concatenate((JP_all_new, open_action), axis=-1)
                del demo_data['actions']
                demo_data['actions'] = JPOpen_all_new

            for i, key in enumerate(data.keys()):
                # delete all obs
                # make states to be the only obs
                demo_data = data[key]
                states = deepcopy(demo_data['states'][...])
                obs_group = demo_data['obs']

                for name, obj in list(obs_group.items()):
                    if isinstance(obj, h5py.Dataset) and name != 'agentview_image':
                        del obs_group[name]

                obs_group.create_dataset('states', data=states, dtype='f8')

    def JP2eePose(self, original_path: str):
        JP2eePose_path = original_path.replace("/datasets_abs/", "/datasets/").replace('.hdf5', '_JP2eePose.hdf5')
        os.makedirs(os.path.dirname(JP2eePose_path), exist_ok=True)
        self._copy2new_h5py_file(original_path, JP2eePose_path)
        cprint(f"Converting\n{original_path}\nto{JP2eePose_path}\n", 'blue')
        with h5py.File(JP2eePose_path, 'r+') as f:
            data = f['data']
            for i, key in enumerate(data.keys()):
                # 1. get a numpy copy of the dataset
                demo_data = data[key]
                JP_curr = deepcopy(demo_data['obs']['robot0_joint_pos'][...])
                pos_ee = deepcopy(demo_data['obs']['robot0_eef_pos'][...])
                quat_ee = deepcopy(demo_data['obs']['robot0_eef_quat'][...])
                open_action = deepcopy(demo_data['actions'][..., -1:])  # open action at t

                # 2. convert to eePose-PosAxis
                PosQuat_curr = np.concatenate([pos_ee, quat_ee], axis=-1)  # eePose at t
                PosAxis_curr = self._PosQuat2PosAxis(PosQuat_curr)

                PosAxis_new = np.concatenate((PosAxis_curr[1:, :], PosAxis_curr[-1:, :]), axis=0)
                PosAxisOpen_new = np.concatenate((PosAxis_new, open_action[:, -1:]), axis=-1)

                JP_new = np.concatenate((JP_curr[1:, :], JP_curr[-1:, :]), axis=0)
                JPOpen_new = np.concatenate((JP_new, open_action), axis=-1)

                demo_data['actions'][...] = PosAxisOpen_new
                for obs_key in list(demo_data['obs'].keys()):
                    del demo_data['obs'][obs_key]

                demo_data['obs'].create_dataset('JPOpen', data=JPOpen_new)
        HDF5Inspector.inspect_hdf5(JP2eePose_path)

    def JP2eePose_debug(self, original_path: str):
        JP2eePose_path = original_path.replace("/datasets_abs/", "/datasets/").replace('.hdf5', '_JP2eePose_degbug.hdf5')
        os.makedirs(os.path.dirname(JP2eePose_path), exist_ok=True)
        cprint(f"Converting\n{original_path}\nto{JP2eePose_path}\n", 'blue')
        self._copy2new_h5py_file(original_path, JP2eePose_path)
        with h5py.File(JP2eePose_path, 'r+') as f:
            data = f['data']
            for i, key in enumerate(data.keys()):
                # 1. get a numpy copy of the dataset
                demo_data = data[key]
                length = demo_data['actions'].shape[0]

                arr_eePose = np.arange(length).reshape(-1, 1)  # shape (108, 1)
                arr_eePose = np.tile(arr_eePose, (1, 7))          # shape (108, 7)，每列是行索引

                arr_JPOpen = np.arange(length).reshape(-1, 1)  # shape (108, 1)
                arr_JPOpen = np.tile(arr_JPOpen, (1, 8))          # shape (108, 8)，每列是行索引
                demo_data['actions'][...] = arr_eePose
                for obs_key in list(demo_data['obs'].keys()):
                    del demo_data['obs'][obs_key]
                demo_data['obs'].create_dataset('JPOpen', data=arr_JPOpen)

    def put_together_ABC(self, A_path: str, B_path: str, C_path: str):

        A_name = os.path.basename(A_path).split('_abs')[0]
        B_name = os.path.basename(B_path).split('_abs')[0]
        C_name = os.path.basename(C_path).split('_abs')[0]

        ABC_path = os.path.join("./first/data/robomimic/datasets/ABC", f"{A_name}_{B_name}_{C_name}_abs_JP2eePose.hdf5")
        # get all the demo keys from A, B, C
        with h5py.File(A_path, 'r') as f_A, h5py.File(B_path, 'r') as f_B, h5py.File(C_path, 'r') as f_C, h5py.File(ABC_path, 'w') as f_ABC:
            keys_A = natsorted(list(f_A['data'].keys()))
            keys_B = natsorted(list(f_B['data'].keys()))
            keys_C = natsorted(list(f_C['data'].keys()))
            f_ABC.create_group('data')
            demo_list = [keys_A, keys_B, keys_C]
            print(len(demo_list))

            available_choices = np.arange(len(keys_A) + len(keys_B) + len(keys_C))
            new_order = np.random.permutation(available_choices)
            print(f"New order of keys: {new_order}")
            print(f"Total number of demos: {len(new_order)}")
            x = new_order // 1000
            y = new_order % 1000

            for i in tqdm(range(len(new_order))):
                this_x = x[i]
                this_y = y[i]
                this_demo = demo_list[this_x][this_y]

                # copy the group from A, B, C to ABC
                if this_x == 0:
                    src_file = f_A
                elif this_x == 1:
                    src_file = f_B
                else:
                    src_file = f_C
                src_group = src_file['data'][this_demo]
                demo_name = f"demo_{i}"

                dst_group = f_ABC['data']
                src_file.copy(src_group, dst_group, name=demo_name)

        pass
    ################
    # private method
    ################

    def _PosQuat2PosAxis(self, PosQuat):
        """
        Convert PosQuat to PosAxis
        :param PosQuat: (N, 7) numpy array, where N is the number of samples
        :return: (N, 6) numpy array, where N is the number of samples
        """
        T_obs = PosQuat2HT(PosQuat)
        T_action = T_obs @ inv(PosEuler2HT(self.PosEuler_offset_action2obs[None, ...]))  # offset between action and obs
        PosAxis = HT2PosAxis(T_action)
        return PosAxis

    @staticmethod
    def _copy2new_h5py_file(src_path, dst_path):
        # 2. 在尝试写入前，检查目标文件是否存在
        if os.path.exists(dst_path):
            # 如果文件已存在，立即引发 FileExistsError 错误
            raise FileExistsError(f"目标文件已存在，无法覆盖: {dst_path}")

        # 如果文件不存在，则执行原始的复制逻辑
        with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'w') as dst:
            for name in src:
                src.copy(name, dst, name)
        cprint(f"成功将 {src_path} 复制到 {dst_path}", 'green')

    def _controller_type_to_JP(self, path: str):
        with h5py.File(path, 'r+') as f:
            env_meta = json.loads(f["data"].attrs["env_args"])
            # for i, key in enumerate(data.keys()):
            #     # 1. get a numpy copy of the dataset
            #     demo_data = data[key]
            #     print(data['demo_0'].keys())
            cprint(env_meta, 'blue')
            new_env_meata = {"env_name": "Stack_D1",
                             "env_version": "1.4.1",
                             "type": 1,
                             "env_kwargs": {"has_renderer": False,
                                            "has_offscreen_renderer": True,
                                            "ignore_done": True,
                                            "use_object_obs": True,
                                            "use_camera_obs": True,
                                            "control_freq": 20,
                                            "controller_configs": {"type": "JOINT_POSITION",
                                                                   "input_max": 1,
                                                                   "input_min": -1,
                                                                   "output_max": 0.05,
                                                                   "output_min": -0.05,
                                                                   "kp": 150,
                                                                   "damping_ratio": 1,
                                                                   "impedance_mode": "fixed",
                                                                   "kp_limits": [0, 300],
                                                                   "damping_ratio_limits": [0, 10],
                                                                   "qpos_limits": None,
                                                                   'control_delta': True,
                                                                   "interpolation": None,
                                                                   "ramp_ratio": 0.2},
                                            "robots": ["Panda"],
                                            "camera_depths": False,
                                            "camera_heights": 84,
                                            "camera_widths": 84,
                                            "render_gpu_device_id": 0,
                                            "reward_shaping": False,
                                            "camera_names": ["birdview", "agentview", "sideview", "robot0_eye_in_hand"]}}
            json_str = json.dumps(new_env_meata)
            f["data"].attrs["env_args"] = json_str
            cprint('Done !Changed env_args to', 'green')
        with h5py.File(path, 'r') as f:
            env_args = f["data"].attrs["env_args"]
            cprint(env_args, 'blue')


# class DatasetConvertor:
#     def traj_eePose(self, filepath):
#         print(f"🚀 调用方法 'traj_eePose' 处理: {filepath}")

#     def traj_JP(self, filepath):
#         print(f"🚀 调用方法 'traj_JP' 处理: {filepath}")

#     def traj_JP_eeloss(self, filepath):
#         print(f"🚀 调用方法 'traj_JP_eeloss' 处理: {filepath}")

#     def pure_lowdim_JP(self, filepath):
#         print(f"🚀 调用方法 'pure_lowdim_JP' 处理: {filepath}")

#     def JP2eePose_debug(self, filepath):
#         print(f"🚀 调用方法 'JP2eePose_debug' 处理: {filepath}")

#     def JP2eePose(self, filepath):
#         print(f"🚀 调用方法 'JP2eePose' 处理: {filepath}")

# ----------------------------------------------------

def main():
    # 1. 定义任务和转换方法
    tasks = {
        "A": "stack_d1",
        "B": "coffee_d2",
        "C": "three_piece_assembly_d2",
        "D": "stack_three_d1",
        "E": "square_d2",
        "F": "threading_d2",
        "G": "hammer_cleanup_d1",
        "H": "mug_cleanup_d1",
        "I": "kitchen_d1",
        "J": "nut_assembly_d0",
        "K": "pick_place_d0",
        "L": "coffee_preparation_d1"
    }
    valid_conversion_methods = [
        'traj_eePose', 'traj_JP', 'traj_JP_eeloss', 'pure_lowdim_JP',
        'JP2eePose_debug', 'JP2eePose'
    ]

    # 2. 设置 argparse 解析器
    parser = argparse.ArgumentParser(
        description="数据集转换工具：对一个或多个任务执行一个或多个转换操作，并生成总结报告。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-t', '--task', type=str, required=True, metavar='ALIASES', help="指定任务别名组合字符串 (例如: 'ADL')。")
    parser.add_argument('-c', '--convert_type', type=str, required=True, nargs='+', choices=valid_conversion_methods, metavar='METHOD', help="指定一个或多个转换方法名，用空格隔开。")
    args = parser.parse_args()

    # 3. 初始化计数器
    task_aliases = args.task.upper()
    methods_to_run = args.convert_type
    total_operations = len(task_aliases) * len(methods_to_run)
    success_count = 0
    skipped_count = 0
    error_count = 0
    unknown_task_count = 0

    print(f"✅ 计划执行: {len(methods_to_run)} 种转换操作")
    print(f"✅ 应用于: {len(task_aliases)} 个任务")
    print(f"✅ 总操作数: {total_operations}\n")

    convertor = DatasetConvertor()

    # 4. 执行嵌套循环并更新计数器
    for alias in task_aliases:
        if alias not in tasks:
            cprint(f"⚠️  警告: 未知的任务别名 '{alias}'，将跳过其所有转换。\n", 'red')
            unknown_task_count += 1
            error_count += len(methods_to_run)  # 该任务的所有操作都计为错误
            continue

        task_name = tasks[alias]
        # 准备演示用的源文件
        src_dir = f'data/robomimic/datasets_abs/{task_name}'
        os.makedirs(src_dir, exist_ok=True)
        file_path = os.path.join(src_dir, f'{task_name}_abs.hdf5')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"源文件不存在: {file_path}")

        print(f"--- [处理任务: {alias} ({task_name})] ---")

        for method_name in methods_to_run:
            try:
                method_to_call = getattr(convertor, method_name)
                cprint(f"  -> 正在执行: '{method_name}'...", 'cyan')
                method_to_call(file_path)
                cprint(f"  ✅ 成功完成: '{method_name}'", 'green')
                success_count += 1
            except FileExistsError:
                cprint(f"  -> 已存在，跳过: '{method_name}'", 'yellow')
                skipped_count += 1
            except Exception as e:
                cprint(f"  ❌ 执行 '{method_name}' 时发生未知错误: {e}", 'red')
                error_count += 1

        print(f"--- [任务 {alias} 处理完毕] ---\n")

    # 5. 打印最终的总结报告
    print("=" * 50)
    print("📊 处理完成，生成总结报告：")
    print("=" * 50)
    cprint(f"  - 成功转换: {success_count}", 'green')
    cprint(f"  - 已存在而跳过: {skipped_count}", 'yellow')
    cprint(f"  - 发生错误 (包括未知任务): {error_count}", 'red')
    print("----------------------------------------")
    print(f"  - 总计划操作数: {total_operations}")
    if unknown_task_count > 0:
        cprint(f"  - 其中包含 {unknown_task_count} 个未知任务别名。", 'magenta')
    print("=" * 50)


if __name__ == '__main__':
    main()
