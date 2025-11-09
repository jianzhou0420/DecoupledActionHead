import matplotlib.cm as cm  # 用于颜色映射
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import matplotlib.pyplot as plt
import zarr
import pickle
import numpy as np
import h5py
import os
import torch
from natsort import natsorted
from jiandecouple.dataset.robomimic_replay_image_dataset import _convert_actions

np.set_printoptions(precision=4, suppress=True)  # 设置 NumPy 打印选项，保留三位小数并禁止科学计数法

tasks_meta = {
    "A": {"name": "stack_d1", "average_steps": 108, "color_rgb": [0.994, 0.406, 0.406], "color_hex": "#FD6767"},
    "B": {"name": "square_d2", "average_steps": 153, "color_rgb": [0.994, 0.655, 0.406], "color_hex": "#FDA767"},
    "C": {"name": "coffee_d2", "average_steps": 224, "color_rgb": [0.994, 0.905, 0.406], "color_hex": "#FDE667"},
    "D": {"name": "threading_d2", "average_steps": 227, "color_rgb": [0.834, 0.994, 0.406], "color_hex": "#D4FD67"},
    "E": {"name": "stack_three_d1", "average_steps": 255, "color_rgb": [0.584, 0.994, 0.406], "color_hex": "#94FD67"},
    "F": {"name": "hammer_cleanup_d1", "average_steps": 286, "color_rgb": [0.406, 0.994, 0.477], "color_hex": "#67FD79"},
    "G": {"name": "three_piece_assembly_d2", "average_steps": 335, "color_rgb": [0.406, 0.994, 0.727], "color_hex": "#67FDB9"},
    "H": {"name": "mug_cleanup_d1", "average_steps": 338, "color_rgb": [0.406, 0.994, 0.976], "color_hex": "#67FDF8"},
    "I": {"name": "nut_assembly_d0", "average_steps": 358, "color_rgb": [0.406, 0.762, 0.994], "color_hex": "#67C2FD"},
    "J": {"name": "kitchen_d1", "average_steps": 619, "color_rgb": [0.406, 0.513, 0.994], "color_hex": "#6782FD"},
    "K": {"name": "pick_place_d0", "average_steps": 677, "color_rgb": [0.549, 0.406, 0.994], "color_hex": "#8B67FD"},
    "L": {"name": "coffee_preparation_d1", "average_steps": 687, "color_rgb": [0.798, 0.406, 0.994], "color_hex": "#CB67FD"}
}


class RetrieveActions:

    @staticmethod
    def get_actions_from_alphabet(tasks='ABCDEFGHIJKL'):
        dataset_dir = "data/robomimic/datasets"

        alphabet = tasks.upper()  # 确保任务字母是大写
        alphabet = natsorted(alphabet)  # 确保字母按自然顺序排序

        ordered_task_names = [tasks_meta[letter]["name"] for letter in alphabet if letter in tasks_meta]
        actions_all = []  # 存储按tasks_meta顺序排列的actions
        for task_name in ordered_task_names:
            # 构造完整的数据集路径，注意文件名格式通常是 "task_name_abs_traj_eePose.hdf5"
            dataset_filename = f"{task_name}_abs_traj_eePose.hdf5"
            dataset_path = os.path.join(dataset_dir, task_name, dataset_filename)  # 假设数据集在 /dataset_dir/task_name/task_name_abs_traj_eePose.hdf5

            # 检查文件是否存在，以避免错误
            if not os.path.exists(dataset_path):
                print(f"警告: 数据集文件未找到，跳过: {dataset_path}")
                continue

            print(f"处理数据集: {dataset_path}")
            this_actions_all = []  # 存储当前数据集的所有 demonstration 的 actions
            try:
                with h5py.File(dataset_path, 'r') as f:
                    data = f['data']
                    demo_names = natsorted(list(data.keys()))  # 演示名称仍然按自然顺序排序
                    print(f"演示数量: {len(demo_names)}")
                    for demo_name in demo_names:
                        this_actions_all.append(data[demo_name]['actions'][:])
                actions_all.append(this_actions_all)
            except Exception as e:
                print(f"读取数据集 {dataset_path} 时发生错误: {e}")
                continue

        return actions_all

    @staticmethod
    def get_actions_all():
        dataset_dir = "data/robomimic/datasets"

        # 确保 actions_all 按照 tasks_meta 的键的字母顺序排列
        # 获取 tasks_meta 中任务名称的有序列表
        # sorted(tasks_meta.keys()) 默认会按字母顺序排序 'A', 'B', 'C'...
        ordered_task_names = [tasks_meta[key]["name"] for key in sorted(tasks_meta.keys())]

        actions_all = []  # 存储按tasks_meta顺序排列的actions

        for task_name in ordered_task_names:
            # 构造完整的数据集路径，注意文件名格式通常是 "task_name_abs_traj_eePose.hdf5"
            dataset_filename = f"{task_name}_abs_traj_eePose.hdf5"
            dataset_path = os.path.join(dataset_dir, task_name, dataset_filename)  # 假设数据集在 /dataset_dir/task_name/task_name_abs_traj_eePose.hdf5

            # 检查文件是否存在，以避免错误
            if not os.path.exists(dataset_path):
                print(f"警告: 数据集文件未找到，跳过: {dataset_path}")
                continue

            print(f"处理数据集: {dataset_path}")
            this_actions_all = []  # 存储当前数据集的所有 demonstration 的 actions
            try:
                with h5py.File(dataset_path, 'r') as f:
                    data = f['data']
                    demo_names = natsorted(list(data.keys()))  # 演示名称仍然按自然顺序排序
                    print(f"演示数量: {len(demo_names)}")
                    for demo_name in demo_names:
                        this_actions_all.append(data[demo_name]['actions'][:])
                actions_all.append(this_actions_all)
            except Exception as e:
                print(f"读取数据集 {dataset_path} 时发生错误: {e}")
                continue

        return actions_all


actions_all = RetrieveActions.get_actions_from_alphabet('A')

actions_A = actions_all[0]  # list of np.array which has shape (N, 7)

# 先计算位移

xyz_A = [action[:, :3] for action in actions_A]  # 提取每个动作的前3个元素作为xyz坐标


def calculate_displacement(actions):
    """
    计算每个动作序列的位移。
    位移是指每个动作与前一个动作之间的欧氏距离。
    """
    displacements = []
    for action in actions:
        displacement = np.linalg.norm(action[1:] - action[:-1], axis=1)
        displacements.append(displacement)
    return displacements


displacement = calculate_displacement(xyz_A)

for traj in actions_A:
    displacement.append(np.linalg.norm(traj[1:, :3] - traj[:-1, :3], axis=1))  # 计算每个动作的位移

for i, traj in enumerate(actions_A):

    plt.plot(displacement[i])
plt.show()
