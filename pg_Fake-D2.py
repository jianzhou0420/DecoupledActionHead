import matplotlib.cm as cm  # 用于颜色映射
import open3d as o3d
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
import matplotlib.pyplot as plt
import zarr
from equi_diffpo.model.common.rotation_transformer import RotationTransformer
import pickle
import numpy as np
import h5py

import os
import torch
from natsort import natsorted
from codebase.z_utils.Rotation_torch import matrix_to_rotation_6d, euler2mat
from equi_diffpo.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from equi_diffpo.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)
from equi_diffpo.dataset.robomimic_replay_image_dataset import _convert_actions

tasks_meta = {
    "A": {
        "name": "stack_d1",
        "average_steps": 108,
    },
    "B": {
        "name": "square_d2",
        "average_steps": 153,
    },
    "C": {
        "name": "coffee_d2",
        "average_steps": 224,
    },
    "D": {
        "name": "threading_d2",
        "average_steps": 227,
    },
    "E": {
        "name": "stack_three_d1",
        "average_steps": 255,
    },
    "F": {
        "name": "hammer_cleanup_d1",
        "average_steps": 286,
    },
    "G": {
        "name": "three_piece_assembly_d2",
        "average_steps": 335,
    },
    "H": {
        "name": "mug_cleanup_d1",
        "average_steps": 338,
    },
    "I": {
        "name": "nut_assembly_d0",
        "average_steps": 358,
    },
    "J": {
        "name": "kitchen_d1",
        "average_steps": 619,
    },
    "K": {
        "name": "pick_place_d0",
        "average_steps": 677,
    },
    "L": {
        "name": "coffee_preparation_d1",
        "average_steps": 687,
    },
}


def get_actions_all():
    dataset_dir = "/media/jian/ssd4t/DP/first/data/robomimic/datasets"

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


def draw_with_default_color(actions_all):
    """
    绘制点云，使用默认颜色。
    """

    actions_all_flattened = np.concatenate([np.concatenate(actions) for actions in actions_all], axis=0)
    print(f"扁平化后的 action 形状: {actions_all_flattened.shape}")

    # 确保 actions_all_flattened 至少有 3 列，分别代表 x, y, z
    if actions_all_flattened.shape[1] < 3:
        print("Action 的维度不足以绘制 3D 点云。")
    else:
        # 提取前三个维度（例如，x, y, z 坐标）
        # Open3D 期望一个 (N, 3) 的 numpy 数组作为点
        points = actions_all_flattened[:, :3]

        # 创建一个 Open3D PointCloud 对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # 计算边界框
        bbox = pcd.get_axis_aligned_bounding_box()
        bbox.color = (0, 0, 0)
        print("bbox is:", bbox)

        # 可选：为点云添加颜色
        # 你也可以根据其他维度或属性来着色
        # pcd.colors = o3d.utility.Vector3dVector(np.array([[1, 0, 0] for _ in range(len(points))]))

        # 可视化点云和边界框
        print("使用 Open3D 可视化包含边界框的 action 点云...")
        o3d.visualization.draw_geometries([pcd, bbox],
                                          window_name="Action 点云和边界框",
                                          width=800, height=600,
                                          left=50, top=50,
                                          mesh_show_back_face=False)

        print("Open3D 可视化已关闭。")

        # 你也可以将点云和边界框保存到文件
        # o3d.io.write_point_cloud("action_point_cloud_with_bbox.ply", pcd)
        # o3d.io.write_triangle_mesh("action_bounding_box.ply", bbox.get_mesh())
        # print("点云和边界框已保存到 .ply 文件。")
    pass

    # 到这里，获得all_action，包含所有数据集的动作数据


def draw_with_respect_to_task(actions_all):

    # 为 actions_all 中的每个子列表生成一组不同的颜色
    num_groups = len(actions_all)
    # 使用 viridis 颜色映射，生成 num_groups 种不同的颜色
    colors = [cm.viridis(i / float(num_groups)) for i in range(num_groups)]

    geometries_to_draw = []

    print("准备点云进行可视化...")
    for i, group_of_actions in enumerate(actions_all):
        # 连接此组中的所有 actions (等同于 this_actions_all)
        if not group_of_actions:
            print(f"跳过索引 {i} 处的空组")
            continue

        concatenated_group_actions = np.concatenate(group_of_actions, axis=0)

        # 确保至少有 3 个维度用于绘图
        if concatenated_group_actions.shape[1] < 3:
            print(f"组 {i} 的 actions 没有足够的维度用于 3D 绘图。跳过。")
            continue

        points = concatenated_group_actions[:, :3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # 为此组分配唯一的颜色
        # Open3D 期望颜色是 (R, G, B) 元组，每个分量在 0-1 之间
        # Matplotlib 颜色映射返回 (R, G, B, A)，所以我们取前 3 个
        group_color = colors[i][:3]
        pcd.colors = o3d.utility.Vector3dVector(np.tile(group_color, (len(points), 1)))

        geometries_to_draw.append(pcd)

    # 在所有组合点周围添加一个边界框
    if geometries_to_draw:
        # 首先，组合所有点以计算全局边界框
        all_points_combined = np.concatenate([np.asarray(p.points) for p in geometries_to_draw], axis=0)

        temp_pcd_for_bbox = o3d.geometry.PointCloud()
        temp_pcd_for_bbox.points = o3d.utility.Vector3dVector(all_points_combined)

        bbox = temp_pcd_for_bbox.get_axis_aligned_bounding_box()
        bbox.color = (1, 0, 0)  # 全局边界框为红色
        geometries_to_draw.append(bbox)
        print("bbox is:", bbox)

        print(f"要绘制的点云组总数: {len(actions_all)}")
        print("使用 Open3D 可视化不同颜色的点云（按组着色）和全局边界框...")
        o3d.visualization.draw_geometries(geometries_to_draw,
                                          window_name="Action 点云（按组着色）和全局边界框",
                                          width=1024, height=768,
                                          left=50, top=50,
                                          mesh_show_back_face=False)
        print("Open3D 可视化已关闭。")
    else:
        print("没有点云可绘制。")


def draw_two_action_groups(group1_actions_all, group1_color, group2_actions_all, group2_color):
    """
    绘制两个点云组，每个组使用指定颜色，并在同一窗口中显示。
    :param group1_actions_all: 第一个 action 组 (actions_all 格式)。
    :param group1_color: 第一个 action 组的颜色 (R, G, B) 元组，值在 0-1 之间。
    :param group2_actions_all: 第二个 action 组 (actions_all 格式)。
    :param group2_color: 第二个 action 组的颜色 (R, G, B) 元组，值在 0-1 之间。
    """
    geometries_to_draw = []
    all_points_for_bbox = []

    # Helper function to process each group
    def process_group(actions_all_list, color):
        if not actions_all_list:
            print("Warning: An action group is empty and will not be drawn.")
            return None, None

        # Flatten the current group's actions
        flattened_actions = np.concatenate([np.concatenate(actions) for actions in actions_all_list], axis=0)
        print(f"Flattened shape for a group: {flattened_actions.shape}")

        if flattened_actions.shape[1] < 3:
            print("Action dimensions are less than 3, cannot plot in 3D. Skipping group.")
            return None, None

        points = flattened_actions[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(points), 1)))

        return pcd, points

    # Process Group 1
    pcd1, points1 = process_group(group1_actions_all, group1_color)
    if pcd1:
        geometries_to_draw.append(pcd1)
        all_points_for_bbox.append(points1)

    # Process Group 2
    pcd2, points2 = process_group(group2_actions_all, group2_color)
    if pcd2:
        geometries_to_draw.append(pcd2)
        all_points_for_bbox.append(points2)

    # Create a combined bounding box if there are any points
    if all_points_for_bbox:
        combined_points = np.concatenate(all_points_for_bbox, axis=0)
        temp_pcd_for_bbox = o3d.geometry.PointCloud()
        temp_pcd_for_bbox.points = o3d.utility.Vector3dVector(combined_points)

        bbox = temp_pcd_for_bbox.get_axis_aligned_bounding_box()
        bbox.color = (0, 0, 0)  # Black color for the combined bounding box
        print("Combined Bounding Box:", bbox)
        geometries_to_draw.append(bbox)
    else:
        print("No valid points found in either group to draw a bounding box.")

    if geometries_to_draw:
        print("Visualizing two action groups with specified colors and a combined bounding box...")
        o3d.visualization.draw_geometries(geometries_to_draw,
                                          window_name="Two Action Groups with Specified Colors",
                                          width=1024, height=768,
                                          left=50, top=50,
                                          mesh_show_back_face=False)
        print("Open3D visualization closed.")
    else:
        print("No geometries to draw.")


actions_all = get_actions_all()

# actions_all = actions_all[:4]  # 仅绘制前三个任务的动作数据
# draw_with_respect_to_task(actions_all[0:1])
draw_with_default_color(actions_all[0:1])  # 绘制第一个任务的动作数据，使用默认颜色
group_1 = [actions_all[0], actions_all[2], actions_all[6]]  # 合并第一个、第三个和第四个任务的动作数据
group_2 = actions_all[3:4]  # 仅使用第四个任务的动作数据
# draw_two_action_groups(group_1, (0, 0, 1), actions_all[4:5], (0, 1, 0))  # 绘制第一个和第二个任务的动作数据，分别使用红色和绿色
