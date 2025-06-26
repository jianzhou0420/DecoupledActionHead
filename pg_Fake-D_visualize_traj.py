import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def generate_mock_trajectories(num_trajectories=1000, num_points_per_trajectory=50):
    """
    生成模拟的轨迹数据。
    返回一个列表，其中每个元素是一个 (N, 3) 的 NumPy 数组，代表一条轨迹。
    """
    trajectories = []
    start_points = np.random.rand(num_trajectories, 3) * 100  # 随机生成起点
    for i in range(num_trajectories):
        # 基于随机游走创建轨迹
        steps = np.random.randn(num_points_per_trajectory, 3).cumsum(axis=0)
        trajectory = start_points[i] + steps
        trajectories.append(trajectory)
    return trajectories


def visualize_trajectories(trajectories):
    """
    将一系列轨迹高效地可视化为一个 LineSet。
    """
    if not trajectories:
        print("没有轨迹可供可视化。")
        return

    # 1. 准备颜色
    # 使用 'viridis', 'jet', 'hsv', 'rainbow' 等 colormap
    num_trajectories = len(trajectories)
    cmap = plt.get_cmap("viridis")
    colors_for_trajectories = [cmap(i / num_trajectories) for i in range(num_trajectories)]

    # 2. 初始化用于构建 LineSet 的列表
    all_points = []
    all_lines = []
    all_colors = []

    point_offset = 0  # 用于追踪全局顶点列表中的索引

    print(f"正在处理 {num_trajectories} 条轨迹...")

    # 3. 遍历所有轨迹，填充列表
    for i, trajectory_points in enumerate(trajectories):
        if len(trajectory_points) < 2:
            continue  # 一条轨迹至少需要两个点才能形成线

        # 获取当前轨迹的颜色 (Matplotlib返回RGBA，我们只需要RGB)
        current_color = colors_for_trajectories[i][:3]

        # 将当前轨迹的点添加到全局点列表
        all_points.extend(trajectory_points)

        # 为当前轨迹创建线段
        num_points_in_traj = len(trajectory_points)
        for j in range(num_points_in_traj - 1):
            # 线段连接的是全局索引
            start_point_index = point_offset + j
            end_point_index = point_offset + j + 1
            all_lines.append([start_point_index, end_point_index])
            all_colors.append(current_color)

        # 更新下一个轨迹的起始点索引
        point_offset += num_points_in_traj

    # 4. 创建 LineSet 对象
    if not all_points:
        print("没有有效的点来创建几何体。")
        return

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(all_points))
    line_set.lines = o3d.utility.Vector2iVector(np.array(all_lines))
    # 为每条线段设置颜色
    line_set.colors = o3d.utility.Vector3dVector(np.array(all_colors))

    # 5. 可视化
    print("处理完成，正在启动可视化窗口...")
    o3d.visualization.draw_geometries(
        [line_set],
        window_name=f"可视化 {num_trajectories} 条轨迹",
        width=1280,
        height=720
    )


# --- 主程序 ---
if __name__ == "__main__":
    # 生成模拟数据
    my_trajectories = generate_mock_trajectories(num_trajectories=1000, num_points_per_trajectory=50)

    # 可视化轨迹
    visualize_trajectories(my_trajectories)
