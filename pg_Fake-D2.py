import open3d as o3d
import os


def save_camera_pose(vis):
    """回调函数：保存相机参数"""
    print("按下了 's' 键，正在保存相机位姿...")
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters("camera_pose.json", param)
    print("相机位姿已保存到 'camera_pose.json'")
    # 返回 False 表示事件已处理，不再传递
    return False


# --- 主程序 ---
# 创建一个示例几何体
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
if not mesh.has_vertex_normals():
    mesh.compute_vertex_normals()

# 初始化可视化窗口
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(window_name="按 's' 保存相机位姿")
vis.add_geometry(mesh)

# 注册键盘回调函数
# register_key_callback(key, callback_function)
# key: 使用大写字母的ASCII码
vis.register_key_callback(ord("S"), save_camera_pose)

print("\n操作指南:")
print("1. 在窗口中用鼠标调整到你喜欢的视角。")
print("2. 按下 's' 键来保存当前的相机位姿。")
print("3. 按 'q' 键关闭窗口。")

vis.run()
vis.destroy_window()

# 验证文件是否已创建
if os.path.exists("camera_pose.json"):
    print("\n验证：成功找到保存的文件 camera_pose.json。")
    # 你可以查看文件内容
    with open("camera_pose.json", 'r') as f:
        print("文件内容预览：\n" + f.read(200) + "...")
