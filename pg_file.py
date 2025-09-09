import os
import re


def remove_date_from_folder_names(root_dir, dry_run=True):
    """
    仅移除指定根目录下第一级子文件夹的日期信息。

    Args:
        root_dir (str): 要处理的根目录。
        dry_run (bool): 如果为 True，则只打印将要执行的操作，而不实际重命名。
    """
    date_pattern = re.compile(r'^\d{2}\.\d{2}\.\d{2}_')
    renamed_count = 0

    print("-" * 40)
    print(f"扫描目录: '{root_dir}'")
    if dry_run:
        print(">>> 试运行模式：不会执行任何重命名操作。<<<")
    print("-" * 40)

    # 使用 os.listdir 获取所有一级子目录和文件
    try:
        entries = os.listdir(root_dir)
    except FileNotFoundError:
        print(f"错误：目录 '{root_dir}' 不存在。")
        return
    except PermissionError:
        print(f"错误：没有权限访问目录 '{root_dir}'。")
        return

    for entry in entries:
        full_path = os.path.join(root_dir, entry)

        # 仅处理文件夹
        if os.path.isdir(full_path):
            if date_pattern.match(entry):
                new_name = date_pattern.sub('', entry)
                new_path = os.path.join(root_dir, new_name)

                if entry == new_name:
                    continue  # 如果新旧名称相同，则跳过

                if dry_run:
                    print(f"[试运行] 将重命名: '{full_path}' -> '{new_path}'")
                else:
                    try:
                        os.rename(full_path, new_path)
                        print(f"已重命名: '{full_path}' -> '{new_path}'")
                        renamed_count += 1
                    except OSError as e:
                        print(f"重命名 '{full_path}' 时出错: {e}")
            else:
                if dry_run:
                    print(f"[试运行] 跳过: '{full_path}' (未找到日期模式)")

    print("-" * 40)
    if not dry_run:
        print(f"操作完成。已重命名 {renamed_count} 个文件夹。")
    else:
        print("试运行完成。未进行任何更改。")
    print("-" * 40)


if __name__ == "__main__":
    # 在这里指定你的文件夹路径
    root_folder = '/media/jian/data/ICRA_Archive/ICRA_DP_T_FILM'

    if not os.path.isdir(root_folder):
        print(f"错误：指定的目录 '{root_folder}' 不存在。")
    else:
        # 第1步：先以试运行模式运行
        remove_date_from_folder_names(root_folder, dry_run=True)

        # 第2步：询问用户是否继续
        confirmation = input("\n脚本将执行重命名操作。是否继续？ (是/否): ").lower()

        if confirmation in ['是', 'yes', 'y']:
            # 第3步：实际运行
            remove_date_from_folder_names(root_folder, dry_run=False)
        else:
            print("操作已由用户取消。")
