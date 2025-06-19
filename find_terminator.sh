#!/bin/bash

# 这是一个用于记录当前所有已打开的 Terminator 窗口位置的脚本。
# 它会列出每个 Terminator 窗口的 X 坐标、Y 坐标、宽度和高度。

echo "正在扫描已打开的窗口以查找 Terminator..."
echo "---------------------------------------------------"

# 使用 wmctrl -l -G 命令列出所有窗口的详细信息（包括几何形状）。
# -l: 列出所有窗口。
# -G: 提供窗口的几何信息 (X, Y, 宽度, 高度)。
# 通过 grep 过滤出包含 "Terminator" 的行。
# 注意：wmctrl 可能需要安装。在大多数基于 Debian/Ubuntu 的系统上，可以使用 'sudo apt install wmctrl' 安装。
# 在 Fedora/RHEL 系统上，可以使用 'sudo dnf install wmctrl' 安装。

wmctrl -l -G | while IFS= read -r line; do
    # 检查行中是否包含 "Terminator"
    if echo "$line" | grep -q "Firefox"; then
        # 提取窗口ID（第一列），X坐标（第三列），Y坐标（第四列），宽度（第五列），高度（第六列）。
        # awk '{print $1, $3, $4, $5, $6}' 用于按列提取数据。
        window_info=$(echo "$line" | awk '{print $1, $3, $4, $5, $6}')

        # 将提取的信息赋值给单独的变量以便阅读
        window_id=$(echo "$window_info" | awk '{print $1}')
        x_pos=$(echo "$window_info" | awk '{print $2}')
        y_pos=$(echo "$window_info" | awk '{print $3}')
        width=$(echo "$window_info" | awk '{print $4}')
        height=$(echo "$window_info" | awk '{print $5}')

        echo "发现 Terminator 窗口:"
        echo "  窗口 ID: $window_id"
        echo "  X 坐标:  $x_pos"
        echo "  Y 坐标:  $y_pos"
        echo "  宽度:    $width 像素"
        echo "  高度:    $height 像素"
        echo "---------------------------------------------------"
    fi
done

echo "扫描完成。"
