#!/usr/bin/env python3
"""
NIfTI 文件原始图像和标签叠加的 3D 交互式可视化
用法: python3 show_nii_overlay_3d.py <image_file> <label_file>
例子：
python3 show_nii_overlay_3d.py \
    nnUNet_raw/Dataset004_Hippocampus/imagesTr/hippocampus_001_0000.nii.gz \
    nnUNet_raw/Dataset004_Hippocampus/labelsTr/hippocampus_001.nii.gz
"""

import argparse
import numpy as np
import nibabel as nib
import plotly.graph_objects as go
from skimage import measure


def create_mesh_data(data, threshold, downsample, color, name, opacity=0.8):
    """
    从3D数据创建mesh数据
    
    Args:
        data: 3D numpy数组
        threshold: 等值面阈值
        downsample: 降采样因子
        color: 网格颜色
        name: 网格名称
        opacity: 透明度
    
    Returns:
        plotly Mesh3d对象，如果失败返回None
    """
    # 降采样
    if downsample > 1:
        data = data[::downsample, ::downsample, ::downsample]
    
    print(f"  降采样后形状: {data.shape}")
    print(f"  数值范围: [{data.min():.2f}, {data.max():.2f}]")
    print(f"  使用阈值: {threshold}")
    
    # 检查数据是否有效
    if data.max() <= threshold:
        print(f"  警告: 最大值 {data.max():.2f} 小于等于阈值 {threshold:.2f}，跳过此数据")
        return None
    
    print(f"  生成 3D 网格...")
    
    try:
        # 生成等值面
        verts, faces, normals, values = measure.marching_cubes(
            data, level=threshold, spacing=(1.0, 1.0, 1.0)
        )
        
        print(f"  生成了 {len(verts)} 个顶点和 {len(faces)} 个面")
        
        # 创建 3D mesh
        x, y, z = verts.T
        i, j, k = faces.T
        
        mesh = go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            opacity=opacity,
            color=color,
            name=name,
            hoverinfo='text',
            text=f'{name}<br>顶点数: {len(verts)}'
        )
        
        return mesh
        
    except Exception as e:
        print(f"  错误: {e}")
        return None


def show_overlay_3d(image_path, label_path, 
                    image_threshold=None, label_threshold=None,
                    downsample=2,
                    image_opacity=0.3, label_opacity=0.8,
                    image_color='lightblue', label_color='red'):
    """
    在同一个交互式3D图中显示原始图像和标签
    
    Args:
        image_path: 原始图像文件路径
        label_path: 标签文件路径
        image_threshold: 原始图像阈值
        label_threshold: 标签阈值
        downsample: 降采样因子
        image_opacity: 原始图像透明度
        label_opacity: 标签透明度
        image_color: 原始图像颜色
        label_color: 标签颜色
    """
    meshes = []
    
    # 加载并处理原始图像
    print(f"\n加载原始图像: {image_path}")
    img = nib.load(image_path)
    image_data = img.get_fdata()
    print(f"原始图像形状: {image_data.shape}")
    
    # 自动选择原始图像阈值
    if image_threshold is None:
        image_threshold = np.percentile(image_data[image_data > 0], 75)
        print(f"自动选择原始图像阈值: {image_threshold:.2f}")
    
    image_mesh = create_mesh_data(
        image_data, 
        image_threshold, 
        downsample, 
        image_color, 
        '原始图像',
        image_opacity
    )
    
    if image_mesh:
        meshes.append(image_mesh)
    
    # 加载并处理标签
    print(f"\n加载标签: {label_path}")
    label_img = nib.load(label_path)
    label_data = label_img.get_fdata()
    print(f"标签形状: {label_data.shape}")
    
    # 检查形状是否匹配
    if image_data.shape != label_data.shape:
        print(f"警告: 图像形状 {image_data.shape} 与标签形状 {label_data.shape} 不匹配")
    
    # 处理多标签的情况
    unique_labels = np.unique(label_data)
    unique_labels = unique_labels[unique_labels > 0]  # 排除背景
    print(f"发现 {len(unique_labels)} 个标签: {unique_labels}")
    
    # 为每个标签创建mesh
    colors = ['red', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'purple', 'pink']
    
    for idx, label_val in enumerate(unique_labels):
        label_mask = (label_data == label_val).astype(float)
        
        if label_threshold is None:
            current_threshold = 0.5
        else:
            current_threshold = label_threshold
        
        current_color = colors[idx % len(colors)] if idx > 0 else label_color
        
        print(f"\n处理标签 {label_val}:")
        label_mesh = create_mesh_data(
            label_mask,
            current_threshold,
            downsample,
            current_color,
            f'标签 {int(label_val)}',
            label_opacity
        )
        
        if label_mesh:
            meshes.append(label_mesh)
    
    # 创建图形
    if not meshes:
        print("\n错误: 没有生成任何网格，请检查阈值设置")
        return
    
    print(f"\n创建交互式3D可视化（共 {len(meshes)} 个网格）...")
    fig = go.Figure(data=meshes)
    
    # 设置布局
    fig.update_layout(
        title='原始图像与标签的 3D 叠加可视化<br><sub>提示: 鼠标可旋转、缩放、平移</sub>',
        scene=dict(
            xaxis_title='X 轴',
            yaxis_title='Y 轴',
            zaxis_title='Z 轴',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1200,
        height=900,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    print("打开交互式窗口...")
    print("提示:")
    print("  - 鼠标左键拖动: 旋转")
    print("  - 鼠标滚轮: 缩放")
    print("  - 鼠标右键拖动: 平移")
    print("  - 点击图例可以隐藏/显示对应的网格")
    
    fig.show()


def main():
    parser = argparse.ArgumentParser(
        description='NIfTI 文件原始图像和标签的 3D 交互式叠加可视化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python3 show_nii_overlay_3d.py image.nii.gz label.nii.gz
  
  # 自定义阈值
  python3 show_nii_overlay_3d.py image.nii.gz label.nii.gz --image-threshold 100 --label-threshold 0.5
  
  # 调整透明度
  python3 show_nii_overlay_3d.py image.nii.gz label.nii.gz --image-opacity 0.5 --label-opacity 0.9
  
  # 更改颜色
  python3 show_nii_overlay_3d.py image.nii.gz label.nii.gz --image-color blue --label-color yellow
  
  # 减少降采样以提高质量（更慢）
  python3 show_nii_overlay_3d.py image.nii.gz label.nii.gz --downsample 1
        """)
    
    parser.add_argument('image_file', help='原始图像 NIfTI 文件路径')
    parser.add_argument('label_file', help='标签 NIfTI 文件路径')
    
    parser.add_argument('--image-threshold', '-it', type=float, default=None,
                        help='原始图像阈值（默认自动选择第75百分位）')
    parser.add_argument('--label-threshold', '-lt', type=float, default=None,
                        help='标签阈值（默认: 0.5）')
    
    parser.add_argument('--downsample', '-d', type=int, default=2,
                        help='降采样因子，越大速度越快但质量越低（默认: 2）')
    
    parser.add_argument('--image-opacity', '-io', type=float, default=0.3,
                        help='原始图像透明度 (0-1)（默认: 0.3）')
    parser.add_argument('--label-opacity', '-lo', type=float, default=0.8,
                        help='标签透明度 (0-1)（默认: 0.8）')
    
    parser.add_argument('--image-color', '-ic', type=str, default='lightblue',
                        help='原始图像颜色（默认: lightblue）')
    parser.add_argument('--label-color', '-lc', type=str, default='red',
                        help='标签颜色（默认: red）')
    
    args = parser.parse_args()
    
    try:
        show_overlay_3d(
            args.image_file, 
            args.label_file,
            args.image_threshold,
            args.label_threshold,
            args.downsample,
            args.image_opacity,
            args.label_opacity,
            args.image_color,
            args.label_color
        )
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()