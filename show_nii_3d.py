#!/usr/bin/env python3
"""
NIfTI 文件 3D 可视化脚本
用法: python3 show_nii_3d.py <nii_file_path>
例子：
# 方案一：matplotlib 3D 表面渲染
python3 show_nii_3d.py nnUNet_raw/Dataset004_Hippocampus/labelsTr/hippocampus_001.nii.gz --mode surface

# 方案一：散点图模式（更快）
python3 show_nii_3d.py nnUNet_raw/Dataset004_Hippocampus/labelsTr/hippocampus_001.nii.gz --mode scatter

"""

import argparse
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure

# 配置中文支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 
                                     'Microsoft YaHei', 'PingFang HK', 'Heiti TC']
plt.rcParams['axes.unicode_minus'] = False


def show_3d_surface(nii_path, threshold=None, downsample=2):
    """
    使用等值面（isosurface）显示 3D 结构
    
    Args:
        nii_path: NIfTI 文件路径
        threshold: 等值面阈值（None 则自动选择）
        downsample: 降采样因子（减少数据量以提高速度）
    """
    print(f"加载文件: {nii_path}")
    img = nib.load(nii_path)
    data = img.get_fdata()
    
    print(f"原始数据形状: {data.shape}")
    print(f"数值范围: [{data.min():.2f}, {data.max():.2f}]")
    
    # 降采样以提高渲染速度
    if downsample > 1:
        data = data[::downsample, ::downsample, ::downsample]
        print(f"降采样后形状: {data.shape}")
    
    # 自动选择阈值
    if threshold is None:
        # 对于分割标签，使用 0.5
        if np.unique(data).size < 10:  # 可能是分割 mask
            threshold = 0.5
        else:
            # 对于灰度图像，使用均值
            threshold = np.mean(data) + 0.5 * np.std(data)
    
    print(f"使用阈值: {threshold}")
    
    # 生成等值面（marching cubes 算法）
    print("生成 3D 网格（这可能需要一些时间）...")
    try:
        verts, faces, normals, values = measure.marching_cubes(
            data, level=threshold, spacing=(1.0, 1.0, 1.0)
        )
    except Exception as e:
        print(f"错误: {e}")
        print("尝试调整阈值或检查数据...")
        return
    
    print(f"生成了 {len(verts)} 个顶点和 {len(faces)} 个面")
    
    # 创建 3D 图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制 3D 表面
    mesh = ax.plot_trisurf(
        verts[:, 0], verts[:, 1], faces, verts[:, 2],
        cmap='viridis', alpha=0.8, linewidth=0, antialiased=True
    )
    
    # 设置标签和标题
    ax.set_xlabel('X 轴', fontsize=12)
    ax.set_ylabel('Y 轴', fontsize=12)
    ax.set_zlabel('Z 轴', fontsize=12)
    ax.set_title(f'3D 表面渲染\n阈值: {threshold:.2f}', fontsize=14, pad=20)
    
    # 添加颜色条
    fig.colorbar(mesh, ax=ax, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.show()


def show_3d_scatter(nii_path, threshold=None, max_points=50000):
    """
    使用散点图显示 3D 体素
    
    Args:
        nii_path: NIfTI 文件路径
        threshold: 显示阈值
        max_points: 最大显示点数
    """
    print(f"加载文件: {nii_path}")
    img = nib.load(nii_path)
    data = img.get_fdata()
    
    print(f"数据形状: {data.shape}")
    print(f"数值范围: [{data.min():.2f}, {data.max():.2f}]")
    
    # 自动选择阈值
    if threshold is None:
        if np.unique(data).size < 10:
            threshold = 0.5
        else:
            threshold = np.percentile(data, 70)  # 只显示前 30% 的高值
    
    print(f"使用阈值: {threshold}")
    
    # 找到所有超过阈值的体素
    mask = data > threshold
    coords = np.argwhere(mask)
    values = data[mask]
    
    print(f"找到 {len(coords)} 个体素")
    
    # 如果点太多，随机采样
    if len(coords) > max_points:
        print(f"随机采样到 {max_points} 个点")
        indices = np.random.choice(len(coords), max_points, replace=False)
        coords = coords[indices]
        values = values[indices]
    
    # 创建 3D 散点图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=values, cmap='hot', alpha=0.6, s=1
    )
    
    ax.set_xlabel('X 轴', fontsize=12)
    ax.set_ylabel('Y 轴', fontsize=12)
    ax.set_zlabel('Z 轴', fontsize=12)
    ax.set_title(f'3D 体素分布\n阈值: {threshold:.2f}', fontsize=14, pad=20)
    
    fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='NIfTI 文件 3D 可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 表面渲染（适合分割标签）
  python show_nii_3d.py image.nii.gz --mode surface
  
  # 散点图（快速预览）
  python show_nii_3d.py image.nii.gz --mode scatter
  
  # 自定义阈值
  python show_nii_3d.py image.nii.gz --mode surface --threshold 0.5
        """)
    
    parser.add_argument('nii_file', help='NIfTI 文件路径')
    parser.add_argument('--mode', '-m', choices=['surface', 'scatter'],
                        default='surface', help='显示模式（默认: surface）')
    parser.add_argument('--threshold', '-t', type=float, default=None,
                        help='阈值（默认自动选择）')
    parser.add_argument('--downsample', '-d', type=int, default=2,
                        help='降采样因子（仅 surface 模式，默认: 2）')
    parser.add_argument('--max-points', '-p', type=int, default=50000,
                        help='最大显示点数（仅 scatter 模式，默认: 50000）')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'surface':
            show_3d_surface(args.nii_file, args.threshold, args.downsample)
        else:
            show_3d_scatter(args.nii_file, args.threshold, args.max_points)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()