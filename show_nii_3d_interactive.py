#!/usr/bin/env python3
"""
NIfTI 文件交互式 3D 可视化（使用 Plotly）
用法: python3 show_nii_3d_interactive.py <nii_file_path>
例子：
python3 show_nii_3d_interactive.py nnUNet_raw/Dataset004_Hippocampus/labelsTr/hippocampus_001.nii.gz
"""

import argparse
import numpy as np
import nibabel as nib
import plotly.graph_objects as go
from skimage import measure


def show_interactive_3d(nii_path, threshold=None, downsample=2):
    """使用 Plotly 创建可交互的 3D 可视化"""
    print(f"加载文件: {nii_path}")
    img = nib.load(nii_path)
    data = img.get_fdata()
    
    print(f"原始数据形状: {data.shape}")
    print(f"数值范围: [{data.min():.2f}, {data.max():.2f}]")
    
    # 降采样
    if downsample > 1:
        data = data[::downsample, ::downsample, ::downsample]
        print(f"降采样后形状: {data.shape}")
    
    # 自动选择阈值
    if threshold is None:
        if np.unique(data).size < 10:
            threshold = 0.5
        else:
            threshold = np.mean(data) + 0.5 * np.std(data)
    
    print(f"使用阈值: {threshold}")
    print("生成 3D 网格...")
    
    # 生成等值面
    verts, faces, normals, values = measure.marching_cubes(
        data, level=threshold, spacing=(1.0, 1.0, 1.0)
    )
    
    print(f"生成了 {len(verts)} 个顶点和 {len(faces)} 个面")
    
    # 创建 3D mesh
    x, y, z = verts.T
    i, j, k = faces.T
    
    fig = go.Figure(data=[
        go.Mesh3d(
            x=x, y=y, z=z,
            i=i, j=j, k=k,
            opacity=0.8,
            colorscale='Viridis',
            intensity=z,  # 根据 z 坐标着色
            name='',
            showscale=True,
            hoverinfo='text',
            text=f'3D 结构<br>阈值: {threshold:.2f}'
        )
    ])
    
    # 设置布局
    fig.update_layout(
        title=f'交互式 3D 可视化<br><sub>阈值: {threshold:.2f}</sub>',
        scene=dict(
            xaxis_title='X 轴',
            yaxis_title='Y 轴',
            zaxis_title='Z 轴',
            aspectmode='data'
        ),
        width=1000,
        height=800,
    )
    
    print("打开交互式窗口...")
    print("提示: 可以用鼠标旋转、缩放、平移")
    fig.show()


def main():
    parser = argparse.ArgumentParser(description='NIfTI 文件交互式 3D 可视化')
    parser.add_argument('nii_file', help='NIfTI 文件路径')
    parser.add_argument('--threshold', '-t', type=float, default=None,
                        help='阈值（默认自动选择）')
    parser.add_argument('--downsample', '-d', type=int, default=2,
                        help='降采样因子（默认: 2）')
    
    args = parser.parse_args()
    
    try:
        show_interactive_3d(args.nii_file, args.threshold, args.downsample)
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()