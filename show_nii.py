#!/usr/bin/env python3
"""
NIfTI 文件可视化脚本
用法: python3 show_nii.py <nii_file_path>

# 三个切片视图解释
 - 1. 轴向切片（Axial）
也叫横断面或水平切面
就像把人体水平切开，从头顶往下看
类似 CT/MRI 扫描时，躺在床上被一层层扫描的那个方向
可以看到左右对称的结构

 - 2. 矢状切片（Sagittal）
也叫侧面切面
就像把人体从中间垂直切开，从左侧或右侧看
可以看到前后方向的结构（比如鼻子、脑、脊柱的前后关系）

 - 3. 冠状切片（Coronal）
也叫额状切面或正面切面
就像把人体从前往后垂直切开，从正面或背面看
可以看到左右和上下的结构

# 为什么需要三个视图？
因为医学图像数据是 3D 的（有宽度、高度、深度），但屏幕是 2D 的。通过这三个互相垂直的切面，医生可以从不同角度全面观察器官的形态、病变的位置和大小。
在你的脚本中：
 - 非交互模式：同时显示三个视图的中间切片，让你快速了解整个 3D 数据
 - 交互模式（-i 参数）：选择一个视图，然后用滑块逐层浏览所有切片

例子：
python3 show_nii.py nnUNet_raw/Dataset004_Hippocampus/imagesTr/hippocampus_001_0000.nii.gz
交互模式：
python3 show_nii.py nnUNet_raw/Dataset004_Hippocampus/imagesTr/hippocampus_001_0000.nii.gz -i
"""

import argparse
import sys
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 配置 matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'Microsoft YaHei', 
                                     'PingFang HK', 'Heiti TC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


class NiftiViewer:
    def __init__(self, nii_path):
        """初始化 NIfTI 查看器"""
        if not os.path.exists(nii_path):
            raise FileNotFoundError(f"文件不存在: {nii_path}")
        
        print(f"正在加载文件: {nii_path}")
        self.nii_img = nib.load(nii_path)
        self.data = self.nii_img.get_fdata()
        self.nii_path = nii_path
        
        print(f"数据形状: {self.data.shape}")
        print(f"数据类型: {self.data.dtype}")
        print(f"数值范围: [{self.data.min():.2f}, {self.data.max():.2f}]")
        
        # 归一化数据以便更好地显示
        self.data_norm = self._normalize_data(self.data)
        
    def _normalize_data(self, data):
        """归一化数据到 [0, 1] 范围"""
        data_min = data.min()
        data_max = data.max()
        if data_max - data_min > 0:
            return (data - data_min) / (data_max - data_min)
        else:
            return data
    
    def show_slices(self):
        """显示三个正交切面的中间切片"""
        # 获取中间切片索引
        mid_axial = self.data.shape[2] // 2
        mid_sagittal = self.data.shape[0] // 2
        mid_coronal = self.data.shape[1] // 2
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 轴向切片 (Axial)
        axes[0].imshow(self.data_norm[:, :, mid_axial].T, cmap='gray', origin='lower')
        axes[0].set_title(f'轴向切片 (Axial)\n切片: {mid_axial}/{self.data.shape[2]}')
        axes[0].axis('off')
        
        # 矢状切片 (Sagittal)
        axes[1].imshow(self.data_norm[mid_sagittal, :, :].T, cmap='gray', origin='lower')
        axes[1].set_title(f'矢状切片 (Sagittal)\n切片: {mid_sagittal}/{self.data.shape[0]}')
        axes[1].axis('off')
        
        # 冠状切片 (Coronal)
        axes[2].imshow(self.data_norm[:, mid_coronal, :].T, cmap='gray', origin='lower')
        axes[2].set_title(f'冠状切片 (Coronal)\n切片: {mid_coronal}/{self.data.shape[1]}')
        axes[2].axis('off')
        
        plt.suptitle(f'NIfTI 查看器: {os.path.basename(self.nii_path)}', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def show_interactive(self, view='axial'):
        """显示可交互的切片浏览器"""
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.15)
        
        if view == 'axial':
            max_slice = self.data.shape[2] - 1
            init_slice = max_slice // 2
            img_data = self.data_norm[:, :, init_slice].T
            view_name = '轴向 (Axial)'
        elif view == 'sagittal':
            max_slice = self.data.shape[0] - 1
            init_slice = max_slice // 2
            img_data = self.data_norm[init_slice, :, :].T
            view_name = '矢状 (Sagittal)'
        elif view == 'coronal':
            max_slice = self.data.shape[1] - 1
            init_slice = max_slice // 2
            img_data = self.data_norm[:, init_slice, :].T
            view_name = '冠状 (Coronal)'
        else:
            raise ValueError(f"不支持的视图: {view}")
        
        im = ax.imshow(img_data, cmap='gray', origin='lower')
        ax.set_title(f'{view_name} - 切片: {init_slice}/{max_slice}')
        ax.axis('off')
        
        # 添加滑块
        ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
        slider = Slider(ax_slider, '切片', 0, max_slice, valinit=init_slice, valstep=1)
        
        def update(val):
            slice_idx = int(slider.val)
            if view == 'axial':
                img_data = self.data_norm[:, :, slice_idx].T
            elif view == 'sagittal':
                img_data = self.data_norm[slice_idx, :, :].T
            elif view == 'coronal':
                img_data = self.data_norm[:, slice_idx, :].T
            
            im.set_data(img_data)
            ax.set_title(f'{view_name} - 切片: {slice_idx}/{max_slice}')
            fig.canvas.draw_idle()
        
        slider.on_changed(update)
        
        plt.suptitle(f'文件: {os.path.basename(self.nii_path)}', fontsize=12)
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='NIfTI (.nii / .nii.gz) 文件可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python show_nii.py image.nii.gz
  python show_nii.py image.nii.gz --view axial
  python show_nii.py image.nii.gz --interactive --view sagittal
        """)
    
    parser.add_argument('nii_file', help='NIfTI 文件路径 (.nii 或 .nii.gz)')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='交互模式，使用滑块浏览切片')
    parser.add_argument('--view', '-v', choices=['axial', 'sagittal', 'coronal'],
                        default='axial', help='选择视图方向 (默认: axial)')
    
    args = parser.parse_args()
    
    try:
        viewer = NiftiViewer(args.nii_file)
        
        if args.interactive:
            viewer.show_interactive(view=args.view)
        else:
            viewer.show_slices()
            
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()