import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import matplotlib.pyplot as plt
import torch


def figComplexField(x, y, E, flag, string=None, mode=1, Circular_color = 0, cmaps=None):
    """
    支持PyTorch张量的复光场可视化函数
    参数：
    x, y: 坐标网格 (numpy数组或torch.Tensor)
    E: 复振幅场 (numpy数组或torch.Tensor)
    flag: 0-空间坐标 / 1-频域坐标
    string: 标题后缀
    mode: 1-pcolor图 / 2-imshow图
    """
    # 自动转换PyTorch张量为numpy
    if isinstance(E, torch.Tensor):
        E = E.detach().cpu().numpy()
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    # 统一处理复数数据
    E_amp = np.abs(E)
    E_pha = np.angle(E)
    
    # 创建可视化图形
    fig = plt.figure(figsize=(10, 4))
    plot_method = {1: 'pcolor', 2: 'imshow'}.get(mode, 'pcolor')
    titles = ['Amplitude', 'Phase']
    if cmaps==None: cmaps = ['hot', 'gray'] 
    else: cmaps=cmaps  # 调整相位使用hsv色图更直观
    
    for i, (data, cmap) in enumerate(zip([E_amp, E_pha], cmaps)):
        ax = plt.subplot(1, 2, i+1)
        
        # 选择绘图方法
        if plot_method == 'pcolor':
            img = ax.pcolor(x, y, data, shading='auto', cmap=cmap)
        else:
            img = ax.imshow(data, cmap=cmap)
        
        # 公共样式设置
        # 添加环形颜色条
        if Circular_color==0:
            cbar = plt.colorbar(img, ax=ax)
            cbar.ax.tick_params(labelsize=12)
        else:
            add_circular_colorbar(fig, ax, cmap=cmap, vmin=0, vmax=2*np.pi, 
                     label="Phase Angle")

        
        # 坐标标签设置
        unit = (' (mm)', r' (mm$^{-1}$)')[flag]
        ax.set_xlabel(f"{['x','kx'][flag]}{unit}", fontsize=14)
        ax.set_ylabel(f"{['y','ky'][flag]}{unit}", fontsize=14)
        ax.set_title(f'{titles[i]} {string or ""}', fontsize=16)
        
        # 统一刻度设置
        ax.tick_params(axis='both', which='major', labelsize=12)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname('Arial')
            label.set_fontsize(12)

    plt.gcf().set_facecolor('white')
    plt.tight_layout()
    plt.show()
    
def add_circular_colorbar(fig, ax, cmap='hsv', vmin=0, vmax=2*np.pi, 
                          label="Phase (rad)", tick_labels=None):
    """
    添加环形颜色条（适用于极坐标数据）

    参数:
        fig: matplotlib Figure 对象
        ax: 主图的坐标轴
        cmap: 颜色映射 (默认 hsv)
        vmin, vmax: 数据范围 (例如 0 到 2π)
        label: 颜色条标签
        tick_labels: 自定义刻度标签 (例如 [0, π, 2π])
    """
    # 创建极坐标系用于颜色条
    cax = fig.add_axes([0.85, 0.15, 0.15, 0.15], projection='polar')  # 调整位置和大小
    
    # 生成角度数据
    theta = np.linspace(0, 2*np.pi, 256)
    r = np.linspace(0.5, 1, 2)  # 环形半径范围
    
    # 创建颜色网格
    theta_grid, r_grid = np.meshgrid(theta, r)
    data = np.tile(theta, (2, 1))  # 复制角度数据
    
    # 绘制颜色环
    cax.pcolormesh(theta, r, data.T, cmap=cmap, shading='auto')
    
    # 设置极坐标参数
    cax.set_theta_zero_location('E')  # 0 度方向向右
    cax.set_theta_direction(-1)       # 顺时针方向
    cax.set_rticks([])                # 隐藏半径刻度
    
    # 设置角度刻度
    if tick_labels is None:
        cax.set_xticks(np.linspace(0, 2*np.pi, 8))
        cax.set_xticklabels(['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', 
                            r'$\pi$', r'$5\pi/4$', r'$3\pi/2$', r'$7\pi/4$'])
    else:
        cax.set_xticks(np.linspace(0, 2*np.pi, len(tick_labels)))
        cax.set_xticklabels(tick_labels)