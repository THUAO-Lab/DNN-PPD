"""
光纤模式插值模块
将光纤端面的模式场插值到入射光的采样空间
"""

import numpy as np
import torch
from scipy import interpolate


def interpolate_mode_to_large_grid(mode_field_small, X_fiber, Y_fiber, 
                                   X_large, Y_large, method='linear'):
    """
    将光纤模式从小网格（光纤端面）插值到大网格（入射光采样空间）
    
    参数:
    ----------
    mode_field_small : torch.Tensor or np.ndarray
        在光纤端面小网格上的模式场，形状为 (H_fiber, W_fiber)，可以是复数
    X_fiber, Y_fiber : torch.Tensor
        光纤端面坐标网格，形状为 (H_fiber, W_fiber)
    X_large, Y_large : torch.Tensor
        大网格（入射光）坐标，形状为 (H_large, W_large)
    method : str, 可选
        插值方法，'linear' 或 'cubic'，默认为 'linear'
    
    返回:
    -------
    torch.Tensor
        插值到大网格上的模式场，形状为 (H_large, W_large)，复数类型
    """
    # 确保输入为numpy数组
    if isinstance(mode_field_small, torch.Tensor):
        mode_np = mode_field_small.detach().cpu().numpy()
    else:
        mode_np = mode_field_small
    
    if isinstance(X_fiber, torch.Tensor):
        x_fiber_np = X_fiber.detach().cpu().numpy()
        y_fiber_np = Y_fiber.detach().cpu().numpy()
    else:
        x_fiber_np = X_fiber
        y_fiber_np = Y_fiber
    
    if isinstance(X_large, torch.Tensor):
        X_large_np = X_large.detach().cpu().numpy()
        Y_large_np = Y_large.detach().cpu().numpy()
    else:
        X_large_np = X_large
        Y_large_np = Y_large
    
    # 获取一维坐标（假设网格是等间距的）
    x_fiber_1d = x_fiber_np[0, :] if x_fiber_np.ndim > 1 else x_fiber_np
    y_fiber_1d = y_fiber_np[:, 0] if y_fiber_np.ndim > 1 else y_fiber_np
    
    # 检查模式场是否为复数
    is_complex = np.iscomplexobj(mode_np)
    
    if is_complex:
        # 分别处理实部和虚部
        real_part = np.real(mode_np)
        imag_part = np.imag(mode_np)
        
        # 创建实部插值函数
        interp_real = interpolate.RegularGridInterpolator(
            (y_fiber_1d, x_fiber_1d),
            real_part,
            method=method,
            bounds_error=False,
            fill_value=0.0
        )
        
        # 创建虚部插值函数
        interp_imag = interpolate.RegularGridInterpolator(
            (y_fiber_1d, x_fiber_1d),
            imag_part,
            method=method,
            bounds_error=False,
            fill_value=0.0
        )
        
        # 在大网格上采样
        points = np.stack([Y_large_np.ravel(), X_large_np.ravel()], axis=-1)
        real_large_flat = interp_real(points)
        imag_large_flat = interp_imag(points)
        
        # 组合实部和虚部
        mode_large_flat = real_large_flat + 1j * imag_large_flat
        
    else:
        # 实数场的情况
        interp_func = interpolate.RegularGridInterpolator(
            (y_fiber_1d, x_fiber_1d),
            mode_np,
            method=method,
            bounds_error=False,
            fill_value=0.0
        )
        
        # 在大网格上采样
        points = np.stack([Y_large_np.ravel(), X_large_np.ravel()], axis=-1)
        mode_large_flat = interp_func(points)
    
    # 重塑为网格形状
    mode_large_np = mode_large_flat.reshape(X_large_np.shape)
    
    # 转换为PyTorch张量
    if is_complex:
        mode_large = torch.from_numpy(mode_large_np).to(torch.cfloat)
    else:
        mode_large = torch.from_numpy(mode_large_np).to(torch.float32)
    
    return mode_large


def interpolate_all_modes(field_distributions, fiber_coords, large_coords, 
                          device='cpu', method='linear'):
    """
    将光纤的所有模式插值到大网格
    
    参数:
    ----------
    field_distributions : np.ndarray or torch.Tensor
        光纤模式场分布，形状为 (H, W, M)，其中M是模式数量
    fiber_coords : tuple
        光纤端面坐标 (X_fiber, Y_fiber)
    large_coords : tuple
        大网格坐标 (X_large, Y_large)
    device : str
        输出张量的设备
    method : str
        插值方法
    
    返回:
    -------
    torch.Tensor
        插值后的模式矩阵，形状为 (H_large*W_large, M)
    list
        模式索引列表
    """
    X_fiber, Y_fiber = fiber_coords
    X_large, Y_large = large_coords
    
    # 获取模式数量
    if len(field_distributions.shape) == 3:
        num_modes = field_distributions.shape[2]
    else:
        num_modes = 1
    
    PIM_list = []
    
    for mode_idx in range(num_modes):
        # 获取单个模式场
        if len(field_distributions.shape) == 3:
            mode_field_small = field_distributions[:, :, mode_idx]
        else:
            mode_field_small = field_distributions
        
        # 插值到大网格
        mode_field_large = interpolate_mode_to_large_grid(
            mode_field_small, X_fiber, Y_fiber, X_large, Y_large, method
        )
        
        # 展平并添加到列表
        PIM_list.append(mode_field_large.ravel())
    
    # 构建PIM矩阵 (H*W, M)
    if PIM_list:
        PIM = torch.stack(PIM_list, dim=1).to(device)
    else:
        PIM = torch.tensor([], device=device)
    
    return PIM


def create_fiber_coordinates(core_diameter, grid_points):
    """
    创建光纤端面的坐标网格
    
    参数:
    ----------
    core_diameter : float
        纤芯直径（米）
    grid_points : int
        网格点数
    
    返回:
    -------
    tuple
        (x_fiber, y_fiber, X_fiber, Y_fiber)
    """
    fiber_range = core_diameter / 2  # ±纤芯半径
    
    # 创建光纤端面坐标
    x_fiber = torch.linspace(-fiber_range, fiber_range, grid_points, dtype=torch.float32)
    y_fiber = torch.linspace(-fiber_range, fiber_range, grid_points, dtype=torch.float32)
    X_fiber, Y_fiber = torch.meshgrid(x_fiber, y_fiber, indexing='xy')
    
    return x_fiber, y_fiber, X_fiber, Y_fiber


def create_incident_coordinates(display_range, grid_points, device='cpu'):
    """
    创建入射光采样空间的坐标网格
    
    参数:
    ----------
    display_range : float
        显示范围（米）
    grid_points : int
        网格点数
    device : str
        设备
    
    返回:
    -------
    tuple
        (x, y, X, Y, dx, dy)
    """
    # 生成坐标网格
    x = torch.linspace(-display_range, display_range, grid_points, 
                       dtype=torch.float32, device=device)
    y = torch.linspace(-display_range, display_range, grid_points,
                       dtype=torch.float32, device=device)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    
    # 计算网格间距
    dx = (X[0, 1] - X[0, 0]).detach().cpu().numpy()
    dy = (Y[1, 0] - Y[0, 0]).detach().cpu().numpy()
    
    return x, y, X, Y, dx, dy


# 测试函数
if __name__ == "__main__":
    # 测试插值函数
    print("测试模式插值函数...")
    
    # 创建测试数据
    grid_points = 64
    core_diameter = 60e-6
    display_range = 1e-4
    
    # 创建光纤端面坐标
    x_fiber, y_fiber, X_fiber, Y_fiber = create_fiber_coordinates(core_diameter, grid_points)
    
    # 创建入射光坐标
    x, y, X, Y, dx, dy = create_incident_coordinates(display_range, grid_points)
    
    # 创建测试模式（高斯光束）
    sigma = core_diameter / 4
    r_fiber = torch.sqrt(X_fiber**2 + Y_fiber**2)
    test_mode = torch.exp(-r_fiber**2 / (2 * sigma**2))
    
    # 测试单个模式插值
    mode_large = interpolate_mode_to_large_grid(test_mode, X_fiber, Y_fiber, X, Y)
    
    print(f"光纤端面网格形状: {X_fiber.shape}")
    print(f"入射光网格形状: {X.shape}")
    print(f"插值后模式形状: {mode_large.shape}")
    print("测试完成!")