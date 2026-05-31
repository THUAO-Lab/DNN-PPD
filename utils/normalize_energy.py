import torch
def normalize_energy(PIM, X, Y, epsilon=1e-10):
    """
    考虑网格间距的能量归一化（严格的能量归一化）
    
    参数:
        PIM: 模式矩阵，形状为 [H*W, M] 或 [H, W, M] 的复数张量
        X, Y: 坐标网格，形状为 [H, W] 或 [H, 1] 和 [W, 1] 或 [H] 和 [W]
        epsilon: 防止除零的小量
        
    返回:
        PIM_normalized: 能量归一化后的模式矩阵
        energies: 原始每个模式的能量（考虑dxdy）
    """
    # 保存原始形状
    original_shape = PIM.shape
    original_dim = PIM.dim()
    
    # 将PIM转换为三维形式 [H, W, M] 以便处理空间维度
    if original_dim == 2:
        # PIM是 [H*W, M]，需要知道H和W
        H_times_W, M = original_shape
        # 我们需要从X和Y推断H和W
        if X.dim() == 2:
            H, W = X.shape
        elif X.dim() == 1:
            H, W = X.shape[0], Y.shape[0]
        else:
            raise ValueError(f"无法从X的形状 {X.shape} 推断网格尺寸")
            
        if H * W != H_times_W:
            raise ValueError(f"PIM的展平尺寸{H_times_W}与推断的网格{H}×{W}={H*W}不匹配")
        
        # 重塑为 [H, W, M]
        PIM_3d = PIM.reshape(H, W, M)
    elif original_dim == 3:
        # 已经是三维，假设是 [H, W, M]
        H, W, M = original_shape
        PIM_3d = PIM
    else:
        raise ValueError(f"PIM 的维度 {original_dim} 不支持")
    
    # 确保是复数类型
    if not torch.is_complex(PIM_3d):
        PIM_3d = PIM_3d.to(torch.complex64)
    
    # 计算网格间距 dx, dy
    if X.dim() == 2:
        # X, Y 是二维网格
        dx = (X[0, 1] - X[0, 0]).abs().item() if W > 1 else 1.0
        dy = (Y[1, 0] - Y[0, 0]).abs().item() if H > 1 else 1.0
    elif X.dim() == 1:
        # X, Y 是一维向量
        dx = (X[1] - X[0]).abs().item() if X.shape[0] > 1 else 1.0
        dy = (Y[1] - Y[0]).abs().item() if Y.shape[0] > 1 else 1.0
    else:
        raise ValueError(f"X 的维度 {X.dim()} 不支持")
    
    # 计算面积元 dA = dx * dy
    dA = dx * dy
    
    # 计算每个模式的能量（考虑dxdy）：∑∑|E|² dxdy ≈ dA * ∑∑|E|²
    # 这里假设矩形网格且dx, dy均匀
    energy_per_pixel = torch.abs(PIM_3d) ** 2  # [H, W, M]
    energy_sum = torch.sum(energy_per_pixel, dim=(0, 1))  # [M]
    energy_integral = energy_sum * dA  # [M]
    
    # 添加epsilon防止除零
    energy_safe = energy_integral + epsilon
    
    # 计算归一化因子：1/√(能量积分)
    norm_factors = 1.0 / torch.sqrt(energy_safe)  # [M]
    
    # 应用归一化：每个模式乘以其归一化因子
    # 使用广播：PIM_3d [H, W, M] * norm_factors [1, 1, M]
    norm_factors_3d = norm_factors.view(1, 1, -1)
    PIM_normalized_3d = PIM_3d * norm_factors_3d
    
    # 验证归一化后的能量
    energy_normalized = torch.sum(torch.abs(PIM_normalized_3d) ** 2, dim=(0, 1)) * dA
    
    
    # 转换回原始形状
    if original_dim == 2:
        PIM_normalized = PIM_normalized_3d.reshape(H*W, M)
    else:
        PIM_normalized = PIM_normalized_3d
    
    return PIM_normalized, energy_integral, energy_normalized