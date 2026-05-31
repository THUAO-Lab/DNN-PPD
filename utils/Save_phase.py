import torch
import numpy as np
from PIL import Image


def save_phase_as_bmp(Z_crop, filename="phase.bmp", bit_depth=8, phase_range='0-2pi'):
    """
    保存复振幅场的相位图为 BMP 文件
    
    Args:
        Z_crop: torch.Tensor, shape=(Ny, Nx)，复振幅场
        filename: 保存路径
        bit_depth: 灰度图位深 (8或16)
        phase_range: '0-2pi' 或 '-pi-pi'，指定SLM期望的相位范围
    """
    # 转 numpy
    Z_np = Z_crop.detach().cpu().numpy()
    
    # 计算相位 [-pi, pi]
    phase = np.angle(Z_np)
    
    # 根据SLM要求转换相位范围
    if phase_range == '0-2pi':
        # 将 [-π, π] 映射到 [0, 2π]
        phase = phase + np.pi  # 现在范围是 [0, 2π]
    
    # 映射到灰度图
    phase_norm = phase_to_grayscale(phase, bit_depth=bit_depth, phase_range=phase_range)
    phase_norm = phase_norm.astype(np.uint8) if bit_depth == 8 else phase_norm.astype(np.uint16)
    
    # 保存为 BMP 灰度图
    img = Image.fromarray(phase_norm)
    img.save(filename, format="BMP")
    print(f"相位图已保存到 {filename}")
    
def phase_to_grayscale(phase, bit_depth=8, return_type='numpy', phase_range='0-2pi'):
    """
    将相位图转换为灰度图

    Args:
        phase: 相位图 (numpy 或 torch 张量)
        bit_depth: 灰度图位深 (8: 0-255, 16: 0-65535)
        return_type: 'numpy' 或 'torch'
        phase_range: 输入相位的范围 ('0-2pi' 或 '-pi-pi')
    Returns:
        灰度图，类型与 return_type 一致
    """

    # 统一转换为 torch 张量便于处理（保留设备信息）
    is_torch = torch.is_tensor(phase)
    if is_torch:
        phase_tensor = phase
    else:
        phase_tensor = torch.from_numpy(phase)

    # 根据相位范围转换到 [0, 2π]
    if phase_range == '-pi-pi':
        phase_tensor = phase_tensor + torch.pi
    # 注意：若 phase_range 已经是 '0-2pi'，则无需转换

    # 确保相位在 [0, 2π] 内（防止浮点误差）
    phase_tensor = torch.clamp(phase_tensor, min=0.0, max=2*torch.pi)

    # 归一化到 [0, 1]
    phase_norm = phase_tensor / (2 * torch.pi)

    max_val = 2**bit_depth - 1
    # 映射到灰度并四舍五入
    gray = torch.round(phase_norm * max_val).to(
        torch.uint8 if bit_depth == 8 else torch.uint16
    )

    if return_type == 'numpy':
        return gray.cpu().numpy()
    else:
        return gray

# 使用示例
# save_phase_as_bmp(Z_crop, "slm_phase.bmp")
