"""
Analyze few-mode fiber modes in the spatial-frequency domain.

This script computes the FFT of individual mode fields, synthesizes mixed
fields from modal coefficients, and evaluates frequency-domain overlap with
the reference modes.
"""
#%% 导入包
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import torch
import utils as ut
import fiber as fib
import matplotlib.pyplot as plt
#%% 所需函数
# 生成光纤模式基下的光场数据
def frequency_domain(PIM_mode, H=512, W=512, device='cpu'):
    """
    将单个光纤模式或混合场转换到频域（FFT）。
    
    参数：
    ----------
    PIM_mode : torch.Tensor
        输入光场系数或模式向量，形状可以是 (N_modes,) 或 (H*W,) 复数
        如果是一维系数向量，会先恢复到 H*W 网格。
    H, W : int
        空间域网格大小
    device : str
        'cpu' 或 'cuda'

    返回：
    -------
    F_field : torch.Tensor
        频域光场，shape [H, W], 复数
    """
    
    # 转到设备
    PIM_mode = PIM_mode.to(device)

    # === 1. 判断是否需要展开为网格 ===
    if PIM_mode.dim() == 1:
        if PIM_mode.numel() == H*W:
            # 直接reshape成网格
            E_field = PIM_mode.reshape(H, W)
        else:
            raise ValueError(f"PIM_mode长度 {PIM_mode.numel()} 与网格大小 {H*W} 不匹配")
    elif PIM_mode.dim() == 2 and PIM_mode.shape == (H, W):
        E_field = PIM_mode
    else:
        raise ValueError(f"PIM_mode维度不符合要求: {PIM_mode.shape}")

    # === 2. 做 FFT ===
    # torch.fft.fft2 默认频域排列：低频在角落
    F_field = torch.fft.fft2(E_field)

    # === 3. fftshift 频谱中心化 ===
    F_field = torch.fft.fftshift(F_field)

    return F_field
    

def synthesize_field(coeff, modes_ref, H, W):
    """
    根据输入模式系数和模式分布生成光场 E

    Parameters
    ----------
    coeff : torch.Tensor
        (B, N_mode) 复数，输入模式系数
    modes_ref : torch.Tensor
        (H*W, N_mode) 复数，每个模式的空间分布（列向量）
    H, W : int
        输出光场大小

    Returns
    -------
    E_out : torch.Tensor
        (B, H, W) 复数光场
    """
    # 叠加模式
    E_flat = torch.matmul(coeff, modes_ref.T)  # (B, H*W)
    E_out = E_flat.reshape(H, W)

    return E_out

def compute_overlap(
    E_mix,          # (H, W) complex，混合光场
    modes_ref,      # (H*W, N) complex，参考模式（已能量归一化）
    dx, dy          # 像素物理尺寸
):
    """
    计算混合光场与每个参考模式的重叠积分系数

    返回：
    -------
    coeffs : torch.Tensor
        (N,) complex，重叠积分系数
    """

    device = E_mix.device
    dtype  = E_mix.dtype

    H, W = E_mix.shape
    _, N = modes_ref.shape

    # 展平混合场
    E_flat = E_mix.reshape(-1)  # (H*W,)

    coeffs = torch.zeros(N, device=device, dtype=dtype)

    for n in range(N):
        u_flat = modes_ref[:, n]  # (H*W,)

        # ===== 核心：与你给的一模一样 =====
        coeffs[n] = torch.sum(torch.conj(u_flat) * E_flat) * dx * dy

    return coeffs
#%% 划分网格
device = "cpu"
grid_points = 256 #每一个模式的采样分布
display_range = 100e-3  # 显示范围±250μm
# 生成坐标网格（覆盖±250μm）
x = torch.linspace(-display_range, display_range, grid_points, 
                   dtype=torch.float32, device = device)
y = torch.linspace(-display_range, display_range, grid_points,
                   dtype=torch.float32, device = device)
X, Y = torch.meshgrid(x, y, indexing='xy')  # 确保坐标方向正确
dx = (X[0,1]-X[0,0]).detach().numpy()
dy = (Y[1,0]-Y[0,0]).detach().numpy()
#%% 导入光纤
core_D=20e-3
Lambda=780e-6
L=10000
n_Core = 1.4469
NA = 0.22
Rho = np.inf            # 曲率半径 (无弯曲)
Theta = 0               # 弯曲方向 (rad)
# 实例化光纤对象
fiber = fib.MultimodeFiberLP(Lambda, core_D, n_Core, NA, L, Rho, Theta, grid_points)
# 求解光纤模式
fiber.solve_modes_LP()
field_distributions = fiber.get_LG_total_field(pol = 'x')
max_modes=field_distributions.shape[2]
prop_constants = fiber.propconst_LG
# 画LP模式 (x偏振)
PIM=torch.tensor(field_distributions[:, :, :max_modes].reshape(-1, max_modes), dtype=torch.cfloat).to(device)#模式基
lm_index = list(zip(fiber.lmap[:max_modes], fiber.mmap[:max_modes]))
#去除y分量
PIM_clean, lm_index_clean = ut.clean_zero_modes(PIM, lm_index, pol='x')
clean_mode = PIM_clean.shape[1]
#%% 生成光纤端面网格
x_fiber = torch.linspace(-core_D/2, core_D/2, grid_points, 
                   dtype=torch.float32, device = device)
y_fiber = torch.linspace(-core_D/2, core_D/2, grid_points,
                   dtype=torch.float32, device = device)
X_fiber, Y_fiber = torch.meshgrid(x_fiber, y_fiber, indexing='xy')  # 确保坐标方向正确
#%% 将模式空间分布插值到入射光场空间分布(小网格-大网格)
PIM_beam = torch.zeros_like(PIM_clean)
for i in range(clean_mode):
    mode_single = PIM_clean[:,i].reshape(grid_points, grid_points)
    PIM_beam[:,i] = fib.interpolate_mode_to_large_grid(mode_single, X_fiber, Y_fiber, X, Y).flatten().T
    
# PIM_beam矩阵能量归一化
PIM_beam, energy_integral, energy_normalized = ut.normalize_energy(PIM_beam, X, Y)
ut.figComplexField(X, Y, PIM_beam[:,1].reshape(grid_points, grid_points), 0, string='PIM 5', 
                   cmaps=['inferno', 'twilight'])
#%% 选取前4个模式并计算频域分布
N = 4 
PIM_selec = PIM_beam[:, :N]
H,W = X.shape
PIM_frequency = torch.zeros_like(PIM_selec)
for i in range(N):
    PIM_frequency[:,i] = frequency_domain(PIM_selec[:,i], H=H, W=W, device=device).flatten().T
    
ut.figComplexField(X, Y, PIM_frequency[:,2].reshape(grid_points, grid_points), 0, string='PIM frequency', 
                   cmaps=['inferno', 'twilight'])
#%% 分析混合光场频域分布
target_freq = torch.zeros(grid_points, grid_points, N, dtype = torch.cfloat, device = device)
coeffs = torch.zeros(N, N, dtype = torch.float, device = device)
for i in range(N):
    # For each row, remove one mode from the mixed field and evaluate its
    # frequency-domain overlap with all reference modes.
    coefficients = torch.ones(1,N, dtype = torch.cfloat, device = device)
    coefficients[0, i]=0.0
    target = synthesize_field(coefficients, PIM_selec, H, W)
    target_freq[:,:,i]= frequency_domain(target, H=H, W=W, device=device)
    # 计算重叠积分
    coeffs_c = compute_overlap(target_freq[:,:,i], PIM_frequency, dx, dy)
    coeffs[i,:] = torch.abs(coeffs_c)/torch.max(torch.abs(coeffs_c))

#%% 画图
# coeffs: shape (4, 4)
values = coeffs.detach().cpu().numpy()  # 若要功率可用 **2

num_sets, num_modes = values.shape  # 4, 4

# === 参数 ===
bar_width = 0.6
group_gap = 2.0

# 颜色（论文常用、区分度好）
colors = [
    '#1f77b4',  # 蓝
    '#ff7f0e',  # 橙
    '#2ca02c',  # 绿
    '#d62728',  # 红
]

plt.figure(figsize=(10, 4))

x_ticks = []
x_ticklabels = []

current_x = 0

for i in range(num_sets):
    xs = np.arange(current_x, current_x + num_modes)
    ys = values[i]

    plt.bar(
        xs,
        ys,
        width=bar_width,
        color=colors[i],
        edgecolor='k',
        label=f'Dataset {i+1}'
    )

    x_ticks.extend(xs)
    x_ticklabels.extend([f'{j+1}' for j in range(num_modes)])

    current_x = xs[-1] + group_gap + 1

# === 分割线（增强区域感）===
current_x = -0.5
for i in range(num_sets - 1):
    current_x += num_modes + group_gap
    plt.axvline(current_x, color='gray', linestyle='--', linewidth=0.8)

# === 轴 & 图例 ===
plt.xlabel('Mode index')
plt.ylabel('Overlap coefficient')
plt.title('Overlap between mixed field and individual modes')

plt.xticks(x_ticks, x_ticklabels)
plt.legend(frameon=False, ncol=4)  # 横向排，像论文

plt.tight_layout()
plt.show()
