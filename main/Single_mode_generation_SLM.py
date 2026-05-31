"""
Generate SLM hologram phase patterns for individual single-mode inputs.

This script synthesizes one selected fiber mode at a time and encodes it into
an SLM phase pattern, which is used as the single-mode input distribution.
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
import torch.nn.functional as F
#%% 所需函数
def resample_field(Z, X_old, Y_old, X_new, Y_new):
    """
    物理尺寸不变的复数场重采样（正确方式）

    Z:     [H_old, W_old] complex
    X_old: [H_old, W_old]
    Y_old: [H_old, W_old]
    X_new: [H_new, W_new]
    Y_new: [H_new, W_new]

    return:
        Z_new: [H_new, W_new] complex
    """
    # 拆成实部和虚部
    Z_real = Z.real
    Z_imag = Z.imag

    # 插值函数
    def interp_real(Zr):
        Zr_in = Zr.unsqueeze(0).unsqueeze(0)

        x_min, x_max = X_old.min(), X_old.max()
        y_min, y_max = Y_old.min(), Y_old.max()

        Xn = 2 * (X_new - x_min) / (x_max - x_min) - 1
        Yn = 2 * (Y_new - y_min) / (y_max - y_min) - 1

        grid = torch.stack((Xn, Yn), dim=-1)

        Zr_new = F.grid_sample(
            Zr_in,
            grid.unsqueeze(0),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )

        return Zr_new.squeeze(0).squeeze(0)

    Zr_new = interp_real(Z_real)
    Zi_new = interp_real(Z_imag)

    return Zr_new + 1j * Zi_new
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
    
#%% 定义SLM参数
if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
device = "cuda" if torch.cuda.is_available() else "cpu"
Nx=1920#SLM像素数x
Ny=1200#SLM像素数y
dx = 8000e-6 #SLM像素大小x
dy = 8000e-6 #SLM像素大小y
H = Nx*dx
W = Ny*dy
grid_points = 256 #每一个模式的期望采样数
#%% 生成整个SLM范围网格
x_slm = torch.linspace(-H/2, H/2, grid_points, 
                   dtype=torch.float32, device = device)
y_slm = torch.linspace(-W/2, W/2, grid_points,
                   dtype=torch.float32, device = device)
X_slm, Y_slm = torch.meshgrid(x_slm, y_slm, indexing='xy')  # 确保坐标方向正确
#%% 生成光斑范围网格
beam_radius = 1.5 #光束半径
# 步长 2*dx，覆盖棋盘单元
x_beam = torch.linspace(-beam_radius, beam_radius, grid_points, 
                   dtype=torch.float32, device = device)
y_beam = torch.linspace(-beam_radius, beam_radius, grid_points,
                   dtype=torch.float32, device = device)
X_beam, Y_beam = torch.meshgrid(x_beam, y_beam, indexing='xy')  # 确保坐标方向正确
#%% 定义光纤
core_D=20e-3
Lambda=780e-6
L=10000
n_Core = 1.4469
NA = 0.22
Rho = np.inf            # 曲率半径 (无弯曲)
Theta = 0               # 弯曲方向 (rad)
mag = 1.0               # 模式放大率
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
#%% 将模式空间分布插值到入射光场空间分布(小网格-大网格)
PIM_beam = torch.zeros_like(PIM_clean)
for i in range(clean_mode):
    mode_single = PIM_clean[:,i].reshape(grid_points, grid_points)
    PIM_beam[:,i] = fib.interpolate_mode_to_large_grid(mode_single, X_beam, Y_beam, X_slm, Y_slm).flatten().T
    
# PIM_beam矩阵能量归一化
PIM_beam, energy_integral, energy_normalized = ut.normalize_energy(PIM_beam, X_slm, Y_slm)
ut.figComplexField(X_slm, Y_slm, PIM_beam[:,1].reshape(grid_points,grid_points), 0, string='Target field') #填补后的光场
#%% 基于生成整个SLM范围网格
# 步长 2*dx，覆盖棋盘单元
Nx_points = Nx // 2
Ny_points = Ny // 2
x_range = torch.linspace(-Nx/2*dx, Nx/2*dx - 2*dx, steps=Nx_points, device=device)
y_range = torch.linspace(-Ny/2*dx, Ny/2*dx - 2*dx, steps=Ny_points, device=device)
X, Y = torch.meshgrid(x_range, y_range, indexing='xy')  # 行 = y，列 = x
#%% 重新对光场网格采样
xi = torch.arange(-Nx/2, Nx/2, device=device) * dx
yi = torch.arange(-Ny/2, Ny/2, device=device) * dx
Xi, Yi = torch.meshgrid(xi, yi, indexing="xy")
#%% 把模式分布保持比例插值回SLM采样数的一半
PIM_slm = torch.zeros(X.shape[0]*X.shape[1], PIM_beam.shape[1], dtype = torch.cfloat, device = device)
for i in range(clean_mode):
    mode_slm = PIM_beam[:,i].reshape(grid_points, grid_points)
    PIM_slm[:,i] = resample_field(mode_slm, X_slm, Y_slm, X, Y).flatten().T
#%% 定义期望生成的分布
N = 4 #生成的模式索引
for i in range(N):
    PIM_selec = PIM_slm[:, :N]
    coefficients = torch.zeros(1,N, dtype = torch.cfloat, device = device)
    coefficients[0, i] = 1.0 #单一模式赋值
    Hp, Wp = X.shape
    target = synthesize_field(coefficients, PIM_selec, Hp, Wp)
    #%% 生成棋盘调制相位
    Lambda_tensor = torch.tensor(Lambda, dtype = torch.float, device = device)
    SLMdesign = fib.SLMSimulator(Nx=Nx, Ny=Ny, dx=dx, wavelength=Lambda_tensor,
                                 Et = target, device=device)
    SLMpattern, E_m= SLMdesign.simulate(beam_radius = beam_radius, X = Xi, Y = Yi, Ein = None, Shining=True, 
                                        period_pixels = 10, show = False)
    #%% 保存相位图
    ut.save_phase_as_bmp(E_m, filename=f"phase_mode_{i}C_grating_3mm.bmp")
