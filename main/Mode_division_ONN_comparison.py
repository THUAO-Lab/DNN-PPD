"""
Compare spherical-phase and microlens-array ONN modal demultiplexing.

This script trains or loads two optical neural network phase modulators under
the same few-mode input/output settings:

1. A traditional single spherical phase design.
2. A microlens-array phase design.

The comparison uses the old clamp phase mapping, reconstructs the output mode
array, and evaluates the two designs with SSIM, coefficient error, overlap, and
phase visualization.
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
from torch.utils.data import TensorDataset
import ONNtrain as ONT

PRETRAINED_DIR = PROJECT_DIR / "Pre_trained_DNN"

#%% 所需函数
# 生成光纤模式基下的光场数据
def generate_modes_field(
    tol,
    num_modes,
    PIM,
    resolution,
    device,
    max_zero_modes=2
):
    """
    生成随机光纤模式叠加场
    每个样本随机将 1~max_zero_modes 个模式系数置为 0

    返回:
        batch: [tol, H, W]
        coeffs_batch: [tol, num_modes]
        zero_mask: [tol, num_modes]，True 表示该模式被置 0
    """

    batch = torch.zeros(
        tol, resolution, resolution,
        dtype=torch.complex64, device=device
    )
    coeffs_batch = torch.zeros(
        tol, num_modes,
        dtype=torch.complex64, device=device
    )
    zero_mask = torch.zeros(
        tol, num_modes,
        dtype=torch.bool, device=device
    )

    # 统一 PIM 形状为 [H*W, num_modes]
    if PIM.dim() == 3:
        PIM_flat = PIM.view(num_modes, -1).T
    elif PIM.dim() == 2:
        PIM_flat = PIM
    else:
        raise ValueError("PIM must be 2D or 3D")

    for i in range(tol):
        # 随机幅度
        amplitudes = 5.0 * torch.rand(num_modes, device=device)
        # amplitudes = 1.0 * torch.ones(num_modes, device=device)

        # 构造复系数（目前只有实幅度，后续你可加相位）
        coeffs = amplitudes.to(torch.complex64)

        # ---------- 新的随机置零逻辑 ----------
        num_zero = torch.randint(
            1, max_zero_modes + 1, (1,), device=device
        ).item()

        zero_indices = torch.randperm(
            num_modes, device=device
        )[:num_zero]

        coeffs[zero_indices] = 0.0
        zero_mask[i, zero_indices] = True
        # ------------------------------------

        field_flat = PIM_flat @ coeffs.unsqueeze(1)
        field = field_flat.view(resolution, resolution)

        batch[i] = field
        coeffs_batch[i] = coeffs

    return batch, coeffs_batch, zero_mask
    
def Orthogonal_test(PIM):
    G = PIM.conj().T @ PIM  # shape [N_modes, N_modes]
    aa = G.detach().numpy()
    return aa

def log_callback(step_or_epoch, loss, is_lbfgs):
    prefix = "LBFGS" if is_lbfgs else "Adam"
    print(f"[{prefix} Step {step_or_epoch:03d}] Loss = {loss:.4e}")
#%% 划分网格
if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
device = "cuda" if torch.cuda.is_available() else "cpu"
grid_points = 512 #每一个模式的采样分布
Layer_num = 4
PHASE_MAPPING = "clamp"
Eval = False
SP_MODEL_PATH = PRETRAINED_DIR / "modulator_comparison_8lemode_4layer_SP_1000.pth"
ARRAY_MODEL_PATH = PRETRAINED_DIR / "modulator_comparison_8lemode_4layer_array_1000.pth"
display_range = 60e-3  # 显示范围±250μm
beam_radius = 40e-3   # 实际光束半径50μm
N = 9 #所需模式数
# 生成坐标网格（覆盖±250μm）
x = torch.linspace(-display_range, display_range, grid_points, 
                   dtype=torch.float32, device = device)
y = torch.linspace(-display_range, display_range, grid_points,
                   dtype=torch.float32, device = device)
X, Y = torch.meshgrid(x, y, indexing='xy')  # 确保坐标方向正确
r = torch.sqrt(X**2 + Y**2)            # 计算径向距离
mask = ((r <= beam_radius))     # 生成二值掩膜
dx = (X[0,1]-X[0,0]).detach().cpu().numpy()
dy = (Y[1,0]-Y[0,0]).detach().cpu().numpy()
#%% 导入光纤
core_D=20e-3
# Lambda=multimode_fiber_para['lambda'][0]*1e-6
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
# fiber.plot_LG_modes()
PIM=torch.tensor(field_distributions[:, :, :max_modes].reshape(-1, max_modes), dtype=torch.cfloat).to(device)#模式基
lm_index = list(zip(fiber.lmap[:max_modes], fiber.mmap[:max_modes]))
#去除y分量
PIM_clean, lm_index_clean = ut.clean_zero_modes(PIM, lm_index, pol='x')
clean_mode = PIM_clean.shape[1]
T = np.diag(np.exp(1j*fiber.propconst_LG*L))
field_x = field_distributions[:, :, :clean_mode]
# C, T_C = fiber.compute_coupling_matrix(prop_constants, L, field_x, delta_eps=0) #计算耦合矩阵
delta_eps_map = 3e-2*np.ones([grid_points, grid_points])
C, T_C = fiber.compute_mode_coupling_T(prop_constants, L, field_x, delta_eps_map, dx, dy)
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
#%% 选取所需的模式分布
PIM_selec = PIM_beam[:,:N] #PIM_beam做正交检验对角线不是1，而是一个小数，可能需要放缩
point = int(torch.ceil(torch.sqrt(torch.as_tensor(N, dtype=torch.int))).item()) #单向点位
#%% 生成模式点位以及阵列
#确定范围
full_range=(-display_range, display_range, -display_range, display_range)
phys_range=(-display_range, display_range, -display_range, display_range)

H,W = X.shape
centers_xy_target = ut.generate_grid_centers(
    I=X,
    n_row=point,
    n_col=point,
    phys_range = phys_range,
    full_range = full_range
)
E_target_tra = ut.place_PIM_modes(
    detector_shape=(H, W),
    PIM=PIM_selec,
    mode_indices=torch.arange(N, device=device),
    centers_xy=centers_xy_target,
    coeff = torch.ones(1,N,dtype = torch.cfloat, device=device),
)

# ut.figComplexField(X, Y, E_target_tra[0,:,:], 
#                     0, string='E in', cmaps = ['inferno', 'twilight']) #画图观察
#%% 生成训练集
tol, P_min, P_max = 1000, 1, N
# 生成随机幅度
E_all, Para_all, zero_flags = generate_modes_field(tol, P_max, PIM_selec, grid_points, device, max_zero_modes=2)
E_all = E_all.detach()
Para_all = Para_all.detach()

# ut.figComplexField(X, Y, E_all[1,:,:], 
#                    0, string=f'E_{i}', cmaps = ['jet', 'gray']) #训练样本
E_target_tra = ut.place_PIM_modes(
    detector_shape=(H, W),
    PIM=PIM_selec,
    mode_indices=torch.arange(N, device=device),
    centers_xy=centers_xy_target,
    coeff = Para_all,
)
tra_set = TensorDataset(E_all)
tra_para = TensorDataset(Para_all) 
E_tra_target = TensorDataset(E_target_tra) #训练集
# ut.figComplexField(X, Y, E_target_tra[1,:,:], 0, string='Mode array', 
#                    cmaps=['inferno', 'twilight']) #平铺后的光场分布
#%% 生成验证集
val, P_min, P_max = 100, 1, N
PIM_selec = PIM_beam[:,:N]
# 生成随机幅度
E_all_val, Para_all_val,_ = generate_modes_field(val, P_max, PIM_selec, grid_points, device, max_zero_modes=2)
E_val_all = E_all_val.detach()
Para_val_all = Para_all_val.detach()
E_target_val = ut.place_PIM_modes(
    detector_shape=(H, W),
    PIM=PIM_selec,
    mode_indices=torch.arange(N, device=device),
    centers_xy=centers_xy_target,
    coeff = Para_val_all,
)
val_set = TensorDataset(E_val_all)
val_para = TensorDataset(Para_val_all) #系数验证集
E_val_target = TensorDataset(E_target_val) #验证集
#%% 定义相位面训练器
SP = True #单一球面相位
d0, d, dmax = ut.compute_d_bounds(Lambda, dx, grid_points, 2, 10)
d = 1.0*d # 层与层之间的传播距离
f = 0.2
front = 0 #前端衍射层数量
if SP:
    Point_num = 1 #单球面相位
else:
    Point_num = centers_xy_target.shape[0]

modulator_SP = ONT.PhaseModulator(grid_points, grid_points, Layer_num, Point_num, d,
                               d_min=d0, d_max=dmax, f1=f, f2 = f, sigma=0.6,
                               learnable_amplitude=False,
                               phase_mapping=PHASE_MAPPING).to(device) #仅训练相位
#%% 模型训练或读取
if Eval:
    if not SP_MODEL_PATH.exists():
        raise FileNotFoundError(f"Pretrained spherical-phase model not found: {SP_MODEL_PATH}")
    modulator_SP.load_state_dict(torch.load(SP_MODEL_PATH, map_location=device))
    modulator_SP.eval()
    trainer_SP = ONT.Modal_array_trainer_comparison(
        modulator_SP,
        PIM_selec,
        1,
        X,
        Y,
        centers_xy_target,
        front,
        Lambda,
        lr=0.01,
        batch_size=16,
        device=device,
        mask=None,
        SP=SP
    )
    trainer_SP.estimate_cell_shape_px()
else:
    for j in range(1):
        for i in range(1):
            trainer_SP = ONT.Modal_array_trainer_comparison(
                modulator_SP,              # 相位调制器模型（如 PhaseModulator）
                PIM_selec,                    # PIM模式基底 ∈ ℂ^{HW × N_modes}
                1,           # 目标模式索引
                X,                     # 空间坐标网格 ∈ ℝ^{H × W}
                Y,                     # 空间坐标网格 ∈ ℝ^{H × W}
                centers_xy_target,
                front,
                Lambda,                # 光场波段（单位：mm）
                lr=0.01*(10**(-i)),               # 学习率
                batch_size=16,          # batch 大小
                device=device,          # 计算设备
                mask=None,                # 掩模板
                SP = SP
            )
            cell_shape_px = trainer_SP.estimate_cell_shape_px() #估计微透镜单元的像素尺寸
            trainer_SP.train_loop(E_tra_target, E_val_target, tra_set, val_set, tra_para, val_para, epochs=140)
            trainer_SP.plot_training_curve() #画出训练及验证曲线
            #=== 训练结束，拿到训练后的模型 ===
            modulator_SP = trainer_SP.modulator
#%% 定义相位面训练器
SP = False #微透镜相位
d0, d, dmax = ut.compute_d_bounds(Lambda, dx, grid_points, 2, 10)
d = 1.0*d # 层与层之间的传播距离
f = 0.2
front = 0 #前端衍射层数量
if SP:
    Point_num = 1 #单球面相位
else:
    Point_num = centers_xy_target.shape[0]

modulator_array = ONT.PhaseModulator(grid_points, grid_points, Layer_num, Point_num, d,
                               d_min=d0, d_max=dmax, f1=f, f2 = f, sigma=0.6,
                               learnable_amplitude=False,
                               phase_mapping=PHASE_MAPPING).to(device) #仅训练相位
#%% 模型训练或读取
if Eval:
    if not ARRAY_MODEL_PATH.exists():
        raise FileNotFoundError(f"Pretrained microlens-array model not found: {ARRAY_MODEL_PATH}")
    modulator_array.load_state_dict(torch.load(ARRAY_MODEL_PATH, map_location=device))
    modulator_array.eval()
    trainer_array = ONT.Modal_array_trainer_comparison(
        modulator_array,
        PIM_selec,
        1,
        X,
        Y,
        centers_xy_target,
        front,
        Lambda,
        lr=0.01,
        batch_size=16,
        device=device,
        mask=None,
        SP=SP
    )
    trainer_array.estimate_cell_shape_px()
else:
    for j in range(1):
        for i in range(1):
            trainer_array = ONT.Modal_array_trainer_comparison(
                modulator_array,              # 相位调制器模型（如 PhaseModulator）
                PIM_selec,                    # PIM模式基底 ∈ ℂ^{HW × N_modes}
                1,           # 目标模式索引
                X,                     # 空间坐标网格 ∈ ℝ^{H × W}
                Y,                     # 空间坐标网格 ∈ ℝ^{H × W}
                centers_xy_target,
                front,
                Lambda,                # 光场波段（单位：mm）
                lr=0.01*(10**(-i)),               # 学习率
                batch_size=16,          # batch 大小
                device=device,          # 计算设备
                mask=None,                # 掩模板
                SP = SP
            )
            cell_shape_px = trainer_array.estimate_cell_shape_px() #估计微透镜单元的像素尺寸
            trainer_array.train_loop(E_tra_target, E_val_target, tra_set, val_set, tra_para, val_para, epochs=140)
            trainer_array.plot_training_curve() #画出训练及验证曲线
            #=== 训练结束，拿到训练后的模型 ===
            modulator_array = trainer_array.modulator
#%% === 模型测试 ===
# 重新随机生成独立测试集，不从训练集或验证集中抽样。
test, P_min, P_max = 100, 1, N
E_all_test, Para_all_test,_ = generate_modes_field(test, P_max, PIM_selec, grid_points, device, max_zero_modes=2)
E_test_all = E_all_test.detach()
Para_test_all = Para_all_test.detach()
test_set = TensorDataset(E_test_all)
test_para = TensorDataset(Para_test_all) #系数测试集

E_target_test = torch.zeros_like(E_test_all)
E_out_rec_SP = torch.zeros_like(E_test_all)
E_target_test_array = torch.zeros_like(E_test_all)
E_out_rec_array = torch.zeros_like(E_test_all)
for i in range(len(E_test_all)):
    E_in_test = test_set[i][0] if isinstance(test_set[0], (tuple, list)) else test_set[0]
    E_in_para = test_para[i][0] if isinstance(test_para[0], (tuple, list)) else test_para[0]
    E_all_test_batch = E_in_test.unsqueeze(0).to(device)
    E_in_para = E_in_para.unsqueeze(0).to(device)
    E_modal = (E_in_para[0,1] * PIM_selec[:,1]).reshape(grid_points, grid_points).unsqueeze(0)
    # 前向传播
    E_target_test[i,:,:] = ut.place_PIM_modes(
        detector_shape=(H, W),
        PIM=PIM_selec,
        mode_indices=torch.arange(N, device=device),
        centers_xy=centers_xy_target,
        coeff = E_in_para,
    )
    
    E_out_rec_SP[i,:,:] = ONT.run_inference(
        E_all_test_batch,             # 输入光场: torch.Tensor [1, H, W] (复数)
        modulator=trainer_SP.modulator,             # 训练好的调制器模型
        PIM=trainer_SP.PIM,                   # 模式基底 ∈ ℂ^{HW × N_modes}
        x=trainer_SP.x, y=trainer_SP.y,                  # 网格坐标 ∈ ℝ^{H × W}
        Lambda=trainer_SP.Lambda,              # 波长
        mask=trainer_SP.mask,             # 掩膜  
        trainer = trainer_SP              #训练器    
    )
    
    E_out_rec_array[i,:,:] = ONT.run_inference(
        E_all_test_batch,             # 输入光场: torch.Tensor [1, H, W] (复数)
        modulator=trainer_array.modulator,             # 训练好的调制器模型
        PIM=trainer_array.PIM,                   # 模式基底 ∈ ℂ^{HW × N_modes}
        x=trainer_array.x, y=trainer_array.y,                  # 网格坐标 ∈ ℝ^{H × W}
        Lambda=trainer_array.Lambda,              # 波长
        mask=trainer_array.mask,             # 掩膜  
        trainer = trainer_array              #训练器    
    )

ut.figComplexField(X, Y, E_out_rec_SP[0,:,:], 
                    0, string='E rec SP', cmaps = ['inferno', 'twilight']) #测试数据对应输出
ut.figComplexField(X, Y, E_out_rec_array[0,:,:], 
                    0, string='E rec array', cmaps = ['inferno', 'twilight']) #测试数据对应输出
ut.figComplexField(X, Y, E_target_test[0,:,:], 
                    0, string='E rec tar', cmaps = ['inferno', 'twilight']) #测试数据对应输出
#%% 计算SSIM
import torch.nn.functional as F
def ssim_torch(img1, img2, C1=1e-4, C2=9e-4):
    """
    img1, img2: (H, W) real
    """
    mu1 = img1.mean()
    mu2 = img2.mean()

    sigma1 = ((img1 - mu1) ** 2).mean()
    sigma2 = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    ssim = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    ssim = ssim / ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))

    return ssim
def compute_ssim_coeffs(x,y,cell_shape_px,centers_xy,
    E_out,        # (H, W) or (1, H, W) complex
    modes_ref,   # (H*W, N) complex
    Para         # (N,) complex or real
):
    """
    返回： (1, N_nonzero) 的 SSIM tensor
    """

    device = E_out.device

    # 保证 E_out 是 (1, H, W)
    if E_out.ndim == 2:
        E_out = E_out.unsqueeze(0)

    _, H, W = E_out.shape
    _, N = modes_ref.shape

    h, w = cell_shape_px
    half_h, half_w = h // 2, w // 2

    # 参考模式 reshape
    modes_hw = modes_ref.T.reshape(N, H, W)

    # 找非零系数索引
    nonzero_idx = torch.where(torch.abs(Para) > 0)[0]

    ssim_vals = []

    for idx in nonzero_idx:
        cx, cy = centers_xy[idx]
        cx = int(torch.round(cx).item())
        cy = int(torch.round(cy).item())

        # patch 区域
        y0 = max(cy - half_h, 0)
        y1 = min(cy + half_h, H)
        x0 = max(cx - half_w, 0)
        x1 = min(cx + half_w, W)

        patch_out = E_out[0, y0:y1, x0:x1]
        
        # 参考模式始终居中
        H_ref, W_ref = modes_hw.shape[1:]
        cy_ref = H_ref // 2
        cx_ref = W_ref // 2
        
        y0r = max(cy_ref - half_h, 0)
        y1r = min(cy_ref + half_h, H_ref)
        x0r = max(cx_ref - half_w, 0)
        x1r = min(cx_ref + half_w, W_ref)
        
        patch_ref = modes_hw[idx, y0r:y1r, x0r:x1r]

        # 强度
        I_out = torch.abs(patch_out)**2
        I_ref = torch.abs(patch_ref)**2

        # 归一化（强烈推荐）
        I_out = I_out / (I_out.max() + 1e-8)
        I_ref = I_ref / (I_ref.max() + 1e-8)

        ssim_val = ssim_torch(I_out, I_ref)
        # plt.figure()
        # plt.imshow(I_out.detach().cpu(), cmap='hot')
        # plt.colorbar(label='Intensity')
        # plt.axis('off')
        # plt.show()
        ssim_vals.append(ssim_val)

    ssim_vals = torch.stack(ssim_vals).unsqueeze(0)  # (1, N_nonzero)

    return ssim_vals

ssim_SP = compute_ssim_coeffs(X, Y, trainer_SP.cell_shape_px, trainer_SP.centers_xy,
    E_out_rec_SP[0,:,:],        # (H, W) or (1, H, W) complex
    PIM_selec,   # (H*W, N) complex
    test_para[0][0]         # (N,) complex or real
)
ssim_array = compute_ssim_coeffs(X, Y, trainer_array.cell_shape_px, trainer_array.centers_xy,
    E_out_rec_array[0,:,:],        # (H, W) or (1, H, W) complex
    PIM_selec,   # (H*W, N) complex
    test_para[0][0]         # (N,) complex or real
)
#%% 画SSIM柱状图
# 转 numpy
ssim_sp_np = ssim_SP.detach().cpu().numpy().reshape(-1)
ssim_array_np = ssim_array.detach().cpu().numpy().reshape(-1)

N = len(ssim_sp_np)
x = np.arange(N)
width = 0.35

plt.figure(figsize=(10, 5))

plt.bar(
    x - width/2,
    ssim_sp_np,
    width=width,
    label="SSIM SP"
)

plt.bar(
    x + width/2,
    ssim_array_np,
    width=width,
    label="SSIM Array"
)

plt.xlabel("Mode index")
plt.ylabel("SSIM")
plt.title("SSIM Comparison per Mode")

plt.xticks(x)
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()
#%% 计算输出模式与能量归一化模式重叠积分
B,H,W = E_target_test.shape
Para_rec_SP = ut.compute_overlap(trainer_SP.x , trainer_SP.y,
    torch.abs(E_out_rec_SP),              # (M, H, W) complex or real
    torch.abs(trainer_SP.PIM),          # (N, H, W) complex, energy-normalized
    trainer_SP.centers_xy,         # (N, 2) pixel coords (cx, cy)
    trainer_SP.cell_shape_px,      # (cell_h, cell_w)
).to(torch.cfloat)
Para_rec_array = ut.compute_overlap(trainer_array.x , trainer_array.y,
    torch.abs(E_out_rec_array),              # (M, H, W) complex or real
    torch.abs(trainer_array.PIM),          # (N, H, W) complex, energy-normalized
    trainer_array.centers_xy,         # (N, 2) pixel coords (cx, cy)
    trainer_array.cell_shape_px,      # (cell_h, cell_w)
).to(torch.cfloat)
#%% 画恢复误差柱状图
# ---- 转 numpy（取幅值）----
target_np = torch.abs(Para_test_all[0,:]).detach().cpu().numpy().reshape(-1)
sp_np     = torch.abs(2*Para_rec_SP[0,:]).detach().cpu().numpy().reshape(-1)
array_np  = torch.abs(2*Para_rec_array[0,:]).detach().cpu().numpy().reshape(-1)
# ---- 计算误差（恢复值 - 真值）----
sp_err = sp_np - target_np
array_err = array_np - target_np

# ---- 模式数量 ----
N = len(target_np)
x = np.arange(N)
width = 0.35

plt.figure(figsize=(12, 5))

plt.bar(x - width/2, sp_err,    width=width, label="SP Error")
plt.bar(x + width/2, array_err, width=width, label="Array Error")

plt.axhline(0, linestyle="--")  # 误差零参考线

plt.xlabel("Mode index")
plt.ylabel("Coefficient error (Recovered - Target)")
plt.title("Recovery Error per Mode")

plt.xticks(x)
plt.legend()
plt.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()
#%% 画出系数分布
ut.plot_coeff_comparison(
    Para_test_all[0,:],          # (N,) or (B, N)
    Para_rec_SP[0,:]*2,         # (N,) or (B, N)
    scale_rec=None,    # None / "auto" / float
    title="Coefficient Comparison"
)
#%% 画出相位面
phases_list = []
SP=False
if SP:
    Trainer = trainer_SP
    ONN_phase = Trainer.modulator.map_phase(Trainer.modulator.phases).detach()
    f_len = Trainer.modulator.f.detach()  #微透镜阵列相位各球面波焦距
else:
    Trainer = trainer_array
    ONN_phase = Trainer.modulator.map_phase(Trainer.modulator.phases).detach()
    f_len = Trainer.modulator.f.detach()  #微透镜阵列相位各球面波焦距


for i in range(Layer_num):
    if i==0:
        if SP:
            phase_len = torch.angle((torch.exp(1j*Trainer.k*(Trainer.x**2+Trainer.y**2)/(2*f_len[i]))).unsqueeze(0))
        else:
            phase_len = Trainer.generate_microlens_phase(f_len[i,:])
    else:
        if i==Layer_num-1:
            if SP:
                phase_len = torch.angle((torch.exp(-1j*Trainer.k*(Trainer.x**2+Trainer.y**2)/(2*f_len[i]))).unsqueeze(0))
            else:
                phase_len = -Trainer.generate_microlens_phase(f_len[i,:])
        else:
            phase_len = torch.zeros_like(phase_len)
    phase_tol = ONN_phase[i,:,:] + phase_len
    phases_list.append(phase_tol)  # 存入列表
    ut.figComplexField(trainer_array.x, trainer_array.y, torch.exp(1j*phase_tol[0,:,:]), 
                       0, string=f'Phase_{i}', cmaps = ['jet', 'twilight']) #调制相位
#%% 保存模型(包括球面相位与微透镜阵列相位)

if not Eval:
    torch.save(trainer_SP.modulator.state_dict(), SP_MODEL_PATH)
    torch.save(trainer_array.modulator.state_dict(), ARRAY_MODEL_PATH)
    print("Saved to:", SP_MODEL_PATH)
    print("Saved to:", ARRAY_MODEL_PATH)
else:
    print("Eval mode: loaded pretrained models, skip saving.")
