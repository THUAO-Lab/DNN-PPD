"""
Design and train the real-fiber SLM ONN phase pattern.

This script uses the measured SLM geometry and the LP-mode model of the real
fiber to train or load the SLM phase modulator, then exports the full SLM phase
bitmap for the few-mode fiber experiment.

Set Eval=True to load the pretrained fiber phase model and skip training.
Set Eval=False to retrain before testing and exporting.
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
from torch.utils.data import TensorDataset
import ONNtrain as ONT
import matplotlib.pyplot as plt
import gc

PRETRAINED_DIR = PROJECT_DIR / "Pre_trained_DNN"
MODEL_PATH = PRETRAINED_DIR / "modulator_slm_semantice_3layer_paper_result_fibre.pth"

#%% 训练集生成
def insert_center(src, dst):
    """
    将小矩阵 src 放到大矩阵 dst 的中心（支持 batch）

    src: (H, W) 或 (B, H, W)
    dst: (H_big, W_big) 或 (B, H_big, W_big)

    返回:
        与 dst 同 shape（但 dtype 跟 src）
    """

    # -------------------------------------------------
    # dtype 对齐
    # -------------------------------------------------
    dst = dst.clone().to(src.dtype)

    # -------------------------------------------------
    # 统一成 batch 形式
    # -------------------------------------------------
    if src.dim() == 2:
        src = src.unsqueeze(0)   # (1, H, W)

    if dst.dim() == 2:
        dst = dst.unsqueeze(0)   # (1, H, W)

    B_s, H_s, W_s = src.shape
    B_d, H_d, W_d = dst.shape

    # -------------------------------------------------
    # batch 对齐（核心）
    # -------------------------------------------------
    if B_d == 1 and B_s > 1:
        # 大图只有1张 → 扩展
        dst = dst.expand(B_s, H_d, W_d).clone()

    elif B_s == 1 and B_d > 1:
        # 小图只有1张 → broadcast
        src = src.expand(B_d, H_s, W_s)

    elif B_s != B_d:
        raise ValueError("Batch size mismatch between src and dst")

    # -------------------------------------------------
    # 计算中心位置
    # -------------------------------------------------
    h_start = (H_d - H_s) // 2
    w_start = (W_d - W_s) // 2

    # -------------------------------------------------
    # 写入
    # -------------------------------------------------
    dst[:, h_start:h_start+H_s, w_start:w_start+W_s] = src

    return dst
def generate_modes_field(
    tol, X, Y,
    num_modes,
    PIM,
    resolution,
    device,
    current_beam_size,
    max_beam_size,
    X_collimated=None,
    Y_collimated=None,
    aperture_mask=None,
    aperture_diameter=3.0,
    dnn_input_beam_size=None,
    phase_range=0.0 * torch.pi,
    eff_N = 2
):
    """
    生成真实FMF训练集光场：
    1. 在8个模式系数中，只随机生成前两个模式的0-1实数系数；
    2. 用系数乘PIM_selec_in并干涉叠加，得到7.057mm准直窗口下的混合场；
    3. 用3mm光阑mask截取中心光斑；
    4. 将3mm光斑按比例缩放到2mm；
    5. 插值到输入的X/Y计算窗口中，作为DNN入射训练样本。
    """
    if X_collimated is None or Y_collimated is None or aperture_mask is None:
        raise ValueError("需要输入X_collimated、Y_collimated和aperture_mask")
    if dnn_input_beam_size is None:
        dnn_input_beam_size = current_beam_size

    batch = torch.zeros(
        tol, resolution, resolution,
        dtype=torch.complex64, device=device
    )

    coeffs_batch = torch.zeros(
        tol, num_modes,
        dtype=torch.complex64, device=device
    )

    bit_batch = torch.zeros(
        tol, num_modes,
        dtype=torch.float32, device=device
    )

    beam_size_batch = torch.zeros(
        tol,
        dtype=torch.float32, device=device
    )

    symbol_batch = []
    category_batch = []

    # -------- 保证PIM为[num_modes,H,W]，这里PIM被视为7.057mm准直窗口下的模式基底 --------
    if PIM.dim() == 2:
        PIM_3d = PIM.T.reshape(num_modes, resolution, resolution)
    else:
        PIM_3d = PIM

    PIM_flat = PIM_3d.reshape(num_modes, -1).T
    aperture_mask = aperture_mask.to(device=device, dtype=torch.float32)
    aperture_mask_complex = aperture_mask.to(torch.complex64)

    # 让3mm光阑内的信息在插值后变成2mm入射光斑。
    # interpolate_mode_to_large_grid按坐标采样，因此这里把原7.057mm坐标乘以2/3。
    resize_ratio = dnn_input_beam_size / aperture_diameter
    X_collimated_rescaled = X_collimated * resize_ratio
    Y_collimated_rescaled = Y_collimated * resize_ratio

    for i in range(tol):
        # ==============================
        # Step 1: 只随机生成前两个模式系数，后6个模式保持为0
        # ==============================
        bits = torch.zeros(num_modes, device=device)
        coeffs = torch.zeros(num_modes, dtype=torch.complex64, device=device)
        # 在 1 到 eff_N 之间随机选择本次实际分析的模式数
        active_N = torch.randint(1, eff_N + 1, (1,), device=device).item()

        amplitudes = torch.zeros(eff_N, device=device)
        amplitudes[:active_N] = torch.rand(active_N, device=device)
        phases = torch.zeros(eff_N, device=device)
        phases[:active_N] = (torch.rand(active_N, device=device) - 0.5) * 2 * phase_range
        coeffs[:eff_N] = amplitudes.to(torch.complex64) * torch.exp(1j * phases)
        bits[:active_N] = (amplitudes[:active_N] > 0).float()
        beam_size_batch[i] = dnn_input_beam_size

        category = "fmf_two_mode_random"
        symbol = f"{amplitudes[0].item():.3f}-{amplitudes[1].item():.3f}"

        bit_batch[i] = bits
        coeffs_batch[i] = coeffs
        symbol_batch.append(symbol)
        category_batch.append(category)

        # ==============================
        # Step 2: 先在7.057mm准直窗口内混合，再截取3mm并缩放到2mm
        # ==============================
        field_flat = PIM_flat @ coeffs.unsqueeze(1)
        field_collimated = field_flat.view(resolution, resolution)
        field_apertured = field_collimated * aperture_mask_complex
        field_input = fib.interpolate_mode_to_large_grid(
            field_apertured,
            X_collimated_rescaled,
            Y_collimated_rescaled,
            X,
            Y
        ).to(device=device, dtype=torch.complex64)
        batch[i] = field_input

    return batch, coeffs_batch, bit_batch, symbol_batch, category_batch, beam_size_batch
#%% 定义SLM参数
if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
Nx=4096#SLM像素数x
Ny=2160#SLM像素数y
Single_phase_pixel = 900 #实际像素
pixel_sc = 3
dx = 3800e-6*pixel_sc #SLM像素大小x
dy = 3800e-6*pixel_sc #SLM像素大小y
Layer_num = 3 #衍射层
H = 30 #SLM与反射镜之间距离(理想或卡尺测量)
N = 8 #分析的模式数量
point = int(torch.ceil(torch.sqrt(torch.as_tensor(N, dtype=torch.int))).item()) #单向点位
offset = [[1,-342],[1,-370],[-15,-388]] # 每一个相位图的偏移2025.5.10
raw_d_ideal = torch.tensor([60.2238, 60.2238], dtype=torch.float32, device=device) # 理想层间距2025.5.10
grid_points = round(Single_phase_pixel/pixel_sc) #每一个模式的期望采样数
Eval = False
#%% 划分SLM
block = ut.partition_slm(
    Nx, Ny,          # SLM pixel number
    dx/pixel_sc, dy/pixel_sc,          # pixel size (mm)
    Layer_num,        # number of diffraction layers
    N_eff=grid_points*pixel_sc,
    device=device,
)
#%% 定义原始参数并计算缩放倍率
display_range = 100e-3  # 原始网络显示范围半径
N_eff = block[0]["Nx_eff"] #SLM中的有效像素
R_slm = (N_eff/pixel_sc * dx) / 2    # 实际对应SLM区域的范围(mm)
scale_xy = R_slm / display_range #缩放倍率
#%% 相位图有效网格
x = torch.linspace(-R_slm, R_slm, grid_points, 
                   dtype=torch.float32, device = device)
y = torch.linspace(-R_slm, R_slm, grid_points,
                   dtype=torch.float32, device = device)
X, Y = torch.meshgrid(x, y, indexing='xy')  # 确保坐标方向正确
#%% 定义光纤
core_D=16e-3
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
del field_distributions, PIM
#%% 定义端面网格
#缩放端面光场(实验中对应4f)
core_scale = core_D * 23 #2.3是一个参数，自己控制区域大小
core_input = 2.0 #入射斑大小
max_beam_size = 2.0# 最大光斑 
collimated_beam_size = 7.057 #准直系统输出光斑直径，单位mm
aperture_diameter = 3.0 #光阑截取直径，单位mm
x_fiber = torch.linspace(-core_scale/2, core_scale/2, grid_points, 
                   dtype=torch.float32, device = device)
y_fiber = torch.linspace(-core_scale/2, core_scale/2, grid_points,
                   dtype=torch.float32, device = device)
X_fiber, Y_fiber = torch.meshgrid(x_fiber, y_fiber, indexing='xy')  # 确保坐标方向正确
x_collimated = torch.linspace(-collimated_beam_size/2, collimated_beam_size/2, grid_points,
                   dtype=torch.float32, device = device)
y_collimated = torch.linspace(-collimated_beam_size/2, collimated_beam_size/2, grid_points,
                   dtype=torch.float32, device = device)
X_collimated, Y_collimated = torch.meshgrid(x_collimated, y_collimated, indexing='xy')  # PIM_selec_in所在的准直光斑窗口
aperture_mask = ((X_collimated**2 + Y_collimated**2) <= (aperture_diameter/2)**2).float()
x_in = torch.linspace(-core_input/2, core_input/2, grid_points, 
                   dtype=torch.float32, device = device)
y_in = torch.linspace(-core_input/2, core_input/2, grid_points,
                   dtype=torch.float32, device = device)
X_in, Y_in = torch.meshgrid(x_in, y_in, indexing='xy')  # 确保坐标方向正确
#%% 通过偏移像素计算每一层的光程变化
offset_tensor = torch.tensor(offset, device=device, dtype=torch.float32)
D_ideal = ((torch.tensor(block[1]['center_pixel'], device=device) - \
       torch.tensor(block[0]['center_pixel'], device=device))*dx/pixel_sc)[0]
offset_dist = torch.sqrt(
    (offset_tensor[1:, 0] * dx / pixel_sc)**2 +
    ((offset_tensor[1:, 1] - offset_tensor[0, 1]) * dy / pixel_sc)**2
)+D_ideal
H_tensor = torch.full_like(offset_dist, H)
L = torch.sqrt(H_tensor**2 + offset_dist**2)
raw_d = raw_d_ideal / 2 + L
#%% 将模式空间分布插值到入射光场空间分布(小网格-大网格)
PIM_selec = torch.zeros_like(PIM_clean[:,:N])
PIM_selec_in = torch.zeros_like(PIM_clean[:,:N])
for i in range(N):
    mode_single = PIM_clean[:,i].reshape(grid_points, grid_points)
    PIM_selec[:,i] = fib.interpolate_mode_to_large_grid(mode_single, X_fiber, Y_fiber, X, Y).flatten().T
    PIM_selec_in[:,i] = mode_single.flatten().T
    
# PIM_beam矩阵能量归一化
PIM_selec, energy_integral, energy_normalized = ut.normalize_energy(PIM_selec, X, Y)
PIM_selec_in, energy_integral, energy_normalized = ut.normalize_energy(PIM_selec_in, X, Y)
del PIM_clean
gc.collect()              # Python 垃圾回收
torch.cuda.empty_cache()  # 释放GPU缓存
#%% 大范围采样XY网格
Para = 1.5
x_big = torch.arange(-Para*R_slm, Para*R_slm, dx, device=device)
y_big = torch.arange(-Para*R_slm, Para*R_slm, dy, device=device)
X_big, Y_big = torch.meshgrid(x_big, y_big, indexing='xy')
H_big, W_big = X_big.shape
mask = ((torch.abs(X_big) <= R_slm) & (torch.abs(Y_big) <= R_slm)).float()
H,W= X.shape #训练的点数
#%% 计算点位
# 语义传播包含数字、字母以及汉字，最多16位
H,W = X.shape
full_range=(-R_slm, R_slm, -R_slm, R_slm)
phys_range=(-R_slm, R_slm, -R_slm, R_slm)
# 微透镜阵列点位
centers_xy_target = ut.generate_grid_centers(
    I=X,
    n_row=point,
    n_col=point,
    phys_range = phys_range,
    full_range = full_range
)
# 光场点位
full_range=(-Para*R_slm, Para*R_slm, -Para*R_slm, Para*R_slm)
phys_range=(-Para*R_slm, Para*R_slm, -Para*R_slm, Para*R_slm)
# 微透镜阵列点位
centers_xy_target_field = ut.generate_grid_centers(
    I=X_big,
    n_row=point,
    n_col=point,
    phys_range = phys_range,
    full_range = full_range
)
#%% 加载训练的网络
d0, d, dmax = ut.compute_d_bounds(Lambda, dx, grid_points, 2, 10)
front = 0 #前端衍射层数量
Point_num = centers_xy_target.shape[0]
d = raw_d.detach().cpu() # 层与层之间的传播距离

f1 = torch.tensor(282.0218+100, dtype=torch.float)+11.2*d[0]
f2 = torch.tensor(397.5858, dtype=torch.float)+1.2*d[0] #出射距离
modulator_1024 = ONT.PhaseModulator(H_big, W_big, Layer_num, Point_num, d,
                               d_min=d0, d_max=dmax, f1=f1, f2 = f2, sigma=0.6,
                               learnable_amplitude=False, freeze_prop=True).to(device) #仅训练相位
with torch.no_grad():
    modulator_1024.raw_d.copy_(torch.tensor(raw_d, dtype=torch.float32, device=device))

#%% 训练主体
#% 生成验证集
val, P_min, P_max = 200, 1, N
# 生成随机幅度
E_all_val, Para_all_val, _, _, _, beam_size_batch = generate_modes_field(val, X, Y, P_max, PIM_selec_in, grid_points, device,
                                                        core_input, max_beam_size,
                                                        X_collimated=X_collimated,
                                                        Y_collimated=Y_collimated,
                                                        aperture_mask=aperture_mask,
                                                        aperture_diameter=aperture_diameter,
                                                        dnn_input_beam_size=core_input, eff_N = 4)
# 光场置入大计算窗口
Value_big = torch.zeros([val, H_big, W_big], dtype=E_all_val.dtype, device = E_all_val.device)
E_all_val_big = insert_center(E_all_val, Value_big)
E_all_val_big = E_all_val_big.detach()
Para_val_all = Para_all_val.detach()

Value_PIM = torch.zeros([N, H_big, W_big], dtype=E_all_val.dtype, device = E_all_val.device)
PIM_big = PIM_selec_in.T.reshape(N, grid_points, grid_points)  # (Hs, Ws, N)
PIM_big = insert_center(PIM_big, Value_PIM)
PIM_big = PIM_big.reshape(N, -1).T  # (H_b*W_b, N)

PIM_selec_big = PIM_selec.T.reshape(N, grid_points, grid_points)  # (Hs, Ws, N)
PIM_selec_big = insert_center(PIM_selec_big, Value_PIM)
PIM_selec_big = PIM_selec_big.reshape(N, -1).T  # (H_b*W_b, N)

E_target_val = ut.place_PIM_modes(
    detector_shape=(H_big, W_big),
    PIM=PIM_selec_big,
    mode_indices=torch.arange(N, device=device),
    centers_xy=centers_xy_target_field,
    coeff = Para_val_all,
)
E_target_val_big = insert_center(E_target_val, Value_big)
val_set = TensorDataset(E_all_val_big)
val_para = TensorDataset(Para_val_all) #系数验证集
E_val_target = TensorDataset(E_target_val_big) #验证集

del Value_PIM, Value_big
gc.collect()              # Python 垃圾回收
torch.cuda.empty_cache()  # 释放GPU缓存

#% 定义损失函数值提取因子
epochs = 120
loss_data = torch.zeros(4,epochs, dtype = torch.float, device = device)
def build_trainer(lr):
    return ONT.Modal_array_fiber_trainer(
        modulator_1024,
        PIM_selec_big,
        1,
        X_big,
        Y_big,
        grid_points, grid_points,
        centers_xy_target,
        centers_xy_target_field,
        front,
        Lambda,
        lr=lr,
        batch_size=16,
        device=device,
        mask=mask,
    )

#% 网络训练或读取
if Eval:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Pretrained fiber SLM phase model not found: {MODEL_PATH}")
    modulator_1024.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    with torch.no_grad():
        modulator_1024.raw_d.copy_(torch.tensor(raw_d, dtype=torch.float32, device=device))
    modulator_1024.eval()
    trainer = build_trainer(lr=0.01)
    cell_shape_px = trainer.estimate_cell_shape_px()
    cell_shape_px_big = trainer.estimate_cell_shape_px_big()
    trainer.init_gaussian_kernel(sigma=1, kernel_size=2)
else:
    for n in range(1):
        tol, P_min, P_max = 1000, 1, N
        E_all, Para_all, _, _, _,_ = generate_modes_field(tol, X, Y, P_max, PIM_selec_in, grid_points, device, 
                                                          core_input, max_beam_size,
                                                          X_collimated=X_collimated,
                                                          Y_collimated=Y_collimated,
                                                          aperture_mask=aperture_mask,
                                                          aperture_diameter=aperture_diameter,
                                                          dnn_input_beam_size=core_input, eff_N=4)
        Value_big = torch.zeros([tol, H_big, W_big], dtype=E_all_val.dtype, device=E_all_val.device)
        E_all_big = insert_center(E_all, Value_big).detach()
        Para_all = Para_all.detach()
        
        E_target_tra = ut.place_PIM_modes(
            detector_shape=(H_big, W_big),
            PIM=PIM_selec_big,
            mode_indices=torch.arange(N, device=device),
            centers_xy=centers_xy_target_field,
            coeff=Para_all,
        )
        E_target_tra_big = insert_center(E_target_tra, Value_big)
        tra_set = TensorDataset(E_all_big)
        tra_para = TensorDataset(Para_all) 
        E_tra_target = TensorDataset(E_target_tra_big)

        for j in range(1):
            for i in range(1):
                trainer = build_trainer(lr=0.01 * (10**(-i)))
                cell_shape_px = trainer.estimate_cell_shape_px()
                cell_shape_px_big = trainer.estimate_cell_shape_px_big()
                trainer.init_gaussian_kernel(sigma=1, kernel_size=2)
                trainer.train_loop(E_tra_target, E_val_target, tra_set, val_set, tra_para, val_para, epochs=epochs)
                trainer.plot_training_curve()
                loss_data[0, :] = torch.tensor(trainer.train_losses, device=device)
                modulator_1024 = trainer.modulator
#%% 模型测试
cell_shape_px = trainer.estimate_cell_shape_px()
E_target_test_big = torch.zeros_like(E_target_val_big)
E_out_rec = torch.zeros_like(E_target_val_big)
for i in range(len(E_all_val)):
    Value_big = torch.zeros([1, H_big, W_big], dtype=E_all_val.dtype, device = E_all_val.device)
    E_in_test = val_set[i][0] if isinstance(val_set[0], (tuple, list)) else val_set[0]
    E_in_para = val_para[i][0] if isinstance(val_para[0], (tuple, list)) else val_para[0]
    E_all_test = E_in_test.unsqueeze(0).to(device)
    E_in_para = E_in_para.unsqueeze(0).to(device)
    E_all_test = insert_center(E_all_test, Value_big)
    # 前向传播
    E_target_val = ut.place_PIM_modes(
        detector_shape=(H_big, W_big),
        PIM=PIM_selec_big,
        mode_indices=torch.arange(N, device=device),
        centers_xy=centers_xy_target_field,
        coeff = E_in_para,
    )
    
    E_target_test_big[i,:,:] = insert_center(E_target_val, Value_big)
    
    E_out_rec[i,:,:] = ONT.run_inference(
        E_all_test,             # 输入光场: torch.Tensor [1, H, W] (复数)
        modulator=trainer.modulator,             # 训练好的调制器模型
        PIM=trainer.PIM,                   # 模式基底 ∈ ℂ^{HW × N_modes}
        x=trainer.x, y=trainer.y,                  # 网格坐标 ∈ ℝ^{H × W}
        Lambda=trainer.Lambda,              # 波长
        mask=trainer.mask,             # 掩膜  
        trainer = trainer              #训练器    
    )

ut.figComplexField(X_big, Y_big, E_out_rec[0,:,:], 
                   0, string='E test out', cmaps = ['inferno', 'hsv']) #测试数据对应输出
ut.figComplexField(X_big, Y_big, E_target_val_big[0,:,:], 
                   0, string='E original modal distribution', cmaps = ['inferno', 'hsv']) #原始数据模式分量
#%% 获取相位面
phases_list = []
modulator_1024.eval() #验证模式

ONN_complex,_ =  modulator_1024(mask) #不包含微透镜阵列的复振幅
f_len_array = modulator_1024.f.detach()  #微透镜阵列相位各球面波焦距
for i in range(Layer_num):
    if i==0:
        complex_len_array = torch.exp(1j*trainer.generate_microlens_phase(f_len_array[i,:]))
    else:
        if i==Layer_num-1:
            complex_len_array = torch.exp(1j*-trainer.generate_microlens_phase(f_len_array[i,:]))
        else:
            complex_len_array = torch.exp(1j*torch.zeros_like(complex_len_array))
    phase_tol = torch.angle(ONN_complex[i].to(device).reshape(H_big, W_big) * complex_len_array)
    phases_list.append(phase_tol)  # 存入列表
    # ut.figComplexField(X_big, Y_big, complex_len_array[0,:,:], 
    #                    0, string=f'Phase_{i}', cmaps = ['jet', 'twilight']) #调制相位
#%% 截取中间相位
def upsample_phase(phase, superpixel_size):
    """
    超像素相位放大（像素复制，不插值）
    
    参数
    ----------
    phase : Tensor
        (H,W) 或 (1,H,W)
    superpixel_size : int
        每个低分辨率像素对应的高分辨率像素数
        例如：
            2 → 2×2 像素块
            4 → 4×4 像素块
    
    返回
    ----------
    phase_up : Tensor
        放大后的相位图
    """
    
    squeeze_flag = False
    
    # -------- 统一维度 --------
    if phase.dim() == 2:
        phase = phase.unsqueeze(0)   # (1,H,W)
        squeeze_flag = True
    elif phase.dim() == 3:
        pass
    else:
        raise ValueError("phase must be (H,W) or (1,H,W)")
    
    # -------- 超像素复制 --------
    # (1,H,W) → (1,H*k,W*k)
    phase_up = phase.repeat_interleave(superpixel_size, dim=-2) \
                    .repeat_interleave(superpixel_size, dim=-1)
    
    # -------- 恢复维度 --------
    if squeeze_flag:
        phase_up = phase_up.squeeze(0)
    
    return phase_up
#重新采样XY
x = torch.linspace(-R_slm, R_slm, grid_points*pixel_sc, 
                   dtype=torch.float32, device = device)
y = torch.linspace(-R_slm, R_slm, grid_points*pixel_sc,
                   dtype=torch.float32, device = device)
X_up, Y_up = torch.meshgrid(x, y, indexing='xy')  # 确保坐标方向正确
phase_effect = []
for i in range(Layer_num):
    phase_small = trainer.get_output_compex(phases_list[i], mask)
    #插值
    phase_up = upsample_phase(phase_small, superpixel_size=pixel_sc)
    phase_effect.append(phase_up)  # 存入列表
    # ut.figComplexField(X_up, Y_up, torch.exp(1j*phase_effect[i][0,:,:]), 
    #                    0, string=f'Phase_{i}', cmaps = ['jet', 'twilight']) #调制相位
if Layer_num == 2:
    phase_effect.append(torch.zeros_like(phase_effect[0]))
#%% 把相位在SLM上平铺(方案1)
slm_phase = ut.assemble_slm(block, phase_effect, SLM_Nx=Nx, SLM_Ny=Ny,tilt_angle=[0.0, 0.0], 
                            offsets=offset, wavelength=Lambda,dx = dx/pixel_sc, dy = dy/pixel_sc, device = device,
                            show_plot=True, cmap='twilight',
                            background_blaze=True,          
                            background_tilt=(-30, 0))
# slm_flipped = torch.flip(slm_phase, dims=[0])
E_m = torch.exp(1j*slm_phase)
#%% 保存相位图
ut.save_phase_as_bmp(E_m, filename="modulator_slm_FMF.bmp")
#%% 保存模型
if not Eval:
    torch.save(trainer.modulator.state_dict(), MODEL_PATH)
    print("Saved to:", MODEL_PATH)
else:
    print("Eval mode: loaded pretrained model, skip saving.")
