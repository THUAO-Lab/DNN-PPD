"""
Single-mode modal decomposition/demultiplexing with an optical neural network.

This script trains or loads an ONN phase modulator that extracts one selected
fiber mode from a few-mode input field.
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
def generate_modes_field(tol, num_modes, PIM, resolution, device, target_index=None, zero_prob=0.1):
    """
    生成随机光纤模式叠加场，有概率将指定索引的系数设为0
    
    参数:
        tol: 样本总数
        num_modes: 模式总数
        PIM: 模式基矩阵，形状 [resolution*resolution, num_modes] 或 [num_modes, resolution, resolution]
        resolution: 分辨率
        device: 设备
        target_index: 目标系数索引，如果为None则随机选择一个
        zero_prob: 将目标系数设为0的概率 (默认0.1=10%)
    
    返回:
        batch: 生成的场分布，形状 [tol, resolution, resolution]
        coeffs_batch: 系数矩阵，形状 [tol, num_modes]
        zero_flags: 标记哪些样本的目标系数被设为0，形状 [tol]
    """
    # 初始化
    batch = torch.zeros(tol, resolution, resolution, dtype=torch.complex64, device=device)
    coeffs_batch = torch.zeros(tol, num_modes, dtype=torch.complex64, device=device)
    zero_flags = torch.zeros(tol, dtype=torch.bool, device=device)
    
    # 检查PIM的形状并调整为合适的形状进行矩阵乘法
    if PIM.dim() == 3:
        # PIM形状为 [num_modes, resolution, resolution]
        # 我们需要将其转换为 [resolution*resolution, num_modes]
        PIM_flat = PIM.view(num_modes, -1).T  # 转置后形状: [H*W, num_modes]
    elif PIM.dim() == 2:
        # PIM形状为 [resolution*resolution, num_modes]
        PIM_flat = PIM
    else:
        raise ValueError(f"PIM 必须是2维或3维张量，但得到维度: {PIM.dim()}")
    
    for i in range(tol):
        # 决定目标索引（如果未指定，则随机选择一个）
        if target_index is None:
            current_target_idx = torch.randint(0, num_modes, (1,), device=device).item()
        else:
            current_target_idx = target_index
        
        # 生成随机幅度 (0-5之间)
        amplitudes = 5.0 * torch.rand(num_modes, device=device)
        
        # 生成随机相位 (0到2π之间)
        # phases = 2 * torch.pi * torch.rand(num_modes, device=device)
        
        # 创建复系数: 幅度 * exp(i * 相位)
        coeffs = amplitudes.to(torch.complex64)
        
        # 以zero_prob的概率将目标索引的系数设为0
        if torch.rand(1, device=device).item() < zero_prob:
            coeffs[current_target_idx] = 0.0
            zero_flags[i] = True
        
        # 模式叠加: 使用矩阵乘法
        # PIM_flat形状: [H*W, num_modes], coeffs形状: [num_modes]
        field_flat = PIM_flat @ coeffs.unsqueeze(1)  # 结果形状: [H*W, 1]
        field = field_flat.view(resolution, resolution)  # 重塑为二维
        
        # 存储结果
        batch[i, :, :] = field
        coeffs_batch[i, :] = coeffs
    
    return batch, coeffs_batch, zero_flags
    
def Orthogonal_test(G):
    G = PIM.conj().T @ PIM  # shape [N_modes, N_modes]
    aa = G.detach().numpy()
    return aa

def log_callback(step_or_epoch, loss, is_lbfgs):
    prefix = "LBFGS" if is_lbfgs else "Adam"
    print(f"[{prefix} Step {step_or_epoch:03d}] Loss = {loss:.4e}")
#%% 划分网格
device = "cuda" if torch.cuda.is_available() else "cpu"
grid_points = 512 #每一个模式的采样分布
display_range = 30e-3  # 显示范围±250μm
beam_radius = 60e-3   # 实际光束半径50μm
Eval=False #是否纯推理
learning_rate = 0.01
# 生成坐标网格（覆盖±250μm）
x = torch.linspace(-display_range, display_range, grid_points, 
                   dtype=torch.float32, device = device)
y = torch.linspace(-display_range, display_range, grid_points,
                   dtype=torch.float32, device = device)
X, Y = torch.meshgrid(x, y, indexing='xy')  # 确保坐标方向正确
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
#%% 生成光纤端面网格
x_fiber = torch.linspace(-core_D/2, core_D/2, grid_points, 
                   dtype=torch.float32, device = device)
y_fiber = torch.linspace(-core_D/2, core_D/2, grid_points,
                   dtype=torch.float32, device = device)
X_fiber, Y_fiber = torch.meshgrid(x_fiber, y_fiber, indexing='xy')  # 确保坐标方向正确
r = torch.sqrt(X**2 + Y**2)            # 计算径向距离
mask = ((r <= core_D))     # 生成二值掩膜
#%% 将模式空间分布插值到入射光场空间分布(小网格-大网格)
PIM_beam = torch.zeros_like(PIM_clean)
for i in range(clean_mode):
    mode_single = PIM_clean[:,i].reshape(grid_points, grid_points)
    PIM_beam[:,i] = fib.interpolate_mode_to_large_grid(mode_single, X_fiber, Y_fiber, X, Y).flatten().T
    
# PIM_beam矩阵能量归一化
PIM_beam, energy_integral, energy_normalized = ut.normalize_energy(PIM_beam, X, Y)
#%% 选取所需的模式分布
Index = 1
E_target = PIM_beam[:,Index].reshape(grid_points, grid_points).detach()
mask_fiber = ((r <= core_D/2))     # 生成光纤端面大小的二值掩膜
#%% 生成训练集
tol, P_min, P_max = 1000, 1, 8
PIM_selec = PIM_beam[:,:8] #PIM_beam做正交检验对角线不是1，而是一个小数，可能需要放缩
# 生成随机幅度
E_all, Para_all, zero_flags = generate_modes_field(tol, P_max, PIM_selec, grid_points, device, Index)
E_all = E_all.detach()
Para_all = Para_all.detach()
tra_set = TensorDataset(E_all)
tra_para = TensorDataset(Para_all) #系数训练集
ut.figComplexField(X, Y, E_all[10,:,:], 
                   0, string=f'E_{i}', cmaps = ['jet', 'gray']) #训练样本
#%% 正交检验测试
dx_t = (X[0,1]-X[0,0]).detach()
dy_t = (Y[1,0]-Y[0,0]).detach()
E_test=E_all[0,:,:]
Para_value = Para_all[0,:]
Single = Para_value[1]*PIM_selec[:, 1].reshape(grid_points, grid_points)
overlap = torch.sum((PIM_selec[:, 1].conj().squeeze()) * (E_test.reshape(-1, 1).squeeze())*dx_t*dy_t)
overlap_Single = torch.sum((PIM_selec[:, 1].conj().squeeze()) * (Single.reshape(-1, 1).squeeze())*dx_t*dy_t)
#测试结果：在算重叠积分的时候需要乘上dx*dy
#%% 生成验证集
val, P_min, P_max = 200, 1, 8
PIM_selec = PIM_beam[:,:8]
# 生成随机幅度
E_all_val, Para_all_val,_ = generate_modes_field(val, P_max, PIM_selec, grid_points, device)
E_val_all = E_all_val.detach()
Para_val_all = Para_all_val.detach()
val_set = TensorDataset(E_val_all)
val_para = TensorDataset(Para_val_all) #系数验证集
#%% 定义相位面训练器
Layer_num = 4 #衍射面的层数
d0, d, dmax = ut.compute_d_bounds(Lambda, dx, grid_points, 2, 10)
# d = 0.5*d0 # 层与层之间的传播距离
d = 1.6*d # 层与层之间的传播距离
f = 5.2*d
front = 0 #前端衍射层数量
Point_num = 1 #球面相位阵列数量
modulator = ONT.PhaseModulator(grid_points, grid_points, Layer_num, Point_num, d,
                               d_min=d0, d_max=dmax, f1=f, f2 = f, sigma=0.6,
                               learnable_amplitude=False,
                               phase_mapping="clamp").to(device) #仅训练相位
#%%模型训练或读取
if Eval:
    #模型读取
    modulator.to(device)
    # ===== 加载已保存的模型 =====
    save_path = PRETRAINED_DIR / "modulator_Simulation_Singlemode_4layer.pth"
    modulator.load_state_dict(torch.load(save_path, map_location=device))
    # ===== 推理模式 =====
    modulator.eval()  # 只做前向传播，不训练，不计算梯度
    trainer = ONT.Modal_Trainer(
        modulator,              # 相位调制器模型（如 PhaseModulator）
        PIM_selec,                    # PIM模式基底 ∈ ℂ^{HW × N_modes}
        Index,           # 目标模式索引
        X,                     # 空间坐标网格 ∈ ℝ^{H × W}
        Y,                     # 空间坐标网格 ∈ ℝ^{H × W}
        front,
        Lambda,                # 光场波段（单位：mm）
        lr=learning_rate,               # 学习率
        batch_size=16,          # batch 大小
        device=device,          # 计算设备
        mask=mask                # 掩模板
    )
else:
    #重新训练
    for j in range(1):
        for i in range(2):
            trainer = ONT.Modal_Trainer(
                modulator,              # 相位调制器模型（如 PhaseModulator）
                PIM_selec,                    # PIM模式基底 ∈ ℂ^{HW × N_modes}
                Index,           # 目标模式索引
                X,                     # 空间坐标网格 ∈ ℝ^{H × W}
                Y,                     # 空间坐标网格 ∈ ℝ^{H × W}
                front,
                Lambda,                # 光场波段（单位：mm）
                lr=learning_rate * (10**(-i)),               # 学习率
                batch_size=16,          # batch 大小
                device=device,          # 计算设备
                mask=mask                # 掩模板
            )
    
            trainer.train_loop(tra_set, val_set, tra_para, val_para, epochs=120)
            trainer.plot_training_curve() #画出训练及验证曲线
            #=== 训练结束，拿到训练后的模型 ===
            modulator = trainer.modulator
        
#%% === 模型测试 ===
# 获取 train_set 中的某个样本分布
E_in_test = tra_set[1][0] if isinstance(tra_set[0], (tuple, list)) else tra_set[0]
E_in_para = tra_para[1][0] if isinstance(tra_para[0], (tuple, list)) else tra_para[0]
E_all_test = E_in_test.unsqueeze(0).to(device)
E_in_para = E_in_para.to(device)
E_modal = (E_in_para[Index] * PIM_selec[:,Index]).reshape(grid_points, grid_points)
# 前向传播

E_rec = ONT.run_inference(
    E_all_test,             # 输入光场: torch.Tensor [1, H, W] (复数)
    modulator=trainer.modulator,             # 训练好的调制器模型
    PIM=trainer.PIM,                   # 模式基底 ∈ ℂ^{HW × N_modes}
    x=trainer.x, y=trainer.y,                  # 网格坐标 ∈ ℝ^{H × W}
    Lambda=trainer.Lambda,              # 波长
    mask=trainer.mask,             # 掩膜      
    trainer = trainer
)
E_rec_mask = torch.abs(E_rec) * torch.exp(1j*torch.angle(E_rec)*mask_fiber)
ut.figComplexField(X, Y, E_rec_mask, 
                   0, string='E test out', cmaps = ['jet', 'gray']) #测试数据对应输出
ut.figComplexField(X, Y, E_modal, 
                   0, string='E original modal distribution', cmaps = ['jet', 'gray']) #原始数据模式分量
# 计算一下重叠积分
Original = E_in_para[Index]*PIM_selec[:, Index].reshape(grid_points, grid_points)
# Rec = E_rec.squeeze(0)*torch.exp(1j*torch.angle(Original)) #恢复的光场
overlap_Original = torch.sum((PIM_selec[:, Index].conj().squeeze()) * (Original.reshape(-1, 1).squeeze())*dx_t*dy_t)
overlap_Rec = 2*torch.sum((torch.abs(PIM_selec[:, Index]).conj().squeeze()) * (torch.abs(E_rec_mask).reshape(-1, 1).squeeze())*dx_t*dy_t)
#%% 当输入光场中不存在该模式时输出分布
E_zeros = torch.zeros(1, grid_points, grid_points, dtype = torch.cfloat, device=device)
coeffs_zeros = torch.zeros(1, P_max, dtype = torch.cfloat, device=device)

# 生成随机幅度
amplitudes = 5.0*torch.rand(P_max, device=device)  # 幅度在0-5之间
amplitudes[Index] = 1.0
# 创建复系数
coeffs = amplitudes.to(torch.complex64)
field = torch.zeros(grid_points, grid_points, dtype=torch.complex64, device=device)
field = (PIM_selec @ coeffs.unsqueeze(1)).reshape(grid_points, grid_points)
E_zeros[0,:,:] = field
coeffs_zeros[0,:] = coeffs

ut.figComplexField(X, Y, E_zeros[0,:,:], 
                   0, string='E test out', cmaps = ['inferno', 'twilight']) #画出场分布
E_zeros_rec = ONT.run_inference(
    E_zeros,             # 输入光场: torch.Tensor [1, H, W] (复数)
    modulator=trainer.modulator,             # 训练好的调制器模型
    PIM=trainer.PIM,                   # 模式基底 ∈ ℂ^{HW × N_modes}
    x=trainer.x, y=trainer.y,                  # 网格坐标 ∈ ℝ^{H × W}
    Lambda=trainer.Lambda,              # 波长
    mask=trainer.mask,             # 掩膜      
    trainer = trainer
)
ut.figComplexField(X, Y, E_zeros_rec[0,:,:], 
                   0, string='E zeros modal distribution', cmaps = ['inferno', 'twilight']) #0模式对应的传输场
#%% 画误差折线条状图
Num = 1000
batch_size = 20   # 20~50，根据显存调

# ===============================
# 预分配结果容器
# ===============================
Para_rec_list = []
Para_target_list = []

# ===============================
# 分 batch 处理
# ===============================
for start in range(0, Num, batch_size):
    end = min(start + batch_size, Num)
    B = end - start

    # -------- 生成测试场 --------
    E_test, Para_test, zero_flags_test = generate_modes_field(
        B, P_max, PIM_selec, grid_points, device, Index
    )

    E_test = E_test.detach()
    Para_test = Para_test.detach()

    # -------- ONT 推理 --------
    with torch.no_grad():
        E_test_rec = ONT.run_inference(
            E_test,
            modulator=trainer.modulator,
            PIM=trainer.PIM,
            x=trainer.x,
            y=trainer.y,
            Lambda=trainer.Lambda,
            mask=trainer.mask,
            trainer=trainer
        )

    # -------- 输出场 --------
    Eout = torch.abs(E_test_rec) * torch.exp(
        1j * torch.angle(E_test_rec) * trainer.mask
    )   # (B, H, W)

    # -------- 拉平 --------
    Eout_vec = Eout.contiguous().view(B, -1).transpose(0, 1)  # (HW, B)

    # -------- PIM --------
    PIM_vec = trainer.PIM.contiguous().transpose(0, 1).conj()  # (M, HW)

    # -------- 重叠积分 --------
    overlap = torch.matmul(
        torch.abs(PIM_vec),
        torch.abs(Eout_vec) * trainer.dx * trainer.dy
    )  # (M, B)

    # -------- 只取目标模式 --------
    Para_rec_batch = overlap[Index, :].T        # (B, 1)
    Para_target_batch = Para_test[:, Index]     # (B, 1)

    Para_rec_list.append(Para_rec_batch.detach().cpu())
    Para_target_list.append(Para_target_batch.detach().cpu())

    # -------- 显存清理（很重要）--------
    del E_test, Para_test, E_test_rec, Eout, Eout_vec, overlap
    torch.cuda.empty_cache()

# ===============================
# 拼接最终结果
# ===============================
Para_rec_all = torch.cat(Para_rec_list, dim=0)       # (Num, 1)
Para_target_all = torch.cat(Para_target_list, dim=0)/2 # (Num, 1)
error_stats = ut.plot_recovery_error(
    Para_rec_all,
    Para_target_all,
    bins=50,
    title="Recovery Coefficient Error Statistics"
)
#%% 画柱状图
ut.plot_rec_vs_target_bar(
    Para_rec_all,
    Para_target_all,
    num_show=15,
    seed=0,   # 固定种子
    title="Recovered vs Target Modal Coefficients"
)
#%% 画出相位面
phases_list = []
ONN_phase = modulator.map_phase(modulator.phases).detach() #不包含微透镜阵列的衍射相位
f_len = modulator.f.detach()  #微透镜阵列相位各球面波焦距
for i in range(Layer_num):
    if i==0:
        phase_len = torch.angle((torch.exp(1j*trainer.k*(trainer.x**2+trainer.y**2)/(2*f_len[i]))).unsqueeze(0))
    else:
        if i==Layer_num-1:
            phase_len = torch.angle((torch.exp(-1j*trainer.k*(trainer.x**2+trainer.y**2)/(2*f_len[i]))).unsqueeze(0))
        else:
            phase_len = torch.zeros_like(phase_len)
    phase_tol = ONN_phase[i,:,:] + phase_len
    phases_list.append(phase_tol)  # 存入列表
    ut.figComplexField(trainer.x, trainer.y, torch.exp(1j*phase_tol[0,:,:]), 
                       0, string=f'Phase_{i}', cmaps = ['jet', 'twilight']) #调制相位
#%% 保存图片
plt.savefig('Single_mode1_rec.jpg',  # 文件名
            dpi=800,                # 关键：设置分辨率，300是出版常用标准
            bbox_inches='tight',    # 推荐：自动裁剪图像周围的白边
            pil_kwargs={'progressive': True}  # 可选：生成渐进式JPEG
            )
#%% 保存模型
save_path = PRETRAINED_DIR / "modulator_Simulation_Singlemode_4layer.pth"
torch.save(trainer.modulator.state_dict(), save_path)

print("Saved to:", save_path)
