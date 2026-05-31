import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math


def compute_field_energy(E_field: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """
    计算二维复振幅场的能量（强度积分）
    
    参数:
    --------
    E_field: torch.Tensor
        复数张量，shape 为 [H, W]，表示复振幅场
    dx: float
        x方向的采样间隔（单位：米）
    dy: float
        y方向的采样间隔（单位：米）

    返回:
    --------
    energy: torch.Tensor
        标量，表示总能量（功率）
    """
    intensity = torch.abs(E_field) ** 2   # [H, W] 实数张量
    area_element = dx * dy
    energy = torch.sum(intensity* area_element) 
    return energy

def compute_d_bounds(
    wavelength_mm,   # λ, mm
    dx_mm,           # pixel size, mm
    N_pixel,         # number of pixels per dimension
    min_factor=2.0,  # 2~5 recommended
    max_factor=50.0  # 20~100 recommended
):
    """
    Compute physical and trainable bounds for diffraction distance z.

    Returns
    -------
    z_nyquist : float
        Absolute physical lower bound (mm)
    z_min_train : float
        Recommended lower bound for training (mm)
    z_max_train : float
        Recommended upper bound for training (mm)
    """

    # Nyquist lower bound
    z_nyquist = (N_pixel * dx_mm**2) / wavelength_mm

    # Effective training bounds
    z_min_train = min_factor * z_nyquist
    z_max_train = max_factor * z_nyquist

    return z_nyquist, z_min_train, z_max_train
def magnify_field(E_in, x, y, M):
    """
    4f 成像系统下的光场放大（物理正确）

    Parameters
    ----------
    E_in : (H, W) complex tensor
        光纤端面复振幅
    x, y : (H, W) tensor
        原物理坐标 (mm)
    M : float
        放大倍率 (f2 / f1)

    Returns
    -------
    E_out : (H, W) complex tensor
        放大后的光场
    x_out, y_out : (H, W)
        放大后的物理坐标
    """

    H, W = E_in.shape

    # 放大后的物理坐标
    x_out = M * x
    y_out = M * y

    # === 坐标反映射 ===
    # E_out(x') = 1/M * E_in(x'/M)
    x_in = x_out / M
    y_in = y_out / M

    # 将物理坐标映射回像素索引
    dx = x[0,1] - x[0,0]
    dy = y[1,0] - y[0,0]

    ix = (x_in - x[0,0]) / dx
    iy = (y_in - y[0,0]) / dy

    # grid_sample 需要 [-1, 1]
    ix_norm = 2 * ix / (W - 1) - 1
    iy_norm = 2 * iy / (H - 1) - 1

    grid = torch.stack((ix_norm, iy_norm), dim=-1).unsqueeze(0)

    E_interp = torch.nn.functional.grid_sample(
        E_in.unsqueeze(0).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True
    )[0,0]

    # 振幅缩放
    E_out = E_interp / M

    return E_out, x_out, y_out


def plot_coeff_comparison(
    coeff_gt,          # (N,) or (B, N)
    coeff_rec,         # (N,) or (B, N)
    mode_indices=None, # 可选: 模式编号
    scale_rec=None,    # None / "auto" / float
    title="Coefficient Comparison"
):
    """
    对比原始系数与恢复系数（柱状图）

    Parameters
    ----------
    coeff_gt : torch.Tensor or np.ndarray
        原始（理想）系数
    coeff_rec : torch.Tensor or np.ndarray
        恢复系数
    mode_indices : array-like, optional
        模式编号（x 轴标签）
    scale_rec : None | "auto" | float
        - None      : 不缩放
        - "auto"    : 最小二乘拟合一个比例因子
        - float     : 手动指定缩放因子（如 2.0）
    title : str
        图标题
    """

    # -------- 转 numpy --------
    if torch.is_tensor(coeff_gt):
        coeff_gt = coeff_gt.detach().cpu().numpy()
    if torch.is_tensor(coeff_rec):
        coeff_rec = coeff_rec.detach().cpu().numpy()

    coeff_gt = np.asarray(coeff_gt).squeeze()
    coeff_rec = np.asarray(coeff_rec).squeeze()

    assert coeff_gt.shape == coeff_rec.shape
    N = coeff_gt.size

    # -------- 缩放恢复系数 --------
    if scale_rec == "auto":
        # 最小二乘意义下拟合比例
        scale = np.dot(coeff_gt, coeff_rec) / (np.dot(coeff_rec, coeff_rec) + 1e-12)
        coeff_rec_plot = scale * coeff_rec
        scale_info = f"(auto scale = {scale:.3f})"
    elif isinstance(scale_rec, (int, float)):
        coeff_rec_plot = scale_rec * coeff_rec
        scale_info = f"(scale = {scale_rec:.3f})"
    else:
        coeff_rec_plot = coeff_rec
        scale_info = ""

    # -------- x 轴 --------
    x = np.arange(N)
    if mode_indices is None:
        mode_indices = x

    width = 0.35

    # -------- 画图 --------
    plt.figure(figsize=(10, 4))
    plt.bar(x - width/2, coeff_gt, width, label="Ground Truth", alpha=0.8)
    plt.bar(x + width/2, coeff_rec_plot, width, label="Recovered", alpha=0.8)

    plt.xticks(x, mode_indices)
    plt.xlabel("Mode index")
    plt.ylabel("Amplitude coefficient")
    plt.title(f"{title} {scale_info}")
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

def E_lateral_shift(E, X, Y, shift_x=0.0, shift_y=0.0):
    """
    对光场施加横向偏心（平移）

    Parameters
    ----------
    E : torch.Tensor
        (H, W) or (B, H, W), complex
    X, Y : torch.Tensor
        (H, W), physical coordinates
    shift_x : float
        偏心量（与 X 单位一致）
    shift_y : float
        偏心量（与 Y 单位一致）

    Returns
    -------
    E_shifted : torch.Tensor
        与 E 同 shape 的偏心光场
    """

    # ===== 保证 batch 维 =====
    if E.dim() == 2:
        E = E.unsqueeze(0)  # (1,H,W)

    B, H, W = E.shape

    # ===== 构造新坐标（反向采样）=====
    Xs = X - shift_x
    Ys = Y - shift_y

    # ===== 归一化到 grid_sample 坐标 [-1,1] =====
    x_min, x_max = X.min(), X.max()
    y_min, y_max = Y.min(), Y.max()

    Xn = 2 * (Xs - x_min) / (x_max - x_min) - 1
    Yn = 2 * (Ys - y_min) / (y_max - y_min) - 1

    grid = torch.stack((Xn, Yn), dim=-1)  # (H,W,2)
    grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # (B,H,W,2)

    # ===== 分别插值实部和虚部 =====
    E_real = F.grid_sample(
        E.real.unsqueeze(1), grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True
    )

    E_imag = F.grid_sample(
        E.imag.unsqueeze(1), grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True
    )

    E_shifted = E_real.squeeze(1) + 1j * E_imag.squeeze(1)

    return E_shifted.squeeze(0) if B == 1 else E_shifted

def partition_slm(
    Nx, Ny,          # SLM pixel number
    dx, dy,          # pixel size (mm)
    n_layers,        # number of diffraction layers
    N_eff=256,
    device="cpu",
):
    """
    Partition SLM into multiple square effective blocks.

    Returns
    -------
    blocks : list of dict
        Each dict contains geometry & coordinate info for one layer.
    """

    """
    Partition SLM for multiple layers:
    - X方向均分 n_layers
    - Y方向固定居中，短边只占有效区域
    """

    if N_eff > Ny:
        raise ValueError(f"N_eff={N_eff} > Ny={Ny}, cannot fit along short edge")

    blocks = []
    Nx_per_layer = Nx // n_layers
    if Nx_per_layer < N_eff:
        raise ValueError(f"每层 Nx {Nx_per_layer} < N_eff {N_eff}")

    # y方向居中
    y_center = Ny // 2
    iy_eff0 = y_center - N_eff // 2
    iy_eff1 = iy_eff0 + N_eff

    for i in range(n_layers):
        # x方向
        x_min = i * Nx_per_layer
        x_max = x_min + Nx_per_layer
        cx = x_min + Nx_per_layer // 2

        # x有效区域
        ix_eff0 = cx - N_eff // 2
        ix_eff1 = ix_eff0 + N_eff

        # 局部坐标 mm
        x = (torch.arange(N_eff, device=device) - N_eff // 2) * dx
        y = (torch.arange(N_eff, device=device) - N_eff // 2) * dy
        X, Y = torch.meshgrid(x, y, indexing="ij")

        blocks.append({
            "layer_id": i,
            "ix_range": (ix_eff0, ix_eff1),
            "iy_range": (iy_eff0, iy_eff1),
            "Nx_eff": N_eff,
            "Ny_eff": N_eff,
            "x": x,
            "y": y,
            "X": X,
            "Y": Y,
            "center_pixel": (cx, y_center),
        })

    return blocks

def assemble_slm(
    blocks,
    phases_list,
    SLM_Nx,
    SLM_Ny,
    tilt_angle=(0.0, 0.0),
    offsets=None,
    wavelength=0.000633,
    dx=0,
    dy=0,
    device='cpu',
    show_plot=True,
    cmap='jet',
    background_blaze=False,          # ⭐ 新增：是否开启背景倾斜
    background_tilt=(0.0, 0.0)       # ⭐ 新增：背景倾斜角（deg）
):
    """
    将多个 phase block 拼接到 SLM 上，并支持：
    - block 平铺
    - offset 偏移
    - tilt 相位
    - background blaze（仅在非block区域）
    """

    import torch
    import numpy as np
    import matplotlib.pyplot as plt

    # -------------------------------------------------
    # offsets 统一处理
    # -------------------------------------------------
    if offsets is not None:
        if isinstance(offsets, list):
            offsets = torch.tensor(offsets, dtype=torch.float32, device=device)
        elif isinstance(offsets, torch.Tensor):
            offsets = offsets.to(device)
        else:
            raise TypeError("offsets must be list or torch.Tensor")

        if offsets.shape[1] != 2:
            raise ValueError("offsets must have shape (N_blocks, 2)")

        if offsets.shape[0] < len(blocks):
            pad = torch.zeros(len(blocks) - offsets.shape[0], 2, device=device)
            offsets = torch.cat([offsets, pad], dim=0)

    # -------------------------------------------------
    # 初始化 SLM + 背景 mask
    # -------------------------------------------------
    slm_phase = torch.zeros(
        (SLM_Ny, SLM_Nx),
        dtype=torch.float32,
        device=device
    )

    bg_mask = torch.ones((SLM_Ny, SLM_Nx), device=device)

    # -------------------------------------------------
    # tilt 参数
    # -------------------------------------------------
    if isinstance(tilt_angle, torch.Tensor):
        theta_x = tilt_angle[0].item()
        theta_y = tilt_angle[1].item()
    else:
        theta_x, theta_y = tilt_angle

    k = 2 * np.pi / wavelength

    # -------------------------------------------------
    # block loop
    # -------------------------------------------------
    for i, blk in enumerate(blocks):

        x_min, x_max = blk['ix_range']
        y_min, y_max = blk['iy_range']
        Nx_blk = blk['Nx_eff']
        Ny_blk = blk['Ny_eff']

        phase_block = phases_list[i].to(device)

        # ----------------------------
        # x flip（反射）
        # ----------------------------
        phase_block = torch.flip(phase_block, dims=[-1])

        # ----------------------------
        # size align
        # ----------------------------
        if phase_block.shape != (Ny_blk, Nx_blk):
            tmp = torch.zeros((Ny_blk, Nx_blk), device=device)

            if len(phase_block.shape) == 3:
                _, ny, nx = phase_block.shape
            else:
                ny, nx = phase_block.shape

            y0 = (Ny_blk - ny) // 2
            x0 = (Nx_blk - nx) // 2
            tmp[y0:y0+ny, x0:x0+nx] = phase_block
            phase_block = tmp

        # ----------------------------
        # tilt inside block
        # ----------------------------
        x_idx = torch.arange(Nx_blk, device=device)
        y_idx = torch.arange(Ny_blk, device=device)
        X_blk, Y_blk = torch.meshgrid(x_idx, y_idx, indexing='xy')

        X_blk = (X_blk - (Nx_blk - 1) / 2) * dx
        Y_blk = (Y_blk - (Ny_blk - 1) / 2) * dy

        phase_tilt_blk = k * (
            X_blk * torch.tan(torch.tensor(theta_x, device=device)) +
            Y_blk * torch.tan(torch.tensor(theta_y, device=device))
        )

        # ----------------------------
        # combine phase
        # ----------------------------
        phase_block = torch.angle(
            torch.exp(1j * phase_block) *
            torch.exp(1j * phase_tilt_blk)
        )

        # ----------------------------
        # y flip
        # ----------------------------
        phase_block = torch.flip(phase_block, dims=[-2])

        # ----------------------------
        # offset
        # ----------------------------
        if offsets is not None:
            shift_x = int(offsets[i, 0].item())
            shift_y = int(offsets[i, 1].item())
        else:
            shift_x = 0
            shift_y = 0

        x_min_s = x_min + shift_x
        x_max_s = x_max + shift_x
        y_min_s = y_min + shift_y
        y_max_s = y_max + shift_y

        # ----------------------------
        # crop
        # ----------------------------
        x0 = max(0, x_min_s)
        x1 = min(SLM_Nx, x_max_s)
        y0 = max(0, y_min_s)
        y1 = min(SLM_Ny, y_max_s)

        if (x1 <= x0) or (y1 <= y0):
            continue

        bx0 = x0 - x_min_s
        by0 = y0 - y_min_s
        bx1 = bx0 + (x1 - x0)
        by1 = by0 + (y1 - y0)

        # ----------------------------
        # write block
        # ----------------------------
        slm_phase[y0:y1, x0:x1] = phase_block[by0:by1, bx0:bx1]

        # ⭐ mark background
        bg_mask[y0:y1, x0:x1] = 0

    # -------------------------------------------------
    # background blaze (ONLY outside blocks)
    # -------------------------------------------------
    if background_blaze:

        k = 2 * np.pi / wavelength

        theta_bg_x = torch.deg2rad(torch.tensor(background_tilt[0], device=device))
        theta_bg_y = torch.deg2rad(torch.tensor(background_tilt[1], device=device))

        x = torch.arange(SLM_Nx, device=device)
        y = torch.arange(SLM_Ny, device=device)

        X, Y = torch.meshgrid(x, y, indexing='xy')

        X = (X - (SLM_Nx - 1) / 2) * dx
        Y = (Y - (SLM_Ny - 1) / 2) * dy

        blaze = k * (
            X * torch.sin(theta_bg_x) +
            Y * torch.sin(theta_bg_y)
        )

        slm_phase = slm_phase + torch.angle(torch.exp(1j*blaze)) * bg_mask

    # -------------------------------------------------
    # visualization
    # -------------------------------------------------
    if show_plot:
        plt.figure(figsize=(10, 6))
        plt.imshow(slm_phase.detach().cpu().numpy(), cmap=cmap, aspect='equal')
        plt.colorbar(label='Phase (rad)')
        plt.title(f'SLM Phase | tilt=({theta_x:.2f},{theta_y:.2f})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.tight_layout()
        plt.show()

    return slm_phase

def compute_slm_tilt_and_distance(block_list, d_layer, n_layers, SLM_Nx, SLM_Ny, dx, dy, device, debug=False):
    """
    基于等腰三角形几何关系，计算：
    - SLM 需要加载的全局倾斜角 theta_x
    - SLM ↔ 平面镜的物理间距 h

    几何模型：
    - 相邻 block 中心间距 D
    - 两次反射，每次传播 d/2
    - sin(theta) = D / d
    """

    # ==========================
    # 1️⃣ 计算所有 block 的全局中心（物理坐标）
    # ==========================
    x_centers = []

    for b in block_list:
        ix0, ix1 = b['ix_range']
        cx_global = 0.5 * (ix0 + ix1)

        # 以 SLM 中心为原点，转为物理坐标（mm）
        x_center = (cx_global - SLM_Nx / 2) * dx
        x_centers.append(x_center)

    x_centers = torch.tensor(x_centers, device=device)

    # ==========================
    # 2️⃣ 计算相邻 block 的中心距离 D
    # ==========================
    x_centers_sorted, _ = torch.sort(x_centers)

    if x_centers_sorted.numel() < 2:
        raise ValueError("Need at least two blocks to determine tilt angle.")

    D_list = x_centers_sorted[1:] - x_centers_sorted[:-1]
    D = D_list.mean()   # 取平均更稳健

    # ==========================
    # 3️⃣ 等腰三角形 → 倾斜角
    # ==========================
    # sin(theta) = D / d
    if torch.abs(D / d_layer) > 1:
        raise ValueError("Invalid geometry: D > d_layer, no real tilt angle.")

    theta_x = torch.asin(D / d_layer)
    theta_y = torch.tensor(0.0, device=device)

    tilt_angle = torch.stack([theta_x, theta_y])  # (2,)

    # ==========================
    # 4️⃣ SLM ↔ 平面镜间距
    # ==========================
    # h = (d/2) * cos(theta)
    slm_mirror_distance = (d_layer / 2) * torch.cos(theta_x)

    # ==========================
    # 5️⃣ Debug 输出
    # ==========================
    if debug:
        print("=== SLM tilt geometry ===")
        print(f"Block center spacing D = {D.item():.4f} mm")
        print(f"Layer distance d      = {d_layer:.4f} mm")
        print(f"theta_x               = {torch.rad2deg(theta_x):.4f} deg")
        print(f"SLM-mirror distance h = {slm_mirror_distance.item():.4f} mm")

    return tilt_angle, slm_mirror_distance

def resize_phase(
    phase,
    Nx_new,
    Ny_new,
    mode='bicubic'
):
    """
    将相位场在物理意义上做连续插值（保持像素尺寸不变，扩大物理范围）
    若输入尺寸已匹配目标尺寸，则直接返回原相位

    Parameters
    ----------
    phase : torch.Tensor
        原始相位，shape = (Ny_old, Nx_old)
    Nx_new, Ny_new : int
        目标尺寸
    mode : str
        插值方式: 'bilinear' | 'bicubic'

    Returns
    -------
    phase_new : torch.Tensor
        插值后的相位，shape = (Ny_new, Nx_new)
    """

    Ny_old, Nx_old = phase.shape[-2], phase.shape[-1]

    # ===== 尺寸一致：直接返回 =====
    if (Nx_old == Nx_new) and (Ny_old == Ny_new):
        return phase

    # 1. 相位 -> 复振幅
    complex_field = torch.exp(1j * phase)

    # 2. 拆成实部和虚部
    real = complex_field.real.unsqueeze(0).unsqueeze(0)  # (1,1,Ny,Nx)
    imag = complex_field.imag.unsqueeze(0).unsqueeze(0)

    # 3. 连续插值
    real_new = F.interpolate(
        real,
        size=(Ny_new, Nx_new),
        mode=mode,
        align_corners=False
    )
    imag_new = F.interpolate(
        imag,
        size=(Ny_new, Nx_new),
        mode=mode,
        align_corners=False
    )

    # 4. 合成复场并取相位
    complex_new = real_new + 1j * imag_new
    phase_new = torch.angle(complex_new.squeeze(0).squeeze(0))

    return phase_new

def embed_center(pattern_small, Nx, Ny, device=None):
    """
    将小图居中放入大图 (Nx, Ny)

    参数
    ----------
    pattern_small : torch.Tensor
        输入小图 (H, W)
    Nx, Ny : int
        输出 SLM 图尺寸
    device : torch.device 或 str
        输出设备

    返回
    ----------
    pattern_big : torch.Tensor
        输出大图 (Nx, Ny)
    """

    if device is None:
        device = pattern_small.device

    H, W = pattern_small.shape  # H=y , W=x

    # 创建大图 (Ny, Nx)
    pattern_big = torch.zeros((Ny, Nx), dtype=pattern_small.dtype, device=device)

    # 中心位置
    start_y = Ny // 2 - H // 2
    start_x = Nx // 2 - W // 2

    end_y = start_y + H
    end_x = start_x + W

    pattern_big[start_y:end_y, start_x:end_x] = pattern_small
    return pattern_big
