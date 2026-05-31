import torch
import utils as ut

def place_PIM_modes(
    detector_shape,
    PIM,
    mode_indices,
    centers_xy,
    coeff,          # (B, M) 复系数
):
    """
    将 B 个输入场的模式系数，映射为 B 张探测器光场分布

    Parameters
    ----------
    detector_shape : tuple
        (H, W)

    PIM : torch.Tensor (complex)
        shape: (H*W, M)

    mode_indices : torch.LongTensor
        shape: (N,)

    centers_xy : torch.Tensor
        shape: (N, 2)

    coeff : torch.Tensor (complex)
        shape: (B, M)

    Returns
    -------
    E_out : torch.Tensor (complex)
        shape: (B, H, W)
    """

    device = PIM.device
    dtype = PIM.dtype

    H, W = detector_shape
    B, M = coeff.shape

    N_mode = mode_indices.numel()
    N_pos  = centers_xy.shape[0]

    N_use = min(N_mode, N_pos)

    mode_indices = mode_indices[:N_use]
    centers_xy   = centers_xy[:N_use]

    # 输出场
    E_out = torch.zeros((B, H, W), device=device, dtype=dtype)

    for i in range(N_use):
        # 取第 i 个模式（完整 HxW）
        mode = PIM[:, mode_indices[i]].reshape(H, W)

        cx, cy = centers_xy[i]
        cx = int(torch.round(cx).item())
        cy = int(torch.round(cy).item())

        dx = cx - W // 2
        dy = cy - H // 2

        shifted = torch.roll(mode, shifts=(dy, dx), dims=(0, 1))
        # shifted: (H, W)

        # 第 i 个模式在 batch 中的振幅系数
        # coeff_i: (B,)
        coeff_i = torch.abs(coeff[:, mode_indices[i]])

        # 广播叠加
        E_out += coeff_i[:, None, None] * shifted[None, :, :]

    return E_out

def generate_grid_centers(
    I,
    n_row,
    n_col,
    phys_range,     # (x_min, x_max, y_min, y_max)，单位 mm
    full_range,     # (X_min, X_max, Y_min, Y_max)，I 对应的物理范围
):
    """
    在给定物理范围内生成网格中心（输出仍是像素坐标）

    phys_range  ⊂ full_range
    """

    if I.dim() == 3:
        _, H, W = I.shape
    elif I.dim() == 2:
        H, W = I.shape
    else:
        raise ValueError("I must be 2D or 3D tensor")

    device = I.device
    dtype  = torch.float32

    x_min, x_max, y_min, y_max = phys_range
    X_min, X_max, Y_min, Y_max = full_range

    # ---------- 物理 → 像素映射 ----------
    def phys_to_px(x, x0, x1, N):
        return (x - x0) / (x1 - x0) * (N - 1)

    x0_px = phys_to_px(x_min, X_min, X_max, W)
    x1_px = phys_to_px(x_max, X_min, X_max, W)
    y0_px = phys_to_px(y_min, Y_min, Y_max, H)
    y1_px = phys_to_px(y_max, Y_min, Y_max, H)

    # 子区域尺寸（像素）
    sub_W = x1_px - x0_px
    sub_H = y1_px - y0_px

    cell_w = sub_W / n_col
    cell_h = sub_H / n_row

    centers = []

    for i in range(n_row):       # y
        for j in range(n_col):   # x
            x_center = x0_px + (j + 0.5) * cell_w
            y_center = y0_px + (i + 0.5) * cell_h
            centers.append([x_center, y_center])

    centers_xy = torch.tensor(
        centers,
        device=device,
        dtype=dtype
    )

    return centers_xy

def compute_overlap(
    x, y,             # (H, W) 物理坐标矩阵，单位 mm
    E_out,             # (M, H, W) complex, 模式阵列输出
    modes_ref,         # (H*W, N) complex, 能量归一化参考模式
    centers_xy,        # (N, 2) pixel, 每个模式中心
    shape_px           # (h, w) 单个模式的像素尺寸
):
    """
    计算模式阵列输出与参考模式的振幅系数（物理积分）。

    Parameters
    ----------
    x, y : torch.Tensor
        (H, W) 物理坐标矩阵，单位 mm
    E_out : torch.Tensor
        (M, H, W) complex, 模式阵列输出
    modes_ref : torch.Tensor
        (H*W, N) complex, 能量归一化参考模式
    centers_xy : torch.Tensor
        (N, 2) pixel, 每个模式中心
    shape_px : tuple
        (h, w) 单个模式的像素尺寸

    Returns
    -------
    coeffs : torch.Tensor
        (M, N) complex, 每个输出场与参考模式的振幅系数
    """
    device = E_out.device
    dtype = E_out.dtype

    M, H, W = E_out.shape
    _, N = modes_ref.shape

    h, w = shape_px
    half_h, half_w = h // 2, w // 2

    # 计算像素物理尺寸
    dx = torch.abs(x[0,1] - x[0,0])
    dy = torch.abs(y[1,0] - y[0,0])

    # reshape 参考模式
    modes_hw = modes_ref.T.reshape(N, H, W)

    coeffs = torch.zeros((M, N), device=device, dtype=dtype)

    for n in range(N):
        cx, cy = centers_xy[n]
        cx = int(torch.round(cx).item())
        cy = int(torch.round(cy).item())

        # 原 patch 边界
        y0 = max(cy - half_h, 0)
        y1 = min(cy + half_h, H)
        x0 = max(cx - half_w, 0)
        x1 = min(cx + half_w, W)

        patch = E_out[:, y0:y1, x0:x1]  # (M, h', w')

        # 居中 pad 回完整尺寸
        padded = torch.zeros((M, H, W), device=device, dtype=dtype)
        h_patch, w_patch = patch.shape[1], patch.shape[2]
        y_start = H//2 - h_patch//2
        x_start = W//2 - w_patch//2
        y_end   = y_start + h_patch
        x_end   = x_start + w_patch
        padded[:, y_start:y_end, x_start:x_end] = patch

        # 展平
        E_flat = padded.reshape(M, -1)
        u_flat = modes_ref[:, n]

        # 物理积分求振幅系数
        coeffs[:, n] = torch.sum(torch.conj(u_flat) * E_flat, dim=1) * dx * dy

    return coeffs