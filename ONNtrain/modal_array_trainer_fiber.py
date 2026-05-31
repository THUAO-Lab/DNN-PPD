import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from ONNtrain import Rayl_Somm_Diffr
import utils as ut
import matplotlib.pyplot as plt
import os


class Modal_array_fiber_trainer(nn.Module):
    def __init__(
        self,
        modulator,              # 相位调制器模型（如 PhaseModulator）
        PIM,                    # PIM模式基底 ∈ ℂ^{HW × N_modes}
        Index,           # 目标模式索引
        x,                     # 空间坐标网格 ∈ ℝ^{H × W}
        y,                     # 空间坐标网格 ∈ ℝ^{H × W}
        H_small, W_small,
        centers_xy,            # xy局部坐标
        centers_xy_field,      # 光场的局部
        front_num,             # 前端衍射层数
        Lambda,                # 光场波段（单位：米）
        lr=0.01,               # 学习率
        batch_size=8,          # batch 大小
        weight_shape=0.0,
        device='cpu',          # 计算设备
        mask=1                # 掩模板
    ):
        super().__init__()
        # 模块与参数赋值
        self.modulator = modulator.to(device)
        self.PIM = PIM.detach().to(device)
        self.Index = Index
        self.x = x.detach().to(device)
        self.y = y.detach().to(device)
        self.Value_big = torch.zeros_like(self.x)
        self.centers_xy = centers_xy.detach().to(device)
        self.centers_xy_field = centers_xy_field.detach().to(device)
        self.Lambda = Lambda
        self.k = 2 * torch.pi / Lambda
        self.device = device
        self.batch_size = batch_size
        self.mask = mask
        self.dx = (x[0,1]-x[0,0]) #像素x大小
        self.dy = (y[1,0]-y[0,0]) #像素y大小
        self.front = front_num
        self.cell_shape_px = None
        self.cell_shape_px_big = None
        self.OAM_phase = torch.zeros_like(self.x) #默认涡旋光相位全0
        self.H_small, self.W_small = H_small, W_small #小窗口对应的像素数
        self.H, self.W = self.x.shape #大窗口对应的像素数
        self.Prop_1 = modulator.Prop_f1.to(device)
        self.Prop_2 = modulator.Prop_f2.to(device)
        self.weight_shape = weight_shape


        # 光学神经网络相位优化参数
        params = list(self.modulator.parameters())
        
        self.optimizer = torch.optim.Adam(params, lr=lr)
        
    def estimate_cell_shape_px(self):
        """
        从规则网格 centers_xy_px 中估计微透镜单元的像素尺寸
        只用于初始化，不用于训练
        """
    
        xs = self.centers_xy[:, 0].float()
        ys = self.centers_xy[:, 1].float()
    
        # ---- x 方向 ----
        xs_unique = torch.unique(xs)
        xs_unique, _ = torch.sort(xs_unique)
    
        if xs_unique.numel() > 1:
            dx = torch.diff(xs_unique)
            cell_w = torch.median(dx)   # 比 min 稳定
        else:
            cell_w = torch.tensor(float(self.W_small), device=xs.device)
    
        # ---- y 方向 ----
        ys_unique = torch.unique(ys)
        ys_unique, _ = torch.sort(ys_unique)
    
        if ys_unique.numel() > 1:
            dy = torch.diff(ys_unique)
            cell_h = torch.median(dy)
        else:
            cell_h = torch.tensor(float(self.H_small), device=ys.device)
    
        # ---- 转为 int（几何参数，不参与反传） ----
        cell_h = int(torch.round(cell_h).item())
        cell_w = int(torch.round(cell_w).item())
    
        self.cell_shape_px = (cell_h, cell_w)
        return cell_h, cell_w
    def estimate_cell_shape_px_big(self):
        """
        在“大窗口坐标系”下，根据模式中心分布估计每个模式占据的像素尺寸
    
        centers_xy 应该已经是大窗口坐标
        """
    
        xs = self.centers_xy_field[:, 0].float()
        ys = self.centers_xy_field[:, 1].float()
    
        # =================================================
        # x 方向
        # =================================================
        xs_unique = torch.unique(xs)
        xs_unique, _ = torch.sort(xs_unique)
    
        if xs_unique.numel() > 1:
            dx = torch.diff(xs_unique)
            cell_w = torch.median(dx)
        else:
            # 👉 fallback：整个窗口
            cell_w = torch.tensor(float(self.W), device=xs.device)
    
        # =================================================
        # y 方向
        # =================================================
        ys_unique = torch.unique(ys)
        ys_unique, _ = torch.sort(ys_unique)
    
        if ys_unique.numel() > 1:
            dy = torch.diff(ys_unique)
            cell_h = torch.median(dy)
        else:
            cell_h = torch.tensor(float(self.H), device=ys.device)
    
        # =================================================
        # 转 int
        # =================================================
        cell_h = int(torch.round(cell_h).item())
        cell_w = int(torch.round(cell_w).item())
    
        self.cell_shape_px_big = (cell_h, cell_w)
    
        return cell_h, cell_w
    def init_gaussian_kernel(self, sigma=3, kernel_size=None):
        """
        预先生成高斯卷积核，并存储在类里
        sigma: 高斯标准差，像素单位
        kernel_size: 卷积核大小，None 自动取 6*sigma+1
        """
        device = self.device
        if kernel_size is None:
            kernel_size = int(6*sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1  # 保证奇数
    
        # 坐标网格
        ax = torch.arange(kernel_size, device=device) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2)/(2*sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1,1,K,K)
        self.gaussian_kernel_dw = kernel  # 存在类里，depthwise 卷积时会 expand
        self.gaussian_kernel_size = kernel_size

    def smooth_modulated_layers(self, modulated_layers):
        """
        对 modulated_layers（list of complex (H,W) 或 (1,H,W)）做高斯相位平滑
        使用已生成 self.gaussian_kernel_dw
        """
        kernel = self.gaussian_kernel_dw
        L = len(modulated_layers)
        Ms = [M.squeeze() for M in modulated_layers]  # (H,W)
        M = torch.stack(Ms, dim=0)  # (L,H,W)
        phase = torch.angle(M)
    
        # ---- depthwise conv ----
        phase = phase.unsqueeze(0)  # (1,L,H,W)
        kernel_dw = kernel.repeat(L, 1, 1, 1)  # (L,1,K,K)
        phase_s = F.conv2d(phase, kernel_dw, padding=self.gaussian_kernel_size//2, groups=L).squeeze(0)  # (L,H,W)
    
        # ---- 重建复振幅 ----
        M_s = torch.exp(1j * phase_s)
        return [M_s[i] for i in range(L)]
        
    def generate_microlens_phase(
        self,
        f_list,
        overlap=0.05,
    ):
    
        N = self.centers_xy.shape[0]
    
        cell_h, cell_w = self.cell_shape_px
        half_h = cell_h * (0.5 + overlap)
        half_w = cell_w * (0.5 + overlap)
    
        # -------------------------------------------------
        # ✅ 坐标网格（pixel）
        # -------------------------------------------------
        yy_px, xx_px = torch.meshgrid(
            torch.arange(self.H, device=self.device),
            torch.arange(self.W, device=self.device),
            indexing="ij"
        )
    
        # -------------------------------------------------
        # ✅ centers 坐标统一（关键）
        # -------------------------------------------------
        if hasattr(self, "H_small") and hasattr(self, "W_small"):
            centers_xy = self.shift_centers(
                self.centers_xy,
                self.H_small, self.W_small,
                self.H, self.W,
                device=self.device
            )
        else:
            centers_xy = self.centers_xy.to(self.device).float()
    
        # -------------------------------------------------
        # 初始化
        # -------------------------------------------------
        phase_sum  = torch.zeros((self.H, self.W), device=self.device)
        weight_sum = torch.zeros((self.H, self.W), device=self.device)
    
        # 像素 → 物理尺度
        dx_phys = self.x[0,1] - self.x[0,0]
        dy_phys = self.y[1,0] - self.y[0,0]
    
        x0 = self.x[0,0]
        y0 = self.y[0,0]
    
        # -------------------------------------------------
        # 主循环
        # -------------------------------------------------
        for i in range(N):
    
            cx_px, cy_px = centers_xy[i]
    
            # -------- window（pixel 坐标）--------
            ux = (xx_px - cx_px) / half_w
            uy = (yy_px - cy_px) / half_h
    
            wx = torch.zeros_like(ux)
            wy = torch.zeros_like(uy)
    
            inside_x = ux.abs() <= 1.0
            inside_y = uy.abs() <= 1.0
    
            wx[inside_x] = 0.5 * (1 + torch.cos(torch.pi * ux[inside_x].abs()))
            wy[inside_y] = 0.5 * (1 + torch.cos(torch.pi * uy[inside_y].abs()))
    
            window = wx * wy
    
            # -------- 物理坐标（连续！）--------
            cx = x0 + cx_px * dx_phys
            cy = y0 + cy_px * dy_phys
    
            dx = self.x - cx
            dy = self.y - cy
    
            # -------- 透镜相位 --------
            phase_i = self.k / (2 * f_list[i]) * (dx**2 + dy**2)
    
            phase_sum  += window * phase_i
            weight_sum += window
    
        weight_sum = torch.clamp(weight_sum, min=0.1)
        phase = phase_sum / weight_sum
    
        return phase.unsqueeze(0)
    
    def shift_centers(self, centers_xy, H_small, W_small, H_big, W_big, device=None):
        """
        将小窗口中的中心坐标平移到大窗口坐标系
    
        centers_xy: (N, 2)  [x, y] pixel
        返回: (N, 2) 新坐标
        """
    
        if device is None:
            device = centers_xy.device
    
        offset_x = (W_big - W_small) // 2
        offset_y = (H_big - H_small) // 2
    
        centers_new = centers_xy.clone().to(device).float()
    
        centers_new[:, 0] += offset_x
        centers_new[:, 1] += offset_y
    
        return centers_new
    
    def compute_overlap(
        self,
        E_out,             # (M, H_big, W_big) complex, 输出光场（大计算窗口，模式阵列）
        modes_ref,         # (H_big*W_big, N) complex, 参考模式（定义在中心坐标系，已归一化）
        H_small=None,
        W_small=None
    ):
        """
        计算输出光场在参考模式基底上的投影系数（overlap integral）
    
        物理意义：
            c_n = ∫∫ u_n*(x,y) · E(x,y) dxdy
    
        但注意：
            E_out 中每个模式位于不同空间位置（阵列形式），
            而 modes_ref 定义在“中心坐标系”，
    
            → 因此必须：
                1. 截取局部 patch
                2. 平移到中心
                3. 再做内积
    
        Parameters
        ----------
        E_out : (M, H_big, W_big)
            输出光场（可能是多个模式阵列）
    
        modes_ref : (H_big*W_big, N)
            参考模式（每一列一个模式，定义在大窗口中心）
    
        H_small, W_small : int
            原始小窗口尺寸（用于将 centers 平移到大窗口坐标系）
    
        Returns
        -------
        coeffs : (M, N)
            每个输出光场在每个参考模式上的复振幅系数
        """
    
        device = E_out.device
        dtype = E_out.dtype
    
        M, H_big, W_big = E_out.shape
        _, N = modes_ref.shape
        use_big_coord = (
                (H_small is None) or (W_small is None) or
                (H_small == H_big and W_small == W_big)
            )
    
        if use_big_coord:
            h, w = self.cell_shape_px_big
            half_h, half_w = h // 2, w // 2
            dx = torch.abs(self.x[0,1] - self.x[0,0])
            dy = torch.abs(self.y[1,0] - self.y[0,0])
            centers_xy = self.centers_xy_field.to(device).float()
        else:
            # -------------------------------------------------
            # ✅ 模式单元尺寸（pixel）
            # -------------------------------------------------
            # 表示每个微透镜/模式占据的空间范围
            h, w = self.cell_shape_px
            half_h, half_w = h // 2, w // 2
        
            # -------------------------------------------------
            # ✅ Step 1: 像素 → 物理面积（用于积分）
            # -------------------------------------------------
            # dx, dy 是物理坐标间距（例如 mm）
            # → dx * dy 对应连续积分中的 dA
            dx = torch.abs(self.x[0,1] - self.x[0,0])
            dy = torch.abs(self.y[1,0] - self.y[0,0])
        
            # -------------------------------------------------
            # ✅ Step 2: 坐标系统对齐（非常关键）
            # -------------------------------------------------
            # centers_xy 原本是在“小窗口坐标系”
            # 如果 E_out 已经被 insert 到“大窗口”，
            # 则需要做中心平移（shift）
            centers_xy = self.shift_centers(
                self.centers_xy,   # 原始中心（小窗口）
                H_small, W_small,
                H_big, W_big,      # 目标大窗口
                device=device
            )
            
    
        # -------------------------------------------------
        # 初始化输出系数
        # -------------------------------------------------
        coeffs = torch.zeros((M, N), device=device, dtype=dtype)
    
        # -------------------------------------------------
        # 主循环：逐个模式计算 overlap
        # -------------------------------------------------
        for n in range(N):
    
            # 当前模式中心（pixel坐标）
            cx, cy = centers_xy[n]
    
            # 转为整数索引（⚠️ 会引入不可导性）
            cx = int(torch.round(cx).item())
            cy = int(torch.round(cy).item())
    
            # -------------------------------------------------
            # Step 3: 从输出光场中截取该模式对应区域（patch）
            # -------------------------------------------------
            # 只取当前模式附近区域，减少干扰 & 提高效率
            y0 = max(cy - half_h, 0)
            y1 = min(cy + half_h, H_big)
            x0 = max(cx - half_w, 0)
            x1 = min(cx + half_w, W_big)
    
            patch = E_out[:, y0:y1, x0:x1]  # (M, h', w')
    
            # -------------------------------------------------
            # Step 4: 将 patch 平移到“大窗口中心”
            # -------------------------------------------------
            # 原因：
            #   modes_ref 是定义在中心坐标系
            #   → 必须把局部模式对齐到中心再比较
            padded = torch.zeros((M, H_big, W_big), device=device, dtype=dtype)
    
            h_patch, w_patch = patch.shape[1], patch.shape[2]
    
            # 计算中心对齐位置
            y_start = H_big // 2 - h_patch // 2
            x_start = W_big // 2 - w_patch // 2
    
            y_end = y_start + h_patch
            x_end = x_start + w_patch
    
            padded[:, y_start:y_end, x_start:x_end] = patch
    
            # -------------------------------------------------
            # Step 5: 展平为向量（方便做内积）
            # -------------------------------------------------            
            E_flat = padded.reshape(M, -1)   # (M, H*W)
            u_flat = modes_ref[:, n]         # (H*W,)
    
            # -------------------------------------------------
            # Step 6: 计算 overlap 积分（核心）
            # -------------------------------------------------
            # 数学形式：
            #   c_n = Σ conj(u_n) * E * dx * dy
            coeffs[:, n] = torch.sum(
                torch.conj(u_flat) * E_flat,
                dim=1
            ) * dx * dy
    
        return coeffs
    
    def insert_center(self, I_small):
        input_is_2d = (I_small.dim() == 2)
    
        I_big = self.Value_big.clone()
    
        # 👉 统一维度
        if input_is_2d:
            I_small = I_small.unsqueeze(0)
    
        if I_big.dim() == 2:
            I_big = I_big.unsqueeze(0)
    
        # 👉 ===== 关键：大矩阵跟随小矩阵 dtype =====
        I_big = I_big.to(I_small.dtype)
    
        B, H, W = I_small.shape
        _, H_big, W_big = I_big.shape
    
        h_start = (H_big - H) // 2
        w_start = (W_big - W) // 2
    
        I_big[:, h_start:h_start+H, w_start:w_start+W] = I_small
    
        if input_is_2d:
            return I_big.squeeze(0)
        else:
            return I_big
        
    def compute_pearson_loss(
        self,
        E_out,        # (M, H_big, W_big)
        modes_ref,    # (H_big*W_big, N)
        coeffs,       # (M, N)
        H_small=None,
        W_small=None
    ):
        device = E_out.device
        dtype = E_out.dtype
    
        M, H_big, W_big = E_out.shape
        _, N = modes_ref.shape
        use_big_coord = (
                (H_small is None) or (W_small is None) or
                (H_small == H_big and W_small == W_big)
            )
        if use_big_coord:
            h, w = self.cell_shape_px_big
            centers_xy = self.centers_xy_field.to(device).float()
        else:
            h, w = self.cell_shape_px
            centers_xy = self.shift_centers(
                self.centers_xy,
                H_small, W_small,
                H_big, W_big,
                device=device
            )
        half_h, half_w = h // 2, w // 2
    
        loss_map = torch.zeros((M, N), device=device)
    
        # ============================
        # 模式掩膜
        # ============================
        mask = (torch.abs(coeffs) > 0).float()

    
        # ============================
        # 主循环
        # ============================
        for n in range(N):
    
            cx, cy = centers_xy[n]
            cx = int(torch.round(cx).item())
            cy = int(torch.round(cy).item())
    
            # ----------------------------
            # Step 1: 裁剪 patch
            # ----------------------------
            y0 = max(cy - half_h, 0)
            y1 = min(cy + half_h, H_big)
            x0 = max(cx - half_w, 0)
            x1 = min(cx + half_w, W_big)
    
            patch = E_out[:, y0:y1, x0:x1]   # (M, h', w')
    
            # ----------------------------
            # Step 2: 平移到中心（和 overlap 一致）
            # ----------------------------
            padded = torch.zeros((M, H_big, W_big), device=device, dtype=dtype)
    
            h_patch, w_patch = patch.shape[1], patch.shape[2]
    
            y_start = H_big // 2 - h_patch // 2
            x_start = W_big // 2 - w_patch // 2
    
            y_end = y_start + h_patch
            x_end = x_start + w_patch
    
            padded[:, y_start:y_end, x_start:x_end] = patch
    
            # ----------------------------
            # Step 3: 展平
            # ----------------------------
            I_out = torch.abs(padded)**2      # (M, H*W)
            I_out = I_out.reshape(M, -1)
            u_flat = modes_ref[:, n]          # (H*W,)
            I_ref = torch.abs(u_flat)**2      # (H*W,)
    
            # ----------------------------
            # Step 4: Pearson
            # ----------------------------
            I_ref_mean = I_ref.mean()
            I_ref_std = I_ref.std() + 1e-8
    
            I_out_mean = I_out.mean(dim=1, keepdim=True)
            I_out_std = I_out.std(dim=1, keepdim=True) + 1e-8
    
            num = torch.sum(
                (I_out - I_out_mean) *
                (I_ref - I_ref_mean),
                dim=1
            )
    
            den = I_out_std.squeeze(1) * I_ref_std * I_out.shape[1]
    
            pearson = num / den   # (M,)
    
            # ----------------------------
            # Step 5: loss + mask
            # ----------------------------
            loss_map[:, n] = (1 - pearson.real) * mask[:, n]
    
        return loss_map
    def phase_smooth_loss(self,
        complex_field,           # (L,H,W) 或 (B,H,W)，实数相位
        weight=1.0,
        use_diff=False,
        phase_ref=None,   # φ0
        mask=None         # (H,W) 或 (B,H,W)
    ):
        """
        基于复振幅相位差分的平滑约束（最稳定版本）
    
        核心：
            Δφ = angle( U1 * conj(U2) )
    
        对应：
            L = ||Δx φ||² + ||Δy φ||²
    
        支持：
            - φ_diff = φ - φ0
            - mask（避免大窗口稀释问题）
    
        Returns
        -------
        loss : scalar
        """
    
        # -------------------------------------------------
        # 1. 维度统一
        # -------------------------------------------------
        if complex_field.dim() == 2:
            complex_field = complex_field.unsqueeze(0)
    
        # -------------------------------------------------
        # 2. 构造复振幅
        # -------------------------------------------------
        U = complex_field
    
        # -------------------------------------------------
        # 3. φ_diff（论文约束）
        # -------------------------------------------------
        if use_diff:
            if phase_ref is None:
                raise ValueError("use_diff=True 时必须提供 phase_ref")
    
            if phase_ref.dim() == 2:
                phase_ref = phase_ref.unsqueeze(0)
    
            U0 = torch.exp(1j * phase_ref)
    
            # 🔥 关键：相位差用复数表达
            U = U * torch.conj(U0)
    
        # -------------------------------------------------
        # 4. 相位差分（最核心）
        # -------------------------------------------------
        dx = torch.angle(
            U[..., :, 1:] * torch.conj(U[..., :, :-1])
        )
    
        dy = torch.angle(
            U[..., 1:, :] * torch.conj(U[..., :-1, :])
        )
    
        # -------------------------------------------------
        # 5. mask（解决大窗口问题）
        # -------------------------------------------------
        if mask is not None:
    
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
    
            # 只在有效区域内计算差分
            mask_x = mask[..., :, 1:] * mask[..., :, :-1]
            mask_y = mask[..., 1:, :] * mask[..., :-1, :]
    
            loss_x = (dx**2 * mask_x).sum() / (mask_x.sum() + 1e-8)
            loss_y = (dy**2 * mask_y).sum() / (mask_y.sum() + 1e-8)
    
            loss = loss_x + loss_y
    
        else:
            loss = (dx**2).mean() + (dy**2).mean()
    
        return weight * loss
    
    def compute_global_pearson_loss(
        self,
        E_out,     # (B, H, W) complex
        E_target   # (B, H, W) complex
    ):
        """
        全局强度 Pearson loss（用于压制杂光）
    
        在整个大窗口上计算：
            I_out vs I_target 的 Pearson 相关
    
        Returns
        -------
        loss : (B,)
        """
    
        device = E_out.device
    
        # ==============================
        # 1. 强度
        # ==============================
        I_out = torch.abs(E_out) ** 2        # (B,H,W)
        I_tar = torch.abs(E_target) ** 2     # (B,H,W)
    
        # ==============================
        # 2. 展平
        # ==============================
        B = I_out.shape[0]
    
        I_out = I_out.reshape(B, -1)   # (B, HW)
        I_tar = I_tar.reshape(B, -1)
    
        # ==============================
        # 3. 均值 & 标准差
        # ==============================
        I_out_mean = I_out.mean(dim=1, keepdim=True)
        I_tar_mean = I_tar.mean(dim=1, keepdim=True)
    
        I_out_std = I_out.std(dim=1, keepdim=True) + 1e-8
        I_tar_std = I_tar.std(dim=1, keepdim=True) + 1e-8
    
        # ==============================
        # 4. Pearson
        # ==============================
        num = torch.sum(
            (I_out - I_out_mean) * (I_tar - I_tar_mean),
            dim=1
        )
    
        den = I_out_std.squeeze(1) * I_tar_std.squeeze(1) * I_out.shape[1]
    
        pearson = num / den   # (B,)
    
        # ==============================
        # 5. loss
        # ==============================
        loss = 1 - pearson
    
        return loss
        
    def forward(self, E_in):
        """
        前向传播，返回最终输出场和中间场列表。
        中间场列表包含所有调制后和传播后的光场（带梯度，可用于正则化）。
        """
        B, H, W = E_in.shape
        _, N = self.PIM.shape
    
        # 获取调制层和传播距离
        modulated_layers, d_list = self.modulator(self.mask)  # [num_layers, H, W]
        f1 = self.modulator.f[0,:]  # 第一个焦距
        Prop_f1 = self.Prop_1
        Prop_f2 = self.Prop_2
        Phase_f1 = self.generate_microlens_phase(f1) # 生成微透镜阵列相位,放置于大窗口
        Espher_f1 = torch.exp(1j * Phase_f1)  # 第一个球面波
    
        E_mid = E_in.to(self.device)
        intermediate_fields = []  # 存储中间场（带梯度）
    
        # ========== 前端传播 ==========
        for i in range(self.front):
            M = modulated_layers[i].to(self.device).reshape(H, W)
            E_mid = E_mid * M
            intermediate_fields.append(E_mid)  # 记录调制后场
    
            if i == self.front - 1:
                continue
            else:
                E_mid = Rayl_Somm_Diffr(E_mid, self.x, self.y, d_list[i], self.k, self.Lambda, 0, device=self.device)
                intermediate_fields.append(E_mid)  # 记录传播后场
    
        # 前焦面传播
        E_mid = Rayl_Somm_Diffr(E_in.to(self.device), self.x, self.y, torch.abs(Prop_f1), self.k, self.Lambda, 0, device=self.device)
    
        # 球面波调制
        E_mid = E_mid * Espher_f1
    
        # ========== 后端传播（保持原有逻辑） ==========
        for i in range(self.modulator.num_layers - self.front):
            M = modulated_layers[i + self.front].to(self.device).reshape(H, W)
            E_mid = E_mid * M
    
            if i == self.modulator.num_layers - self.front - 1:
                continue
            else:
                # 这里按原逻辑分情况处理
                if i == self.modulator.num_layers - self.front - 2:
                    # 传播一次后，再乘以第二个球面波
                    E_mid = Rayl_Somm_Diffr(E_mid, self.x, self.y, d_list[i + self.front], self.k, self.Lambda, 0, device=self.device)
                    intermediate_fields.append(E_mid)  # 记录传播后场
                    f = self.modulator.f[i + 1, :]  # 第二个焦距
                    Phase = self.generate_microlens_phase(f)
                    Espher = torch.exp(-1j * Phase)
                    E_mid = E_mid * Espher
                else:
                    E_mid = Rayl_Somm_Diffr(E_mid, self.x, self.y, d_list[i + self.front], self.k, self.Lambda, 0, device=self.device)
                    intermediate_fields.append(E_mid)  # 记录传播后场
    
        # 输出面
        E_f2_out = Rayl_Somm_Diffr(E_mid, self.x, self.y, torch.abs(Prop_f2), self.k, self.Lambda, 0, device=self.device)
        return E_f2_out, intermediate_fields

    def compute_loss(self, Eout, Ein_para, E_in, E_target, intermediate_fields):
        """
        Compute the modal overlap loss used for training.
        """
        Para_rec = self.compute_overlap(torch.abs(Eout), torch.abs(self.PIM))
        loss_overlap = torch.sqrt(torch.mean(torch.abs(Para_rec - (Ein_para / 2))**2))

        loss = loss_overlap
        if self.weight_shape > 0:
            loss_map = self.compute_pearson_loss(torch.abs(Eout), torch.abs(self.PIM), Ein_para)
            loss_shape = torch.sqrt(torch.mean(loss_map**2))
            loss = loss + self.weight_shape * loss_shape

        return loss

    def train_loop(self, E_target_tra, E_target_val, train_set, val_set, train_para, val_para, epochs=1000):
        """
        联合训练：调制器 + 解码器神经网络
        完整流程：E_in → 调制器 → 光纤传播 → 强度图 → 解码器 → E_in_hat → 调制器 →E_mid_hat
        Loss = E_in 与 E_in_hat 的差异（可加上额外正则项）
        """
        # 初始化损失记录器
        self.train_losses = []
        self.val_losses = []
        #数据打包
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        train_para_loader = DataLoader(train_para, batch_size=self.batch_size, shuffle=False)
        val_para_loader = DataLoader(val_para, batch_size=self.batch_size, shuffle=False)
        E_target_tra_loader = DataLoader(E_target_tra, batch_size=self.batch_size, shuffle=False)
        E_target_val_loader = DataLoader(E_target_val, batch_size=self.batch_size, shuffle=False)

        for epoch in range(epochs):
            self.modulator.train()
            total_loss = 0

            for (E_batch, para_batch, E_target_batch) in tqdm(zip(train_loader, train_para_loader, E_target_tra_loader), total=len(train_loader), desc=f"[Train] Epoch {epoch+1}/{epochs}"):
                if isinstance(E_batch, (tuple, list)):
                    E_batch = E_batch[0]
                    para_batch = para_batch[0]
                    E_target_batch = E_target_batch[0]
                E_batch = E_batch.to(self.device)
                para_batch = para_batch.to(self.device)
                E_target_batch = E_target_batch.to(self.device)

                # 前向传播 + 反向传播
                self.optimizer.zero_grad()
                E_f2_out, intermediate_fields = self.forward(E_batch)
                loss = self.compute_loss(E_f2_out, para_batch, E_batch, E_target_batch, intermediate_fields)
                loss.backward()
                # for name, param in self.reconstructor.decoder1.named_parameters():
                #     if param.requires_grad:
                #         print(f"{name}: grad norm = {param.grad.norm()}")
                self.optimizer.step()
                total_loss += loss.item()

            # 打印训练损失
            self.train_losses.append(total_loss) #保存训练损失函数值
            print(f"[Train] Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4e}")

            # ====== 验证阶段 ======
            self.modulator.eval()
            with torch.no_grad():
                val_loss = 0
                for E_batch, para_batch, E_target_batch in zip(val_loader, val_para_loader, E_target_val_loader):
                    if isinstance(E_batch, (tuple, list)):
                        E_batch = E_batch[0]
                        para_batch = para_batch[0]
                        E_target_batch = E_target_batch[0]
                    E_batch = E_batch.to(self.device)
                    para_batch = para_batch.to(self.device)
                    E_target_batch = E_target_batch.to(self.device)
                    
                    E_f2_out, intermediate_fields = self.forward(E_batch)
                    loss = self.compute_loss(E_f2_out, para_batch, E_batch, E_target_batch, intermediate_fields)  # ✅ 参数顺序一致
                    val_loss += loss.item()
                self.val_losses.append(val_loss) #保存验证损失函数值
                print(f"[Val]   Epoch {epoch+1} Loss: {val_loss / len(val_loader):.4e}")
    def export_model_state(self, save_dir="trained_weights", Paint = False):
        """
        保存训练后的：
        - 每一层相位调制器的相位分布（保存为 .pt 和 .png）
        - 神经网络的权重参数
        """
        os.makedirs(save_dir, exist_ok=True)
    
        # === 保存调制器相位参数 ===
        with torch.no_grad():
            phases = self.modulator.phases.detach().cpu()  # [num_layers, H, W]
            reconstructor_weights = self.reconstructor.state_dict()
            for i in range(self.modulator.num_layers):
                phase_i = torch.clamp(phases[i], -torch.pi, torch.pi)
                torch.save(phase_i, f"{save_dir}/phase_layer_{i+1}.pt")
    
                # 可视化保存为 PNG 图像
                if Paint==True:
                    plt.imshow(phase_i.numpy(), cmap="twilight", vmin=-torch.pi, vmax=torch.pi)
                    plt.colorbar()
                    plt.title(f"Phase Layer {i+1}")
                    plt.savefig(f"{save_dir}/phase_layer_{i+1}.png")
                    plt.close()
    
        # === 保存重建网络参数 ===
        torch.save(self.reconstructor.state_dict(), f"{save_dir}/reconstructor_weights.pth")
    
        print(f"[Export] Phase & Network weights saved to {save_dir}/")
        return phases,reconstructor_weights
    
    def get_output_compex(self, complex_field, mask):
            # ===== 找到 mask 的边界 =====
            coords = torch.nonzero(mask)
    
            y_min = coords[:, 0].min().item()
            y_max = coords[:, 0].max().item()-1
            x_min = coords[:, 1].min().item()
            x_max = coords[:, 1].max().item()-1
    
            # ===== 为了避免多/少一行（关键修正）=====
            # 这里统一用“半开区间 +1”并确保尺寸精确
            y_max = y_max + 1
            x_max = x_max + 1
    
            # ===== 裁剪 =====
            if complex_field.dim() == 2:
                complex_field_effect = complex_field[
                    y_min:y_max,
                    x_min:x_max
                ]
            else:
                complex_field_effect = complex_field[
                    :,
                    y_min:y_max,
                    x_min:x_max
                ]
    
            return complex_field_effect
    
    def plot_training_curve(self, save_path=None):
        """绘制训练和验证损失曲线，可选择保存图像"""
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)
        
        # 绘制双对数曲线
        plt.semilogy(epochs, self.train_losses, 'b-o', label='Train Loss')
        plt.semilogy(epochs, self.val_losses, 'r--s', label='Validation Loss')
        
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss (log scale)')
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.legend()
        
        # 自动调整Y轴范围，排除零值
        min_loss = min(min(self.train_losses), min(self.val_losses))
        max_loss = max(max(self.train_losses), max(self.val_losses))
        plt.ylim(0.5 * min_loss, 2 * max_loss)
        
        # 保存或显示图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"[Plot] Training curve saved to {save_path}")
        else:
            plt.show()
        
        # plt.close()

