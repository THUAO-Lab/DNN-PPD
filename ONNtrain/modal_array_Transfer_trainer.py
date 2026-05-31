import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from ONNtrain import Rayl_Somm_Diffr
import matplotlib.pyplot as plt
import os


class Modal_array_transfer_trainer(nn.Module):
    """Trainer for the modal-array ONN transfer module."""

    def __init__(
        self,
        modulator,
        PIM,
        Index,
        x,
        y,
        H_small,
        W_small,
        centers_xy,
        centers_xy_field,
        front_num,
        Lambda,
        lr=0.01,
        batch_size=8,
        device='cpu',
        mask=1,
    ):
        super().__init__()
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
        self.dx = x[0, 1] - x[0, 0]
        self.dy = y[1, 0] - y[0, 0]
        self.front = front_num
        self.cell_shape_px = None
        self.cell_shape_px_big = None
        self.OAM_phase = torch.zeros_like(self.x)
        self.H_small, self.W_small = H_small, W_small
        self.H, self.W = self.x.shape
        self.Prop_1 = modulator.Prop_f1.to(device)
        self.Prop_2 = modulator.Prop_f2.to(device)

        self.optimizer = torch.optim.Adam(list(self.modulator.parameters()), lr=lr)
        
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
        Store a Gaussian kernel for scripts that still initialize the trainer
        with smoothing parameters. The current overlap-only loss does not call
        the smoothing path directly.
        """
        device = self.device
        if kernel_size is None:
            kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
    
        ax = torch.arange(kernel_size, device=device) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        self.gaussian_kernel_dw = kernel.unsqueeze(0).unsqueeze(0)
        self.gaussian_kernel_size = kernel_size
        
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
    
        if input_is_2d:
            I_small = I_small.unsqueeze(0)
    
        if I_big.dim() == 2:
            I_big = I_big.unsqueeze(0)
    
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
        
    def forward(self, E_in):
        """
        Propagate a batch through the trained optical stack.
        """
        _, H, W = E_in.shape
    
        modulated_layers, d_list = self.modulator(self.mask)
        f1 = self.modulator.f[0, :]
        Prop_f1 = self.Prop_1
        Prop_f2 = self.Prop_2
        Phase_f1 = self.generate_microlens_phase(f1)
        Espher_f1 = torch.exp(1j * Phase_f1)
    
        E_mid = E_in.to(self.device)
        intermediate_fields = []
    
        for i in range(self.front):
            M = modulated_layers[i].to(self.device).reshape(H, W)
            E_mid = E_mid * M
            intermediate_fields.append(E_mid)
    
            if i == self.front - 1:
                continue
            else:
                E_mid = Rayl_Somm_Diffr(E_mid, self.x, self.y, d_list[i], self.k, self.Lambda, 0, device=self.device)
                intermediate_fields.append(E_mid)
    
        E_mid = Rayl_Somm_Diffr(E_in.to(self.device), self.x, self.y, torch.abs(Prop_f1), self.k, self.Lambda, 0, device=self.device)
        E_mid = E_mid * Espher_f1
    
        for i in range(self.modulator.num_layers - self.front):
            M = modulated_layers[i + self.front].to(self.device).reshape(H, W)
            E_mid = E_mid * M
    
            if i == self.modulator.num_layers - self.front - 1:
                continue
            else:
                if i == self.modulator.num_layers - self.front - 2:
                    E_mid = Rayl_Somm_Diffr(E_mid, self.x, self.y, d_list[i + self.front], self.k, self.Lambda, 0, device=self.device)
                    intermediate_fields.append(E_mid)
                    f = self.modulator.f[i + 1, :]
                    Phase = self.generate_microlens_phase(f)
                    Espher = torch.exp(-1j * Phase)
                    E_mid = E_mid * Espher
                else:
                    E_mid = Rayl_Somm_Diffr(E_mid, self.x, self.y, d_list[i + self.front], self.k, self.Lambda, 0, device=self.device)
                    intermediate_fields.append(E_mid)
    
        E_f2_out = Rayl_Somm_Diffr(E_mid, self.x, self.y, torch.abs(Prop_f2), self.k, self.Lambda, 0, device=self.device)
    
        return E_f2_out, intermediate_fields

    def compute_loss(self, Eout, Ein_para, E_in, E_target, intermediate_fields):
        """Overlap-integral loss used by this transfer trainer."""
        Para_rec = self.compute_overlap(torch.abs(Eout), torch.abs(self.PIM))
        return torch.sqrt(torch.mean(torch.abs(Para_rec - (Ein_para / 2))**2))

    def train_loop(self, E_target_tra, E_target_val, train_set, val_set, train_para, val_para, epochs=1000):
        """Run the train/validation loop without changing caller-side datasets."""
        self.train_losses = []
        self.val_losses = []

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        train_para_loader = DataLoader(train_para, batch_size=self.batch_size, shuffle=False)
        val_para_loader = DataLoader(val_para, batch_size=self.batch_size, shuffle=False)
        E_target_tra_loader = DataLoader(E_target_tra, batch_size=self.batch_size, shuffle=False)
        E_target_val_loader = DataLoader(E_target_val, batch_size=self.batch_size, shuffle=False)

        for epoch in range(epochs):
            self.modulator.train()
            total_loss = 0

            train_iter = zip(train_loader, train_para_loader, E_target_tra_loader)
            for (E_batch, para_batch, E_target_batch) in tqdm(train_iter, total=len(train_loader), desc=f"[Train] Epoch {epoch+1}/{epochs}"):
                if isinstance(E_batch, (tuple, list)):
                    E_batch = E_batch[0]
                    para_batch = para_batch[0]
                    E_target_batch = E_target_batch[0]
                E_batch = E_batch.to(self.device)
                para_batch = para_batch.to(self.device)
                E_target_batch = E_target_batch.to(self.device)

                self.optimizer.zero_grad()
                E_f2_out, intermediate_fields = self.forward(E_batch)
                loss = self.compute_loss(E_f2_out, para_batch, E_batch, E_target_batch, intermediate_fields)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            self.train_losses.append(total_loss)
            print(f"[Train] Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4e}")

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
                    loss = self.compute_loss(E_f2_out, para_batch, E_batch, E_target_batch, intermediate_fields)
                    val_loss += loss.item()
                self.val_losses.append(val_loss)
                print(f"[Val]   Epoch {epoch+1} Loss: {val_loss / len(val_loader):.4e}")
    def export_model_state(self, save_dir="trained_weights", Paint = False):
        """Export trained phase layers and optional auxiliary network weights."""
        os.makedirs(save_dir, exist_ok=True)
    
        with torch.no_grad():
            phases = self.modulator.phases.detach().cpu()
            reconstructor_weights = None
            if hasattr(self, "reconstructor"):
                reconstructor_weights = self.reconstructor.state_dict()

            for i in range(self.modulator.num_layers):
                phase_i = torch.clamp(phases[i], -torch.pi, torch.pi)
                torch.save(phase_i, f"{save_dir}/phase_layer_{i+1}.pt")
    
                if Paint:
                    plt.imshow(phase_i.numpy(), cmap="twilight", vmin=-torch.pi, vmax=torch.pi)
                    plt.colorbar()
                    plt.title(f"Phase Layer {i+1}")
                    plt.savefig(f"{save_dir}/phase_layer_{i+1}.png")
                    plt.close()

        if reconstructor_weights is not None:
            torch.save(reconstructor_weights, f"{save_dir}/reconstructor_weights.pth")
    
        print(f"[Export] Phase & Network weights saved to {save_dir}/")
        return phases, reconstructor_weights
    
    def get_output_compex(self, complex_field, mask):
        coords = torch.nonzero(mask)

        y_min = coords[:, 0].min().item()
        y_max = coords[:, 0].max().item()
        x_min = coords[:, 1].min().item()
        x_max = coords[:, 1].max().item()

        if complex_field.dim() == 2:
            return complex_field[y_min:y_max, x_min:x_max]
        return complex_field[:, y_min:y_max, x_min:x_max]
    
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


