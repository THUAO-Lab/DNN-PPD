import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from ONNtrain import Rayl_Somm_Diffr
import utils as ut
import matplotlib.pyplot as plt
import os


class Modal_array_trainer_comparison(nn.Module):
    def __init__(
        self,
        modulator,              # 相位调制器模型（如 PhaseModulator）
        PIM,                    # PIM模式基底 ∈ ℂ^{HW × N_modes}
        Index,           # 目标模式索引
        x,                     # 空间坐标网格 ∈ ℝ^{H × W}
        y,                     # 空间坐标网格 ∈ ℝ^{H × W}
        centers_xy,            # xy局部坐标
        front_num,             # 前端衍射层数
        Lambda,                # 光场波段（单位：米）
        lr=0.01,               # 学习率
        batch_size=8,          # batch 大小
        device='cpu',          # 计算设备
        mask=1,                # 掩模板
        SP=False
    ):
        super().__init__()
        # 模块与参数赋值
        self.modulator = modulator.to(device)
        self.PIM = PIM.detach().to(device)
        self.Index = Index
        self.x = x.detach().to(device)
        self.y = y.detach().to(device)
        self.centers_xy = centers_xy.detach().to(device)
        self.Lambda = Lambda
        self.k = 2 * torch.pi / Lambda
        self.device = device
        self.batch_size = batch_size
        self.mask = mask
        self.dx = (x[0,1]-x[0,0]) #像素x大小
        self.dy = (y[1,0]-y[0,0]) #像素y大小
        self.front = front_num
        self.cell_shape_px = None
        self.OAM_phase = torch.zeros_like(self.x) #默认涡旋光相位全0
        self.H, self.W = self.x.shape
        self.SP = SP

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
            cell_w = torch.tensor(float(self.W), device=xs.device)
    
        # ---- y 方向 ----
        ys_unique = torch.unique(ys)
        ys_unique, _ = torch.sort(ys_unique)
    
        if ys_unique.numel() > 1:
            dy = torch.diff(ys_unique)
            cell_h = torch.median(dy)
        else:
            cell_h = torch.tensor(float(self.H), device=ys.device)
    
        # ---- 转为 int（几何参数，不参与反传） ----
        cell_h = int(torch.round(cell_h).item())
        cell_w = int(torch.round(cell_w).item())
    
        self.cell_shape_px = (cell_h, cell_w)
        return cell_h, cell_w
        
    def generate_microlens_phase(
        self,
        f_list,
        overlap=0.05,
    ):
        N = self.centers_xy.shape[0]
    
        cell_h, cell_w = self.cell_shape_px
        half_h = cell_h * (0.5 + overlap)
        half_w = cell_w * (0.5 + overlap)
    
        yy_px, xx_px = torch.meshgrid(
            torch.arange(self.H, device=self.device),
            torch.arange(self.W, device=self.device),
            indexing="ij"
        )
    
        phase_sum  = torch.zeros((self.H, self.W), device=self.device)
        weight_sum = torch.zeros((self.H, self.W), device=self.device)
    
        for i in range(N):
            cx_px, cy_px = self.centers_xy[i].float()
    
            ux = (xx_px - cx_px) / half_w
            uy = (yy_px - cy_px) / half_h
    
            wx = torch.zeros_like(ux)
            wy = torch.zeros_like(uy)
    
            inside_x = ux.abs() <= 1.0
            inside_y = uy.abs() <= 1.0
    
            wx[inside_x] = 0.5 * (1 + torch.cos(torch.pi * ux[inside_x].abs()))
            wy[inside_y] = 0.5 * (1 + torch.cos(torch.pi * uy[inside_y].abs()))
    
            window = wx * wy
    
            # centers 固定，不参与反传
            cx = self.x[int(cy_px), int(cx_px)]
            cy = self.y[int(cy_px), int(cx_px)]
    
            dx = self.x - cx
            dy = self.y - cy
    
            phase_i = self.k / (2 * f_list[i]) * (dx**2 + dy**2)
    
            phase_sum  += window * phase_i
            weight_sum += window
    
        weight_sum = torch.clamp(weight_sum, min=0.1)
        phase = phase_sum / weight_sum
    
        return phase.unsqueeze(0)
    
    def compute_overlap(self,
        E_out,             # (M, H, W) complex, 模式阵列输出
        modes_ref,         # (H*W, N) complex, 能量归一化参考模式
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

        h, w = self.cell_shape_px
        half_h, half_w = h // 2, w // 2

        # 计算像素物理尺寸
        dx = torch.abs(self.x[0,1] - self.x[0,0])
        dy = torch.abs(self.y[1,0] - self.y[0,0])


        coeffs = torch.zeros((M, N), device=device, dtype=dtype)

        for n in range(N):
            cx, cy = self.centers_xy[n]
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
        
    def forward(self, E_in):
        """
        一次完整前向传播（训练用）：
        输入：
            E_in        ∈ ℂ^{B × H × W}    # 入射光场
            modulator   → M ∈ ℂ^{H × W}    # 相位调制器（复振幅）
            PIM_basis   ∈ ℂ^{HW × N}      # 模态基底

        输出：
            E_in_hat    ∈ ℂ^{B × H × W}    # 重建入射光场
        """
        B, H, W = E_in.shape
        _, N = self.PIM.shape

        # 获取调制器的多个层，相位调制器的输出是一个列表 [num_layers, H, W]
        modulated_layers, d_list = self.modulator()  # [num_layers, H, W]
        f1 = self.modulator.f[0,:] #第一个焦距
        Prop_f1 = self.modulator.Prop_f1 #第一个微透镜传播距离
        Prop_f2 = self.modulator.Prop_f2 #第二个微透镜传播距离
        if self.SP:
            Espher_f1 = (torch.exp(1j*self.k*(self.x**2+self.y**2)/(2*f1))).unsqueeze(0) #单一球面波相位
        else:
            Phase_f1 = self.generate_microlens_phase(f1) #生成卷积相位所需微透镜阵列相位
            Espher_f1 = torch.exp(1j*Phase_f1) #第一个球面波(1, H, W)

        E_mid = E_in.to(self.device)   
        # 输入光场 E_in 被前端衍射层逐层传播，每一层的调制器应用到 E_in
        for i in range(self.front):
            M = modulated_layers[i].to(self.device).reshape(H, W)  # 第 i 层的调制器 转为调制矩阵
            E_mid = E_mid * M
            if i == self.front-1:
                continue
            else:
                E_mid = Rayl_Somm_Diffr(E_mid, self.x, self.y, d_list[i], self.k, self.Lambda, 0, device=self.device)  # 第i到i+1层的传播
                
        
        E_mid = Rayl_Somm_Diffr(E_mid, self.x, self.y, torch.abs(Prop_f1), self.k, self.Lambda, 0, device=self.device)  # 前焦面传播
        #在下一个面上预设微透镜相位或单一透镜相位
        E_mid = E_mid * Espher_f1 #球面波调制
        
        
        for i in range(self.modulator.num_layers-self.front):
            M = modulated_layers[i+self.front].to(self.device).reshape(H, W)  # 第 i 层的调制器 转为调制矩阵
            E_mid = E_mid * M
            if i == self.modulator.num_layers-self.front-1:
                continue
            else:
                if i==self.modulator.num_layers-self.front-2:
                    E_mid = Rayl_Somm_Diffr(E_mid, self.x, self.y, d_list[i+self.front], self.k, self.Lambda, 0, device=self.device)  # 第i到i+1层的传播
                    f= self.modulator.f[i+1,:] #第一个焦距
                    if self.SP: #判断是微透镜阵列相位还是单一球面相位
                        Espher = (torch.exp(-1j*self.k*(self.x**2+self.y**2)/(2*f))).unsqueeze(0) #第二个单一球面相位
                    else:
                        Phase = self.generate_microlens_phase(f) #生成卷积相位所需微透镜阵列相位
                        Espher = torch.exp(-1j*Phase) #第一个球面波(1, H, W)
                    E_mid = E_mid * Espher
                else:
                    E_mid = Rayl_Somm_Diffr(E_mid, self.x, self.y, d_list[i+self.front], self.k, self.Lambda, 0, device=self.device)  # 第i到i+1层的传播

        E_f2_out = Rayl_Somm_Diffr(E_mid, self.x, self.y, torch.abs(Prop_f2), self.k, self.Lambda, 0, device=self.device)  # 输出层到远场传输(B,H,W)

        return E_f2_out #输出像面场 (B, H, W)

    def compute_loss(self, Eout, Ein_para, E_target):
        """
        损失函数：模式重叠积分，先只用一项
        """
            
        with torch.no_grad():  # 确保不会构图
            B, H, W = Eout.shape
            PIM_vec = self.PIM.contiguous().transpose(0, 1).conj() #(M, H*W)
        
        # 模式分布约束
        # Eout = torch.abs(Eout) * torch.exp(1j*torch.angle(Eout)*self.mask)
        E_target = E_target.contiguous().view(B, -1) #(H*W, B)
        Eout_vec = Eout.contiguous().view(B, -1) #(H*W, B)
        # 计算幅度
        amp_out = torch.abs(Eout_vec)  # (M, H*W)
        amp_target = torch.abs(E_target)  # (M, H*W)
        
        # # 计算幅度差
        amp_diff = amp_out - amp_target/2  # (M, H*W)
        # 计算模长差
        # amp_diff = torch.abs(Eout_vec - E_target)  # (M, H*W)
        
        # 计算每个模式的RMS (均方根误差)
        # RMS = sqrt(mean(squared_errors))
        rms_per_mode = torch.sqrt(torch.mean(amp_diff**2, dim=1))  # 形状: (M,)
        
        # 合并M个RMS (求平均或求和)
        loss_distri = torch.sqrt(torch.mean(rms_per_mode**2))  # 平均RMS
        
        # 模式重叠积分约束
        Para_rec = self.compute_overlap(torch.abs(Eout), torch.abs(self.PIM))
        # 重叠积分与系数绝对值差值做loss
        loss_overlap = torch.sqrt(torch.mean(torch.abs(Para_rec-(Ein_para/2))**2))

        
        loss = loss_distri/10 + loss_overlap * 20
        # loss = loss_overlap

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
                E_f2_out = self.forward(E_batch)
                loss = self.compute_loss(E_f2_out, para_batch, E_target_batch)
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
                    
                    E_f2_out = self.forward(E_batch)
                    loss = self.compute_loss(E_f2_out, para_batch, E_target_batch)  # ✅ 参数顺序一致
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

