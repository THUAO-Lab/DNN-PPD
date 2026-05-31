import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseModulator(nn.Module):
    """
    可训练复振幅调制器（相位 + 可选振幅），包括多个衍射层。
    默认使用单位振幅，仅训练相位，范围在 [-π, π]。
    """
    def __init__(
        self, H, W, num_layers=3, N=9, d=1,
        d_min=20e-3, d_max=1,
        f1=1, f2=2, sigma=1.0,
        learnable_amplitude=False,
        freeze_prop=False,
        phase_mapping="sin",
    ):
        super().__init__()
    
        # ========= 保存为 buffer =========
        self.register_buffer("H_buf", torch.tensor(H))
        self.register_buffer("W_buf", torch.tensor(W))
        self.register_buffer("num_layers_buf", torch.tensor(num_layers))
    
        self.H = H
        self.W = W
        self.num_layers = num_layers
        self.sigma = sigma
        self.freeze_prop = freeze_prop  # ✅ 保存开关
        self.phase_mapping = phase_mapping
        if self.phase_mapping not in {"sin", "clamp"}:
            raise ValueError("phase_mapping must be 'sin' or 'clamp'")
    
        # ========= 可训练参数 =========
        self.phases = nn.Parameter(
            2 * torch.zeros(num_layers, H, W)
        )
    
        self.learnable_amplitude = learnable_amplitude
        if learnable_amplitude:
            self.log_amp = nn.Parameter(torch.zeros(H, W))
    
        self.f = nn.Parameter(torch.ones(num_layers, N) * f1)
    
        # =====================================================
        # ✅ Prop_f1 / Prop_f2：根据 freeze_prop 决定类型
        # =====================================================
        if freeze_prop:
            # 👉 不训练（类似 raw_d）
            self.register_buffer("Prop_f1", torch.ones(1) * f1)
            self.register_buffer("Prop_f2", torch.ones(1) * f2)
        else:
            # 👉 参与训练
            self.Prop_f1 = nn.Parameter(torch.ones(1) * f1)
            self.Prop_f2 = nn.Parameter(torch.ones(1) * f2)
    
        # ========= 不训练参数 =========
        raw_d_init = torch.ones(num_layers - 1) * d
        self.register_buffer("raw_d", raw_d_init)
    
        self.register_buffer("d_min_buf", torch.tensor(d_min))
        self.register_buffer("d_max_buf", torch.tensor(d_max))

    @property
    def d_min(self):
        return self.d_min_buf.item()

    @property
    def d_max(self):
        return self.d_max_buf.item()
    
    def gaussian_smooth_phase(self, phase, kernel_size=7):
        """
        对相位做 Gaussian 卷积平滑
    
        phase: [H, W] or [B, H, W]
        sigma: 高斯标准差
        kernel_size: 奇数，如 5,7,9
        """
    
        if phase.dim() == 2:
            phase = phase.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        elif phase.dim() == 3:
            phase = phase.unsqueeze(1)  # [B,1,H,W]
    
        device = phase.device
    
        coords = torch.arange(kernel_size, device=device) - kernel_size // 2
        Y, X = torch.meshgrid(coords, coords, indexing='ij')
    
        kernel = torch.exp(-(X**2 + Y**2) / (2 * self.sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
    
        phase_smooth = F.conv2d(phase, kernel, padding=kernel_size//2)
    
        return phase_smooth.squeeze(1).squeeze(0)

    def map_phase(self, phase):
        """
        Convert the trainable phase parameter to the physical phase.

        Use phase_mapping="sin" for the current parameterization. Use
        phase_mapping="clamp" for older pretrained weights whose phase tensor
        was trained as a directly clamped phase map.
        """
        if self.phase_mapping == "sin":
            return torch.pi * torch.sin(phase)
        if self.phase_mapping == "clamp":
            return torch.clamp(phase, -torch.pi, torch.pi)
        raise ValueError("phase_mapping must be 'sin' or 'clamp'")

    def forward(self, mask = None):
        """
        输出：
            - modulated_layers: List[Tensor[H, W]], 每层复数调制器
            - d_list: Tensor[L-1], 每层之间的传播距离
        """
        modulated_layers = []

        for i in range(self.num_layers):
            phase = self.map_phase(self.phases[i])
            # phase = self.gaussian_smooth_phase(phase, kernel_size=7)
            if mask==None:
                mod = torch.exp(1j * phase)
            else:
                mod = torch.exp(1j * phase) * mask
        
            if self.learnable_amplitude:
                amp = torch.sigmoid(self.log_amp)
                mod *= amp
        
            modulated_layers.append(mod)

        # clamp 传播距离
        d_list = torch.clamp(self.raw_d, self.d_min_buf, self.d_max_buf)

        return modulated_layers, d_list
