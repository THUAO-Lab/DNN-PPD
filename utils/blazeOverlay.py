import torch
import matplotlib.pyplot as plt
class BlazeOverlay:
    """
    在任意输入图案上叠加闪耀光栅相位
    """

    def __init__(self, Ny, Nx, dx=8000e-6, wavelength=633e-6, device="cuda"):
        """
        Nx, Ny: SLM 像素数
        dx: 像素间距
        Et: 目标光场
        wavelength: 波长
        """
        self.Nx = Nx
        self.Ny = Ny
        self.dx = dx
        self.lambda_ = wavelength
        self.k = 2 * torch.pi / self.lambda_
        self.device = device

        # 坐标网格
        x = torch.arange(-Nx/2, Nx/2, 1, device=device) * dx * 2
        y = torch.arange(-Ny/2, Ny/2, 1, device=device) * dx * 2
        self.x, self.y = torch.meshgrid(x, y, indexing="xy")

    #生成闪耀光栅相位
    def generate_blaze_phase(self,
                             period_pixels=32,
                             direction='x',
                             dtype=torch.float32):
        """
        生成闪耀光栅相位分布
        """
    
        # 创建坐标网格
        x, y = torch.meshgrid(
            torch.arange(self.Ny, dtype=dtype, device=self.device),
            torch.arange(self.Nx, dtype=dtype, device=self.device),
            indexing='ij'
        )
    
        # ============================
        # 新增：判断是否反向
        # ============================
        sign = 1
    
        if direction == '-x':
            sign = -1
            direction = 'x'
    
        elif direction == '-y':
            sign = -1
            direction = 'y'
    
        # ============================
        # 原有代码逻辑（完全保持）
        # ============================
    
        # 根据方向计算相位斜坡
        if direction == 'x':
            # 水平方向闪耀光栅
            phase_slope = sign * x / period_pixels * (2*torch.pi)
    
        elif direction == 'y':
            # 垂直方向闪耀光栅
            phase_slope = sign * y / period_pixels * (2*torch.pi)
    
        elif 'deg' in direction:
            # 斜方向闪耀光栅，例如'45deg'
            try:
                angle = float(direction.replace('deg', ''))
            except:
                raise ValueError(f"无法解析角度方向: {direction}")
            
            # 将角度转换为弧度
            angle_rad = angle * torch.pi / 180.0
            
            # 计算斜方向相位斜坡
            phase_slope = sign * (x * torch.cos(angle_rad) + y * torch.sin(angle_rad)) / period_pixels * (2*torch.pi)
    
        else:
            raise ValueError(f"不支持的direction参数: {direction}。请选择'horizontal', 'vertical', 或角度如'45deg'")
        
        # 应用模运算得到锯齿波相位
        blaze_phase = torch.remainder(phase_slope, (2*torch.pi))
        
        return blaze_phase    

    # 在任意图案上叠加闪耀相位
    def apply_blaze(self,
                    input_pattern,
                    period_pixels=32,
                    direction='x',show=True):
        """
        在输入图案上叠加闪耀光栅相位

        参数:
        ----------
        input_pattern : torch.Tensor
            可以是:
                (H, W) 实数相位图
                (H, W) 复数光场
                (B, H, W) 批量复场
        period_pixels : float
            闪耀周期（像素）
        direction : str
            光栅方向

        返回:
        ----------
        output_pattern : torch.Tensor
        Ei : torch.Tensor
        """

        blaze_phase = self.generate_blaze_phase(
            period_pixels=period_pixels,
            direction=direction,
            dtype=torch.float32
        )

        # 扩展 batch 维
        if input_pattern.dim() == 3:
            blaze_phase = blaze_phase.unsqueeze(0)

        # ---------------------------
        # 情况 1：输入是复场
        # ---------------------------
        if torch.is_complex(input_pattern):

            # output = input_pattern * torch.exp(1j * blaze_phase)
            output = (torch.angle(input_pattern) + blaze_phase).squeeze(0) #输出为相位分布，注意不是复振幅分布

        # ---------------------------
        # 情况 2：输入是纯相位图
        # ---------------------------
        else:
            output = (input_pattern + blaze_phase).squeeze(0)
            
        # 形成复振幅场
        # 输入场
        Ei = torch.zeros((self.Ny, self.Nx), dtype=torch.complex64, device=self.device)
        # 计算纵向居中位置
        start_row = (Ei.shape[0] - output.shape[0]) // 2
        end_row = start_row + output.shape[0]
        Ei[start_row:end_row, :] = torch.exp(1j * output)
        if show:
            plt.figure(figsize=(8,5))
            plt.imshow((output.detach().cpu().numpy() + torch.pi) / (2*torch.pi), cmap="gray")
            plt.colorbar(label="Normalized Phase")
            plt.title("SLM Pattern")
            plt.axis("off")
            plt.show()

        return output, Ei