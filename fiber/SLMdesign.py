import torch
import torch.fft
import matplotlib.pyplot as plt


class SLMSimulator:
    def __init__(self, Nx=1920, Ny=1200, dx=8000e-6, wavelength=633e-6, Et = None, device="cuda"):
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
        self.Et = Et
        self.device = device

        # 坐标网格
        x = torch.arange(-Nx/2, Nx/2, 1, device=device) * dx * 2
        y = torch.arange(-Ny/2, Ny/2, 1, device=device) * dx * 2
        self.x, self.y = torch.meshgrid(x, y, indexing="xy")

    def generate_pattern(self, U, show=True):
        """
        生成 SLM pattern (棋盘式编码)
        """
        Uamp = torch.abs(U)
        Upha = torch.angle(U)
        Uamp = Uamp / Uamp.max()

        alpha = torch.acos(Uamp)
        A = Upha + alpha
        B = Upha - alpha

        SLMpattern = torch.zeros((self.Ny, self.Nx), device=self.device)
        SLMpattern[0::2, 0::2] = A
        SLMpattern[1::2, 1::2] = A
        SLMpattern[0::2, 1::2] = B
        SLMpattern[1::2, 0::2] = B
        
        # if show:
        #     plt.figure(figsize=(8,5))
        #     plt.imshow((SLMpattern.cpu().numpy() + torch.pi) / (2*torch.pi), cmap="gray")
        #     plt.colorbar(label="Normalized Phase")
        #     plt.title("SLM Pattern")
        #     plt.axis("off")
        #     plt.show()

        return SLMpattern
    def blaze_phase(self, period_pixels=1, direction='x', dtype=torch.float32):
        """
        生成闪耀光栅相位分布
        
        参数:
        ----------
        period_pixels : float
            闪耀光栅周期（像素数）
        direction : str, 可选
            光栅方向: 'horizontal', 'vertical', 或角度字符串如 '45deg'
        dtype : torch.dtype, 可选
            输出张量数据类型
        device : str, 可选
            计算设备
        
        返回:
        ----------
        phase_pattern : torch.Tensor
            闪耀光栅相位图，形状为(height, width)
        """
        # 创建坐标网格
        y, x = torch.meshgrid(
            torch.arange(self.Ny, dtype=dtype, device=self.device),
            torch.arange(self.Nx, dtype=dtype, device=self.device),
            indexing='ij'
        )
        # 根据方向计算相位斜坡
        if direction == 'x':
            # 水平方向闪耀光栅
            phase_slope = x / period_pixels * (2*torch.pi)
        elif direction == 'y':
            # 垂直方向闪耀光栅
            phase_slope = y / period_pixels * (2*torch.pi)
        elif 'deg' in direction:
            # 斜方向闪耀光栅，例如'45deg'
            try:
                angle = float(direction.replace('deg', ''))
            except:
                raise ValueError(f"无法解析角度方向: {direction}")
            
            # 将角度转换为弧度
            angle_rad = angle * torch.pi / 180.0
            
            # 计算斜方向相位斜坡
            phase_slope = (x * torch.cos(angle_rad) + y * torch.sin(angle_rad)) / period_pixels * (2*torch.pi)
        else:
            raise ValueError(f"不支持的direction参数: {direction}。请选择'horizontal', 'vertical', 或角度如'45deg'")
        
        # 应用模运算得到锯齿波相位
        phase_pattern = torch.remainder(phase_slope, (2*torch.pi))
        
        return phase_pattern

    def simulate(self, beam_radius = 3, X=None, Y=None, Ein = None, show=True, Shining=False, period_pixels = 20, symbol=1):
        """
        主流程：生成输入光场，SLM调制，再传播
        """
        # 光阑裁剪
        if X==None:
            xi = torch.arange(-self.Nx/2, self.Nx/2, device=self.device) * self.dx
            yi = torch.arange(-self.Ny/2, self.Ny/2, device=self.device) * self.dx
            Xi, Yi = torch.meshgrid(xi, yi, indexing="xy")
        else:
            Xi = X
            Yi = Y
        mask = (Xi**2 + Yi**2 <= beam_radius**2)
        U = self.Et
        Chessboard = symbol*self.generate_pattern(U, show = show)
        Chessboard[~mask] = 0
        if Shining:
            Shining_grating = self.blaze_phase(period_pixels=period_pixels)
        else:
            Shining_grating = torch.zeros_like(Chessboard)
        pattern = Chessboard + torch.angle(torch.exp(1j * Shining_grating))
        # plt.figure(figsize=(8,5))
        # plt.imshow((pattern.cpu().numpy()), cmap="gray")
        # plt.colorbar(label="Normalized Phase")
        # plt.title("SLM Pattern")
        # plt.axis("off")
        # plt.show()

        # 输入场
        Ei = torch.zeros((self.Ny, self.Nx), dtype=torch.complex64, device=self.device)
        if Ein==None:
            # 计算纵向居中位置
            start_row = (Ei.shape[0] - pattern.shape[0]) // 2
            end_row = start_row + pattern.shape[0]
            Ei[start_row:end_row, :] = torch.exp(1j * pattern)
        else:
            start_row = (Ei.shape[0] - pattern.shape[0]) // 2
            end_row = start_row + pattern.shape[0]
            Ei[start_row:end_row, :] = Ein*torch.exp(1j * pattern)        

        # Ei=Ei*mask
        Ei[~mask] = 0
        if show:
            plt.figure(figsize=(8,5))
            plt.imshow((pattern.cpu().numpy() + torch.pi) / (2*torch.pi), cmap="gray")
            plt.colorbar(label="Normalized Phase")
            plt.title("SLM Pattern")
            plt.axis("off")
            plt.show()

        return pattern, Ei
