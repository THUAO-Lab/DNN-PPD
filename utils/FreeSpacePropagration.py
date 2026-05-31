import torch
import numpy as np

def FreeSpacePropagrationFFT(Ein, x, y, z, k=0, Lambda=633e-6, numpad=2, device = 'cpu'):
    """优化后的传播函数，分离PyTorch和NumPy实现路径"""
    if isinstance(Ein, torch.Tensor):
        return _pytorch_impl(Ein, x, y, z, k, Lambda, torch.tensor(numpad), device = device)
    else:
        return _numpy_impl(Ein, x, y, z, k, Lambda, numpad)

def _pytorch_impl(Ein, x, y, z, k, Lambda, numpad, device = 'cpu'):
    """PyTorch专用实现"""
    # 参数强制实数化
    def enforce_real(var):
        if isinstance(var, torch.Tensor) and var.is_complex():
            return var.real
        return var
    
    # === 转移到设备 ===
    Ein = Ein.to(device)
    x = x.to(device)
    y = y.to(device)
    z = z.to(device)
    k = k.to(device)
    Lambda = Lambda.to(device)
    numpad = numpad.to(device)
    # === 传播条件计算 ===
    device = Ein.device 
    
    Nx = Ein.shape[0]
    Ny = Ein.shape[1]
    
    dx = (x[0,1] - x[0,0]).item()
    dy = (y[1,0] - y[0,0]).item()
    
    # 取最小采样间隔保证稳定性
    dmin = min(dx, dy)
    
    if 1 - (Lambda / 2 / dmin)**2 < 0:
        Zc = 0
    else:
        Zc = 2 * max(Nx, Ny) * dmin**2 / Lambda * torch.sqrt(1 - (Lambda / 2 / dmin)**2)
    
    # === 动态填充 ===
    if torch.abs(z) < Zc:
        Np = torch.round(Lambda * torch.abs(z) / 2 / dmin**2 / torch.sqrt(1 - (Lambda / 2 / dmin)**2))
    else:
        Np = torch.tensor(max(Nx, Ny), device=device)
    
    if Np % 2 == 1:
        Np = Np + 1
    
    pad_size = int(Np.item() // 2)
    
    # === 执行传播 ===
    Ein_Pad = torch.nn.functional.pad(Ein, (pad_size, pad_size, pad_size, pad_size))
    
    Nx_pad = Nx + int(Np)
    Ny_pad = Ny + int(Np)
    
    fx = torch.linspace(-1/(2*dx), 1/(2*dx) - 1/(Nx_pad*dx), Nx_pad, device=device)
    fy = torch.linspace(-1/(2*dy), 1/(2*dy) - 1/(Ny_pad*dy), Ny_pad, device=device)
    
    fx, fy = torch.meshgrid(fx, fy, indexing='ij')

    if torch.abs(z) < Zc:  # 角谱法
        H = torch.exp(1j * k * z * torch.sqrt(1 - (Lambda*fx)**2 - (Lambda*fy)**2)) # 避免nan
        H = torch.nan_to_num(torch.real(H), nan=0.0)+1j*torch.nan_to_num(torch.imag(H), nan=0.0)
        FEin_pad = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(Ein_Pad)))
        Eout_Pad = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fftshift(FEin_pad * H)))
    else:       # 瑞利-索末菲
        L = (-2*torch.min(torch.min(x))).item()
        xp,yp = torch.meshgrid(torch.linspace(-L,L-dx,Nx_pad), torch.linspace(-L,L-dy,Ny_pad))
        xp = xp.to(device)
        yp = yp.to(device)
        r = torch.sqrt(xp**2 + yp**2 + z**2)
        h = 1/2/torch.pi*z/r*(1./r-1j*k)*torch.exp(1j*k*r)/r #impulse
        Eout_Pad = torch.fft.ifftshift(torch.fft.ifft2(torch.fft.fft2(torch.fft.fftshift(Ein_Pad)) *torch.fft.fft2(torch.fft.fftshift(h))))*dx**2

    # === 裁剪输出 ===
    start = pad_size
    end = -pad_size if pad_size != 0 else None
    Eout = Eout_Pad[start:end, start:end]
    return Eout

def _numpy_impl(Ein, x, y, z, k, Lambda, numpad):
    """NumPy专用实现""" 
    num=Ein.shape[0]
    dx=x[0,1]-x[0,0]
    if 1-(Lambda/2/dx)**2<0:
        Zc=0
    else:
        Zc=2*num*dx**2/Lambda*np.sqrt(1-(Lambda/2/dx)**2)
    
    if np.abs(z)<Zc:
        Np=int(np.around(Lambda*np.abs(z)/2/dx**2/np.sqrt(1-(Lambda/2/dx)**2)))
    else:
        Np = num
    if Np%2 == 1:
        Np = Np + 1
    
    Ein_Pad=np.pad(Ein,(Np//2, Np//2))
    n=num+Np
    fx,fy = np.meshgrid(np.linspace(-1/2/dx,1/2/dx-1/n/dx,n), np.linspace(-1/2/dx,1/2/dx-1/n/dx,n))
    if np.abs(z) < Zc:
        H = np.exp(1j * k * z * np.sqrt(1 - (Lambda*fx)**2 - (Lambda*fy)**2)) # 避免nan
        H[np.isnan(H)] = 0 
        FEin_pad = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Ein_Pad)))
        Eout_Pad = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(FEin_pad * H)))
    else:
        L = -2*np.min(np.min(x))
        xp,yp = np.meshgrid(np.linspace(-L,L-dx,n), np.linspace(-L,L-dx,n))
        r = np.sqrt(xp**2 + yp**2 + z**2)
        h = 1/2/np.pi*z/r*(1./r-1j*k)*np.exp(1j*k*r)/r #impulse
        Eout_Pad = np.fft.ifftshift(np.fft.ifft2(np.fft.fft2(np.fft.fftshift(Ein_Pad)) *np.fft.fft2(np.fft.fftshift(h))))*dx**2
        
    pad_size = Np // 2    
    start = pad_size
    end = -pad_size if pad_size !=0 else None
    Eout = Eout_Pad[start:end, start:end]
    return Eout
