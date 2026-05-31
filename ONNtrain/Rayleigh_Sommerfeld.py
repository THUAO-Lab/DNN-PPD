import torch
import torch.nn.functional as F


def Rayl_Somm_Diffr(Ein, x, y, z, k, Lambda, numpad=None, device='cpu'):
    """
    Batch-compatible ASM + Rayleigh–Sommerfeld propagator
    
    Parameters
    ----------
    Ein : (B,H,W) complex
    x,y : (H,W)
    z   : scalar tensor
    """
    Ein = Ein.to(device)
    x = x.to(device)
    y = y.to(device)
    z = torch.as_tensor(z, dtype=torch.float32, device=device)
    Lambda = torch.as_tensor(Lambda, dtype=torch.float32, device=device)
    k = torch.as_tensor(k, dtype=torch.float32, device=device)

    B, H, W = Ein.shape
    dx = (x[0,1] - x[0,0]).abs()
    dy = (y[1,0] - y[0,0]).abs()
    dmin = torch.minimum(dx, dy)

    # ==============================
    # 临界传播距离
    # ==============================
    cond = 1 - (Lambda / (2*dmin))**2
    if cond < 0:
        Zc = torch.tensor(0., device=device)
    else:
        Zc = 2 * max(H, W) * dmin**2 / Lambda * torch.sqrt(cond)

    # ==============================
    # 动态 padding
    # ==============================
    if torch.abs(z) < Zc:
        Np = torch.round(Lambda * torch.abs(z) / (2*dmin**2) / torch.sqrt(torch.clamp(cond, min=1e-12)))
    else:
        Np = torch.tensor(max(H, W), device=device)+numpad

    if (Np % 2) == 1:
        Np = Np + 1

    pad_size = int(Np.item() // 2)

    Ein_pad = F.pad(Ein, (pad_size, pad_size, pad_size, pad_size))

    Hpad = H + int(Np)
    Wpad = W + int(Np)

    # ==============================
    # 频率坐标
    # ==============================
    fx = torch.linspace(-1/(2*dx), 1/(2*dx) - 1/(Hpad*dx), Hpad, device=device)
    fy = torch.linspace(-1/(2*dy), 1/(2*dy) - 1/(Wpad*dy), Wpad, device=device)
    fx, fy = torch.meshgrid(fx, fy, indexing='ij')

    # ==============================
    # ===== Angular Spectrum =====
    # ==============================
    if torch.abs(z) < Zc:
        root = 1 - (Lambda*fx)**2 - (Lambda*fy)**2
        root = torch.clamp(root, min=0)  # 去 evanescent
        
        H_prop = torch.exp(1j * k * z * torch.sqrt(root))
        H_prop = torch.complex(
            torch.nan_to_num(H_prop.real, nan=0.0, posinf=0.0, neginf=0.0),
            torch.nan_to_num(H_prop.imag, nan=0.0, posinf=0.0, neginf=0.0),
        )

        FEin = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(Ein_pad, dim=(-2,-1))), dim=(-2,-1))
        Eout_pad = torch.fft.ifftshift(
            torch.fft.ifft2(torch.fft.fftshift(FEin * H_prop, dim=(-2,-1))),
            dim=(-2,-1)
        )

    # ==============================
    # ===== Rayleigh–Sommerfeld =====
    # ==============================
    else:
        Lx = (-2*torch.min(x)).item()
        Ly = (-2*torch.min(y)).item()

        xp = torch.linspace(-Lx, Lx-dx, Hpad, device=device)
        yp = torch.linspace(-Ly, Ly-dy, Wpad, device=device)
        xp, yp = torch.meshgrid(xp, yp, indexing='ij')

        r = torch.sqrt(xp**2 + yp**2 + z**2)
        h = (1/(2*torch.pi)) * z/r * (1/r - 1j*k) * torch.exp(1j*k*r) / r

        Hh = torch.fft.fft2(torch.fft.fftshift(h))
        FEin = torch.fft.fft2(torch.fft.fftshift(Ein_pad, dim=(-2,-1)))
        Eout_pad = torch.fft.ifftshift(
            torch.fft.ifft2(FEin * Hh),
            dim=(-2,-1)
        ) * dx * dy

    # ==============================
    # 裁剪
    # ==============================
    s = pad_size
    eH = -pad_size if pad_size != 0 else None
    eW = -pad_size if pad_size != 0 else None
    Eout = Eout_pad[:, s:eH, s:eW]

    return Eout