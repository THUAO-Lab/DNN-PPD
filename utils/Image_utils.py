import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_func
from matplotlib.widgets import RectangleSelector
from scipy.interpolate import make_interp_spline
import os


def load_image(
    img_path,
    mode="gray",        # "gray" or "rgb"
    show=False,
    resize=None,        # (H, W) or None
    device="cpu"
):
    """
    Load image and return torch uint8 tensor in [0,255].

    Returns
    -------
    img : torch.Tensor (uint8)
        Gray: (H, W)
        RGB : (H, W, 3)
    """

    img = Image.open(img_path)

    if mode == "gray":
        img = img.convert("L")
    elif mode == "rgb":
        img = img.convert("RGB")
    else:
        raise ValueError("mode must be 'gray' or 'rgb'")

    if resize is not None:
        img = img.resize(resize[::-1], Image.BILINEAR)

    img_np = np.asarray(img, dtype=np.uint8)
    img_torch = torch.from_numpy(img_np).to(device)

    if show:
        plt.figure(figsize=(4, 4))
        if mode == "gray":
            plt.imshow(img_np, cmap="gray", vmin=0, vmax=255)
        else:
            plt.imshow(img_np)
        plt.axis("off")
        plt.title(f"{mode.upper()} image (uint8)")
        plt.show()

    return img_torch

def load_image_paths(base_folder="image"):
    rgb_folder = os.path.join(base_folder, "rgb")
    gray_folder = os.path.join(base_folder, "gray")

    def collect_images(folder):
        if not os.path.exists(folder):
            return []
        files = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]
        # 假设是数字命名：1.png, 2.png ...
        files.sort(key=lambda x: int(os.path.splitext(x)[0]))
        return [os.path.join(folder, f) for f in files]

    rgb_images = collect_images(rgb_folder)
    gray_images = collect_images(gray_folder)

    return rgb_images, gray_images

def image_to_bits(img_uint8, n_bits=8):
    """
    Convert gray or RGB image to binary encoding (torch).

    Parameters
    ----------
    img_uint8 : torch.Tensor
        Gray: (H, W)
        RGB : (H, W, 3)
        dtype torch.uint8
    n_bits : int
        Number of bits per channel

    Returns
    -------
    bits : torch.Tensor (uint8)
        Gray: (H*W, n_bits)
        RGB : (H*W, n_bits, 3)
        Values are {0,1}, MSB -> LSB
    """

    assert img_uint8.dtype == torch.uint8

    device = img_uint8.device
    bit_weights = (1 << torch.arange(n_bits - 1, -1, -1, device=device)).to(torch.uint8)
    # e.g. [128, 64, ..., 1]

    # 灰度图
    if img_uint8.dim() == 2:
        H, W = img_uint8.shape
        pixels = img_uint8.reshape(-1, 1)          # (H*W, 1)

        bits = (pixels & bit_weights) > 0           # (H*W, n_bits)
        return bits.to(torch.uint8)

    # RGB 图
    elif img_uint8.dim() == 3:
        H, W, C = img_uint8.shape
        assert C == 3, "RGB image must have 3 channels"

        pixels = img_uint8.reshape(-1, 3)           # (H*W, 3)

        bits = (pixels[:, None, :] & bit_weights[None, :, None]) > 0
        # (H*W, n_bits, 3)

        return bits.to(torch.uint8)

    else:
        raise ValueError("Input must be gray (H,W) or RGB (H,W,3)")
        
def bits_to_image(bits, n_bits=8, H=None, W = None):
    """
    Convert binary encoding back to uint8 image.

    Parameters
    ----------
    bits : torch.Tensor (uint8 or bool)
        Gray: (H*W, n_bits)
        RGB : (H*W, n_bits, 3)
        Values {0,1}, MSB -> LSB

    n_bits : int
        Number of bits per channel

    Returns
    -------
    img_uint8 : torch.Tensor (uint8)
        Gray: (H, W)
        RGB: (H, W, 3)
    """

    device = bits.device
    bit_weights = (1 << torch.arange(n_bits - 1, -1, -1, device=device)).to(torch.uint8)
    # [128, 64, ..., 1]

    if bits.dim() == 2:
        # 灰度图
        H_W, _ = bits.shape
        if H==None:
            H = W = int(H_W**0.5)  # 假设原图是方形，如果非方形，需要额外传入 H, W
        pixels = torch.sum(bits.to(torch.uint8) * bit_weights[None, :], dim=1)
        img_uint8 = pixels.reshape(H, W)
        return img_uint8

    elif bits.dim() == 3:
        # RGB 图
        H_W, n_b, C = bits.shape
        if H==None:
            H = W = int(H_W**0.5)
        assert n_b == n_bits and C == 3
        pixels = torch.sum(bits.to(torch.uint8) * bit_weights[None, :, None], dim=1)  # (H*W, 3)
        img_uint8 = pixels.reshape(H, W, 3)
        return img_uint8

    else:
        raise ValueError("Bits tensor must be (H*W, n_bits) or (H*W, n_bits, 3)")
    

def compute_ssim(img1, img2, show=False, title="SSIM"):
    """
    计算两张图像的 SSIM，并可选显示对比图
    img1, img2: torch.Tensor, 灰度(H,W)或RGB(H,W,3), float或uint8
    """
    
    I1 = img1.cpu().numpy()
    I2 = img2.cpu().numpy()
    
    if I1.ndim == 3 and I1.shape[2] == 3:  # RGB
        s = np.mean([ssim_func(I1[...,c], I2[...,c], data_range=I2.max()-I2.min()) for c in range(3)])
    else:
        s = ssim_func(I1, I2, data_range=I2.max()-I2.min())
    
    if show:
        plt.figure(figsize=(6,3))
        plt.subplot(1,2,1)
        plt.imshow(I1, cmap='gray' if I1.ndim==2 else None)
        plt.title("Original")
        plt.axis('off')
        plt.subplot(1,2,2)
        plt.imshow(I2, cmap='gray' if I1.ndim==2 else None)
        plt.title(f"Recovered\nSSIM={s:.4f}")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    return s

def compute_psnr(img1, img2, max_val=None):
    """
    计算两张图像的 PSNR
    img1, img2: torch.Tensor, 灰度(H,W)或RGB(H,W,3), float或uint8
    max_val: 图像最大值, 若 None 自动取 img2.max()
    """
    I1 = img1.float()
    I2 = img2.float()
    mse = torch.mean((I1 - I2)**2).item()
    if max_val is None:
        max_val = I2.max().item()
    psnr = 10 * np.log10(max_val**2 / (mse + 1e-12))
    return psnr

def interactive_zoom(img):
    """
    左图：原图（鼠标框选）
    右图：放大后的区域（真实灰度 / RGB + 网格 + 像素值）

    Parameters
    ----------
    img : 
        Gray : (H, W)
        RGB  : (H, W, 3)
        numpy array or torch tensor
    """

    # ===== 转 numpy =====
    if hasattr(img, "detach"):
        img = img.detach().cpu().numpy()

    assert img.ndim in (2, 3), "Image must be grayscale or RGB"

    H, W = img.shape[:2]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))

    # ===== 左图：原图 =====
    if img.ndim == 2:
        ax0.imshow(img, cmap="gray", interpolation="nearest")
    else:
        ax0.imshow(img, interpolation="nearest")

    ax0.set_title("Select Region (Drag Mouse)")
    ax0.set_xlim(0, W)
    ax0.set_ylim(H, 0)

    # ===== 右图：占位 =====
    ax1.set_title("Zoomed Region")

    text_handles = []

    # ===== 框选回调 =====
    def onselect(eclick, erelease):
        nonlocal text_handles

        # 清除旧标注
        for t in text_handles:
            t.remove()
        text_handles = []

        if eclick.xdata is None or erelease.xdata is None:
            return

        x0, y0 = int(eclick.xdata), int(eclick.ydata)
        x1, y1 = int(erelease.xdata), int(erelease.ydata)

        x0, x1 = sorted([x0, x1])
        y0, y1 = sorted([y0, y1])

        x0 = np.clip(x0, 0, W - 1)
        x1 = np.clip(x1, 1, W)
        y0 = np.clip(y0, 0, H - 1)
        y1 = np.clip(y1, 1, H)

        patch = img[y0:y1, x0:x1]
        if patch.size == 0:
            return

        h, w = patch.shape[:2]

        # ===== 右图刷新（关键：clear + imshow）=====
        ax1.clear()

        if patch.ndim == 2:
            ax1.imshow(
                patch,
                cmap="gray",
                interpolation="nearest",
                vmin=patch.min(),
                vmax=patch.max()
            )
        else:  # RGB
            ax1.imshow(
                patch,
                interpolation="nearest"
            )

        ax1.set_title(f"Zoomed Region [{w}×{h}]")
        ax1.set_xlim(-0.5, w - 0.5)
        ax1.set_ylim(h - 0.5, -0.5)
        ax1.set_xticks(np.arange(w))
        ax1.set_yticks(np.arange(h))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.grid(True, color="red", linewidth=0.6)

        # ===== 标注像素值 =====
        for iy in range(h):
            for ix in range(w):
                val = patch[iy, ix]

                if patch.ndim == 2:
                    s = f"{val:.2f}"
                else:
                    r, g, b = val
                    s = f"({r:.2f},{g:.2f},{b:.2f})"

                text_handles.append(
                    ax1.text(
                        ix, iy,
                        s,
                        ha="center",
                        va="center",
                        color="yellow",
                        fontsize=9
                    )
                )

        fig.canvas.draw_idle()

    # ===== 矩形选择器 =====
    selector = RectangleSelector(
        ax0,
        onselect,
        useblit=True,
        button=[1],
        minspanx=1,
        minspany=1,
        spancoords="pixels",
        interactive=True
    )

    plt.tight_layout()
    plt.show(block=True)

def plot_fidelity_matrix(
    fidelity_matrix,
    image_names=None,
    title="Image Fidelity Matrix",
    cmap="viridis",
    save_path=None
):
    """
    绘制图片集保真度矩阵

    Parameters
    ----------
    fidelity_matrix : np.ndarray, shape (N, N)
        F[i, j] = fidelity( reconstructed_j , original_i )
    image_names : list[str], optional
        图片名称（用于坐标轴）
    title : str
        图标题
    cmap : str
        colormap
    save_path : str or None
        若不为 None，则保存图片
    """

    fidelity_matrix = np.asarray(fidelity_matrix)
    N = fidelity_matrix.shape[0]

    plt.figure(figsize=(6, 5))
    im = plt.imshow(fidelity_matrix, cmap=cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    if image_names is None:
        image_names = [f"{i+1}" for i in range(N)]

    plt.xticks(range(N), image_names, rotation=45)
    plt.yticks(range(N), image_names)

    plt.xlabel("Reconstructed image index")
    plt.ylabel("Original image index")
    plt.title(title)

    # 在格子里写数值（论文风格）
    for i in range(N):
        for j in range(N):
            plt.text(
                j, i,
                f"{fidelity_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if fidelity_matrix[i, j] < 0.5 else "black",
                fontsize=9
            )

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()

def compute_fidelity_matrix(original_images, reconstructed_images, fidelity_func):
    """
    构造保真度矩阵

    F[i, j] = fidelity( reconstructed_j , original_i )
    """
    N = len(original_images)
    F = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            F[i, j] = fidelity_func(
                reconstructed_images[j],
                original_images[i]
            )
    return F

def plot_mode_boxplot(Para_rec_all, bits, title="Mode-wise Recovery Error"):
    """
    Para_rec_all : (B, N) recovered continuous coefficients
    bits         : (B, 8) or (B, 8, 3) target bits
    """

    # ---------- 目标 bits 展平 ----------
    if bits.dim() == 3:
        bits_flat = bits.reshape(bits.shape[0], -1)
    else:
        bits_flat = bits

    bits_flat = bits_flat.float().cpu().numpy()

    # ---------- 恢复系数 ----------
    Para_real = Para_rec_all.real.cpu().numpy()

    assert Para_real.shape == bits_flat.shape, "Shape mismatch!"

    # ---------- 误差 ----------
    error = Para_real - bits_flat   # (B, N)

    N = error.shape[1]

    # ---------- 箱线图 ----------
    plt.figure(figsize=(0.4 * N + 3, 6))
    bp = plt.boxplot(
        [error[:, k] for k in range(N)],
        showfliers=False
    )

    # ---------- 画须内散点 ----------
    for k in range(N):
        data_k = error[:, k]

        # 计算箱线图统计量
        q1 = np.percentile(data_k, 25)
        q3 = np.percentile(data_k, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        # 只保留须内数据
        valid = data_k[(data_k >= lower) & (data_k <= upper)]

        if valid.size == 0:
            continue

        # 最多取 10 个点
        n_show = min(10, valid.size)
        sample = np.random.choice(valid, size=n_show, replace=False)

        # 画散点（加点横向抖动）
        x = (k + 1) + 0.05 * np.random.randn(n_show)
        plt.scatter(x, sample, color="C1", s=15, alpha=0.8)

    plt.axhline(0, color="r", linestyle="--", linewidth=1)
    plt.xlabel("Mode index")
    plt.ylabel("Recovery error (recovered − target)")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
def plot_crosstalk( bits,
                    Para_rec_all,
                    bins=70,
                    eps=1e-12,
                    title="Statistical Mode Crosstalk Distribution"):
    """
    bits          : (B, 8) or (B, 8, 3)
    Para_rec_all  : (B, N) complex tensor (recovered modal coefficients)
    bins          : histogram bin number
    eps           : avoid divide-by-zero
    """

    # ===============================
    # Step 1: bits reshape
    # ===============================
    if bits.dim() == 3:
        bits_flat = bits.reshape(bits.shape[0], -1)  # (B, 24)
    else:
        bits_flat = bits                              # (B, 8)

    bits_flat = bits_flat.float().to(Para_rec_all.device)

    # ===============================
    # Step 2: 功率计算
    # ===============================
    Para_real = Para_rec_all.real          # (B, N)
    Para_power = Para_real ** 2            # 功率

    signal_mask = bits_flat == 1
    crosstalk_mask = bits_flat == 0

    signal_power = (Para_power * signal_mask).sum(dim=1)       # (B,)
    crosstalk_power = (Para_power * crosstalk_mask).sum(dim=1) # (B,)

    # ===============================
    # Step 3: Crosstalk (dB)
    # ===============================
    XT_dB = 10 * torch.log10(crosstalk_power / (signal_power + eps))
    XT_dB = XT_dB.detach().cpu().numpy()

    # 去掉 inf / nan
    XT_dB = XT_dB[np.isfinite(XT_dB)]

    # ===============================
    # Step 4: 绘图
    # ===============================
    plt.figure(figsize=(8, 5))

    counts, bins_edge, _ = plt.hist(
        XT_dB,
        bins=bins,
        edgecolor="k",
        alpha=0.7,
        label="Pixel counts"
    )

    # ---------- 平滑拟合 ----------
    bin_centers = 0.5 * (bins_edge[1:] + bins_edge[:-1])

    if len(bin_centers) > 5:  # 防止样本过少
        spl = make_interp_spline(bin_centers, counts, k=3)
        x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 500)
        y_smooth = spl(x_smooth)
        plt.plot(x_smooth, y_smooth, 'r-', linewidth=2, label="Smooth fit")

    plt.xlabel("Crosstalk (dB)")
    plt.ylabel("Pixel count")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ===============================
    # Step 5: 统计量输出
    # ===============================
    print(f"Mean XT   : {XT_dB.mean():.2f} dB")
    print(f"Median XT : {np.median(XT_dB):.2f} dB")
    print(f"Best XT   : {XT_dB.min():.2f} dB")
    print(f"Worst XT  : {XT_dB.max():.2f} dB")

    return XT_dB

def plot_recovery_error(Para_rec,
                        Para_target,
                        bins=70,
                        eps=1e-12,
                        title="Statistical Recovery Coefficient Error"):
    """
    Para_rec     : (B, N) complex tensor, recovered coefficients
    Para_target  : (B, N) complex tensor, target coefficients
    bins         : histogram bin number
    eps          : avoid divide-by-zero
    """

    # ===============================
    # Step 1: 取幅值
    # ===============================
    rec_amp = torch.abs(Para_rec)
    tgt_amp = torch.abs(Para_target)

    # ===============================
    # Step 2: 相对误差
    # ===============================
    error = torch.abs(rec_amp - tgt_amp) / (tgt_amp + eps)

    # 拉平成一维
    error_flat = error.reshape(-1).detach().cpu().numpy()

    # ===============================
    # Step 3: 去除异常值
    # ===============================
    # 去除 nan / inf
    error_flat = error_flat[np.isfinite(error_flat)]

    # 只去除明显异常（误差 > 6）
    error_flat = error_flat[error_flat <= 3]

    # ===============================
    # Step 4: 绘图（log-binned histogram）
    # ===============================
    plt.figure(figsize=(8, 5))
    
    # --- log-spaced bins ---
    min_err = max(error_flat.min(), 1e-6)   # 避免 log(0)
    max_err = error_flat.max()
    
    log_bins = np.logspace(
        np.log10(min_err),
        np.log10(max_err),
        bins
    )
    
    counts, bins_edge, _ = plt.hist(
        error_flat,
        bins=log_bins,
        edgecolor="k",
        alpha=0.7,
        label="Coefficient count"
    )
    
    # ---------- 平滑拟合 ----------
    bin_centers = np.sqrt(bins_edge[:-1] * bins_edge[1:])  # log 中心
    
    if len(bin_centers) > 5:
        spl = make_interp_spline(bin_centers, counts, k=3)
        x_smooth = np.logspace(
            np.log10(bin_centers.min()),
            np.log10(bin_centers.max()),
            500
        )
        y_smooth = spl(x_smooth)
        plt.plot(x_smooth, y_smooth, 'r-', linewidth=2, label="Smooth fit")
    
    plt.xscale("log")
    
    plt.xlabel("Relative amplitude error")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ===============================
    # Step 5: 统计量输出
    # ===============================
    print(f"Mean error   : {error_flat.mean():.4e}")
    print(f"Median error : {np.median(error_flat):.4e}")
    print(f"Best error   : {error_flat.min():.4e}")
    print(f"Worst error  : {error_flat.max():.4e}")

    return error_flat

def plot_rec_vs_target_bar(Para_rec,
                           Para_target,
                           num_show=10,
                           seed=None,
                           title="Recovered vs Target Coefficients"):
    """
    随机选取若干样本，画恢复系数与目标系数的柱状对比图

    参数
    ----------
    Para_rec : torch.Tensor
        恢复的系数，形状 (B, N) 或 (B,)
    Para_target : torch.Tensor
        目标系数，形状 (B, N) 或 (B,)
    num_show : int
        要展示的样本数量
    seed : int or None
        随机种子（None 表示每次随机）
    title : str
        图标题
    """

    # ===============================
    # Step 1: 取幅值并拉平成一维
    # ===============================
    rec = torch.abs(Para_rec).detach().cpu().numpy().reshape(-1)
    tgt = torch.abs(Para_target).detach().cpu().numpy().reshape(-1)

    assert len(rec) == len(tgt), "Recovered 和 Target 长度不一致"

    # ===============================
    # Step 2: 随机选样本
    # ===============================
    if seed is not None:
        np.random.seed(seed)

    num_show = min(num_show, len(rec))
    idx = np.random.choice(len(rec), size=num_show, replace=False)

    rec_show = rec[idx]
    tgt_show = tgt[idx]

    # ===============================
    # Step 3: 画柱状图
    # ===============================
    x = np.arange(num_show)
    width = 0.35

    plt.figure(figsize=(10, 5))

    plt.bar(
        x - width / 2,
        tgt_show,
        width=width,
        color='tab:blue',
        alpha=0.8,
        label='Target coefficient'
    )

    plt.bar(
        x + width / 2,
        rec_show,
        width=width,
        color='tab:orange',
        alpha=0.8,
        label='Recovered coefficient'
    )

    plt.xlabel("Sample index")
    plt.ylabel("Coefficient magnitude")
    plt.title(title)

    plt.xticks(x, [f"{i+1}" for i in range(num_show)])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()