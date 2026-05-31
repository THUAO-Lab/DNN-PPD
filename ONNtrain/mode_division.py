import torch
import matplotlib.pyplot as plt
import utils as ut
from ONNtrain import Rayl_Somm_Diffr

@torch.no_grad()
def run_inference(
    E_in,             # 输入光场: torch.Tensor [1, H, W] (复数)
    modulator,             # 训练好的调制器模型
    PIM,                   # 模式基底 ∈ ℂ^{HW × N_modes}
    x, y,                  # 网格坐标 ∈ ℝ^{H × W}
    Lambda,              # 波长
    mask = 1,             # 掩膜     
    trainer=None          #训练器
):
    trainer.modulator.train() #模型测试
    E_f2_out = trainer.forward(E_in)
    return E_f2_out[0] #输出像面场 (B, H, W)
    # device = E_in.device
    # B, H, W = E_in.shape
    # _, N = PIM.shape    
    # k = 2*torch.pi/Lambda

    # # 获取调制器的多个层，相位调制器的输出是一个列表 [num_layers, H, W]
    # modulated_layers, d_list = modulator()  # [num_layers, H, W]
    # f1 = modulator.f1 #球面波焦距
    # f2 = modulator.f2 #球面波焦距


    # # 输入光场 E_in 被逐层传播，每一层的调制器应用到 E_in
    # E_mid = E_in.to(device)
    # for i in range(modulator.num_layers):
    #     M = modulated_layers[i].to(device).reshape(H, W)  # 第 i 层的调制器 转为调制矩阵
    #     E_mid = E_mid * M
    #     E_mid = Rayl_Somm_Diffr(E_mid, x, y, d_list[i], k, Lambda, 0, device=device)  # 第i到i+1层的传播

    # #生成球面波
    # #瑞利索末菲传播
    # E_f1 = Rayl_Somm_Diffr(E_mid, x, y, torch.abs(f1), k, Lambda, 0, device=device)  # 输出层到远场传输
    # Espher_f1 = (torch.exp(-1j*k*(x**2+y**2)/(2*f1))).unsqueeze(0) #第一个球面波(1, H, W)
    # E_f1_out = E_f1 * Espher_f1
    
    # #生成第二个球面波
    # Espher_f2 = (torch.exp(-1j*k*(x**2+y**2)/(2*f2))).unsqueeze(0) #第一个球面波(1, H, W)
    # E_f2_in = E_f1_out * Espher_f2
    # #瑞利索末菲传播
    # E_f2_out = Rayl_Somm_Diffr(E_f2_in, x, y, torch.abs(f2), k, Lambda, 0, device=device)  # 输出层到远场传输(B,H,W)

    # return E_f2_out #输出像面场 (B, H, W)


def visualize_recovery(x, y, E_in, E_rec, save_path=None):
    """
    可视化恢复结果
    """
    B,H,W = E_in.shape
    for i in range(B):
        ut.figComplexField(x, y, E_in[i,:,:], 1, string=f'input_{i}') #入射场
        ut.figComplexField(x, y, E_rec[i,:,:], 1, string=f'recovery_{i}') #恢复场
    if save_path:
        plt.savefig(save_path, dpi=600)
        
def plot_rms_error(E_in, E_rec, save_path=None, Type = 'complex'):
    """
    绘制预测光场与真实光场之间的 RMS 误差统计图。

    参数:
        E_in:  [B, H, W] torch.complex — 原始光场
        E_rec: [B, H, W] torch.complex — 恢复光场
        save_path: 可选字符串，若指定则保存图像为文件（自动加 .png）

    返回:
        RMS_list: list[float] — 每个样本的 RMS 误差
    """
    B, H, W = E_in.shape
    RMS_list = []

    for i in range(B):
        if Type == 'complex':
            diff = E_in[i] - E_rec[i]
            if Type == 'amp':
                diff = torch.abs(E_in[i]) - torch.abs(E_rec[i])
            else:
                diff = torch.angle(E_in[i]) - torch.angle(E_rec[i])
        rms = torch.sqrt(torch.mean(torch.abs(diff) ** 2)).item()
        RMS_list.append(rms)

    # 绘图
    plt.figure(figsize=(6, 4))
    plt.plot(RMS_list, 'ko', label='RMS Error')  # 黑色圆点
    max_rms = max(RMS_list)
    plt.axhline(y=max_rms, color='r', linestyle='--', label=f'Max RMS = {max_rms:.4e}')
    plt.xlabel("Sample Index")
    plt.ylabel("RMS Error")
    plt.title("RMS Error between Predicted and Ground Truth Fields")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(f"{save_path}.png", dpi=600)

    plt.show()
    return RMS_list

