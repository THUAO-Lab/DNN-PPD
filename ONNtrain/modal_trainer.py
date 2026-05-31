import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from ONNtrain import Rayl_Somm_Diffr
import matplotlib.pyplot as plt
import os


class Modal_Trainer(nn.Module):
    """Trainer for the single-mode ONN modal transfer model."""

    def __init__(
        self,
        modulator,
        PIM,
        Index,
        x,
        y,
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
        self.Lambda = Lambda
        self.k = 2 * torch.pi / Lambda
        self.device = device
        self.batch_size = batch_size
        self.mask = mask
        self.front = front_num
        self.dx = x[0, 1] - x[0, 0]
        self.dy = y[1, 0] - y[0, 0]

        self.optimizer = torch.optim.Adam(list(self.modulator.parameters()), lr=lr)
        
    def forward(self, E_in):
        """Propagate a batch through the single-mode optical stack."""
        _, H, W = E_in.shape

        modulated_layers, d_list = self.modulator()
        f1 = self.modulator.f[0, :]
        Prop_f1 = self.modulator.Prop_f1
        Prop_f2 = self.modulator.Prop_f2
        Espher_f1 = torch.exp(1j * self.k * (self.x**2 + self.y**2) / (2 * f1)).unsqueeze(0)

        E_mid = E_in.to(self.device)      
        for i in range(self.front):
            M = modulated_layers[i].to(self.device).reshape(H, W)
            E_mid = E_mid * M
            if i == self.front - 1:
                continue
            else:
                E_mid = Rayl_Somm_Diffr(E_mid, self.x, self.y, d_list[i], self.k, self.Lambda, 0, device=self.device)
                
        
        E_mid = Rayl_Somm_Diffr(E_mid, self.x, self.y, torch.abs(Prop_f1), self.k, self.Lambda, 0, device=self.device)
        E_mid = E_mid * Espher_f1
        
        
        for i in range(self.modulator.num_layers - self.front):
            M = modulated_layers[i + self.front].to(self.device).reshape(H, W)
            E_mid = E_mid * M
            if i == self.modulator.num_layers - self.front - 1:
                continue
            else:
                if i == self.modulator.num_layers - self.front - 2:
                    E_mid = Rayl_Somm_Diffr(E_mid, self.x, self.y, d_list[i + self.front], self.k, self.Lambda, 0, device=self.device)
                    f = self.modulator.f[i + 1, :]
                    Espher = torch.exp(-1j * self.k * (self.x**2 + self.y**2) / (2 * f)).unsqueeze(0)
                    E_mid = E_mid * Espher
                else:
                    E_mid = Rayl_Somm_Diffr(E_mid, self.x, self.y, d_list[i + self.front], self.k, self.Lambda, 0, device=self.device)

        
        E_f2_out = Rayl_Somm_Diffr(E_mid, self.x, self.y, torch.abs(Prop_f2), self.k, self.Lambda, 0, device=self.device)

        return E_f2_out

    def compute_loss(self, Eout, Ein_para):
        """Overlap-integral loss for the selected single-mode channel."""
            
        with torch.no_grad():
            B = Eout.shape[0]
            PIM_vec = self.PIM.contiguous().transpose(0, 1).conj()
            Ein_para_vex = Ein_para.contiguous().transpose(0, 1)
        
        para_target = Ein_para_vex[self.Index, :]
        Eout = torch.abs(Eout) * torch.exp(1j * torch.angle(Eout) * self.mask)
        Eout_vec = Eout.contiguous().view(B, -1).transpose(0, 1)
        overlap = torch.matmul(torch.abs(PIM_vec), torch.abs(Eout_vec) * self.dx * self.dy)
        para_overlap = overlap[self.Index, :]
        loss_overlap = torch.sqrt(torch.mean(torch.abs(para_overlap - (para_target / 2))**2))

        return 10 * loss_overlap

    def train_loop(self, train_set, val_set, train_para, val_para, epochs=1000):
        """Run the train/validation loop without changing caller-side datasets."""
        self.train_losses = []
        self.val_losses = []

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=False)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)
        train_para_loader = DataLoader(train_para, batch_size=self.batch_size, shuffle=False)
        val_para_loader = DataLoader(val_para, batch_size=self.batch_size, shuffle=False)

        for epoch in range(epochs):
            self.modulator.train()
            total_loss = 0

            train_iter = zip(train_loader, train_para_loader)
            for (E_batch, para_batch) in tqdm(train_iter, total=len(train_loader), desc=f"[Train] Epoch {epoch+1}/{epochs}"):
                if isinstance(E_batch, (tuple, list)):
                    E_batch = E_batch[0]
                    para_batch = para_batch[0]
                E_batch = E_batch.to(self.device)
                para_batch = para_batch.to(self.device)

                self.optimizer.zero_grad()
                E_f2_out = self.forward(E_batch)
                loss = self.compute_loss(E_f2_out, para_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            self.train_losses.append(total_loss)
            print(f"[Train] Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4e}")

            self.modulator.eval()
            with torch.no_grad():
                val_loss = 0
                for E_batch, para_batch in zip(val_loader, val_para_loader):
                    if isinstance(E_batch, (tuple, list)):
                        E_batch = E_batch[0]
                        para_batch = para_batch[0]
                    E_batch = E_batch.to(self.device)
                    para_batch = para_batch.to(self.device)
                    E_f2_out = self.forward(E_batch)
                    loss = self.compute_loss(E_f2_out, para_batch)
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

