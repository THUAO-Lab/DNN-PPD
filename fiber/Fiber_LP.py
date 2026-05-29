import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv, kv
from scipy.linalg import expm
from scipy.optimize import minimize_scalar
import utils as ut

class MultimodeFiberLP:
    def __init__(self, lambda_, D, n_core, NA, Length, Rho, Theta, N, L = 15):
        """
        初始化光纤参数
        :param lambda_: 波长 (m)
        :param D: 纤芯直径 (m)
        :param n_core: 纤芯折射率
        :param NA: 数值孔径
        :param Length: 多段MMF长度数组或单个长度 (m)
        :param Rho: 曲率半径数组或单个 (m), np.inf代表无弯曲
        :param Theta: 曲率方向角度数组或单个 (rad)
        :param N: 计算网格大小 (int)
        """
        self.lambda_ = lambda_
        self.D = D
        self.NA = NA

        self.Length = np.atleast_1d(Length)
        self.Rho = np.atleast_1d(Rho)
        self.Theta = np.atleast_1d(Theta)
        self.N = N

        self.c = 3e11
        self.epsilon = 8.85e-12
        self.k0 = 2*np.pi/self.lambda_
        self.w = 2*np.pi*self.c/self.lambda_
        self.mu = 4*np.pi*1e-7


        self.a = self.D/2
        self.n_core = n_core
        self.n_clad = np.sqrt(self.n_core**2 - self.NA**2)
        self.imp_core = self.w * self.mu / (self.k0 * self.n_core)
        self.nndif = (self.n_core**2 - self.n_clad**2)/2/(n_core**2)
        self.nnsum = (self.n_core**2 + self.n_clad**2)/2/(self.n_core**2)
        self.epsilon_core = self.epsilon*self.n_core**2
        self.epsilon_clad = self.epsilon*self.n_clad**2

        # 计算网格和极坐标
        self.img_size = 1.06 * self.D
        x = np.linspace(-self.img_size/2, self.img_size/2, self.N)
        y = np.linspace(-self.img_size/2, self.img_size/2, self.N)
        self.x, self.y = np.meshgrid(x, y)
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.area_pixel = self.dx * self.dy
        self.theta = np.arctan2(self.y, self.x)
        self.rho = np.sqrt(self.x**2 + self.y**2)

        self.L = L     # 最大角动量，若光纤有很多模式则增加这个值
        self.dbn = int(1e4)  # beta采样点数，注意可能较大

        # 计算结果初始化
        self.T = None
        self.C = None
        self.T_C = None
        self.NMode = None
        self.lmap = None
        self.mmap = None
        self.LPxymap = None
        self.EHHEmap = None
        self.propconst = None
        self.propconst_LG = None
        self.Ex = None
        self.Ey = None
        self.Ez = None
        self.Hx = None
        self.Hy = None
        self.Hz = None
        
        self.Er = None
        self.Ep = None
        self.Ez = None
        self.Hr = None
        self.Hp = None
        self.Hz = None

    def _LP_transcend_func(self, x, l):
        """
        LP模式特征方程函数
        """
        ha = self.a * np.sqrt((self.n_core*self.k0)**2 - x**2)
        qa = self.a * np.sqrt(x**2 - (self.n_clad*self.k0)**2)
        with np.errstate(divide='ignore', invalid='ignore'):
            term1 = ha * (jv(l + 1, ha) / jv(l, ha))
            term2 = qa * (kv(l+1, qa) / kv(l, qa))

        return term1 - term2
    
    def _PIM_Kd_func(self, x, l):
        """
        PIM模式Kd方程
        """
        return l/x*kv(l, x)-kv(l+1, x)
    
    def _PIM_R_func(self, x, l):
        """
        PIM模式R方程
        """
        ha = self.a * np.sqrt((self.n_core*self.k0)**2 - x**2)
        qa = self.a * np.sqrt(x**2 - (self.n_clad*self.k0)**2)
        return np.sqrt( (self.nndif**2)*(self._PIM_Kd_func(qa, l)/(qa*kv(l,qa)))**2 + (l*x/(self.n_core*self.k0))**2 * ( qa**(-2) + ha**(-2) ) **2 )
    
    def _EH_transcend_func(self, x, l):
        """
        EH模式特征方程函数
        """
        ha = self.a * np.sqrt((self.n_core*self.k0)**2 - x**2)
        qa = self.a * np.sqrt(x**2 - (self.n_clad*self.k0)**2)
        with np.errstate(divide='ignore', invalid='ignore'):
            term1 = jv(l+1,ha)/(ha*jv(l,ha))
            term2 = self.nnsum*(self._PIM_Kd_func(qa, l)/(qa*kv(l,qa))) - (l/(ha**2) - self._PIM_R_func(x, l))

        return term1 - term2
    
    def _HE_transcend_func(self, x, l):
        """
        HE模式特征方程函数
        """
        ha = self.a * np.sqrt((self.n_core*self.k0)**2 - x**2)
        qa = self.a * np.sqrt(x**2 - (self.n_clad*self.k0)**2)
        with np.errstate(divide='ignore', invalid='ignore'):
            term1 = jv(l-1,ha)/(ha*jv(l,ha))
            term2 = self.nnsum*(self._PIM_Kd_func(qa, l)/(qa*kv(l,qa))) - (l/(ha**2) - self._PIM_R_func(x, l))

        return term1 + term2
    def ErEp_2_ExEy(self):
        """
        将Er和Ep转化为Ex和Ey
        """
            # 假设 X, Y 是你的空间坐标网格
        Er = self.Er
        Ep = self.Ep
        phi = self.theta  # 极角
        
        # 假设 Er 和 Ep 是你已经计算好的电场分量（复数）
        cos_phi = np.expand_dims(np.cos(phi), axis=2)
        sin_phi = np.expand_dims(np.sin(phi), axis=2)
        Ex = Er * cos_phi - Ep * sin_phi
        Ey = Er * sin_phi + Ep * cos_phi

        return Ex, Ey


    # def _propconst_fine_search(self, func_abs, beta_guess, beta_min, beta_max):
    #     """
    #     利用brentq精细搜索beta根
    #     """
    #     beta_min = max(beta_min, self.n_clad * self.k0)
    #     beta_max = min(beta_max, self.n_core * self.k0)

    #     try:
    #         root = brentq(lambda x: func_abs(x), beta_min, beta_max)
    #     except ValueError:
    #         root = beta_guess
    #     return root

    def _calc_mode_field(self, beta, l, LPxymap, rho, theta):
        """
        计算矢量 LP 模式场分布。
    
        参数：
            beta: float, 模式传播常数
            l: int, 模式角动量量子数
            LPxymap: int, 1 表示 LPx 模式，0 表示 LPy 模式
            rho, theta: np.ndarray, 极坐标 (N x N)
    
        返回：
            Ex, Ey, Ez, Hx, Hy, Hz: 模式的电磁场 (复数 N x N ndarray)
        """
        a = self.a
        n_core = self.n_core
        n_clad = self.n_clad
        k0 = self.k0
        w = self.w
        mu = 4 * np.pi * 1e-7
        z = 0  # 默认只考虑 z=0 面的分布
    
        h = np.sqrt((n_core * k0) ** 2 - beta ** 2)
        q = np.sqrt(beta ** 2 - (n_clad * k0) ** 2)
        ha = h * a
        qa = q * a
    
        B = jv(l, ha) / kv(l, qa)
    
        N = rho.shape[0]
        Ex = np.zeros((N, N), dtype=np.complex128)
        Ey = np.zeros((N, N), dtype=np.complex128)
        Ez = np.zeros((N, N), dtype=np.complex128)
        Hx = np.zeros((N, N), dtype=np.complex128)
        Hy = np.zeros((N, N), dtype=np.complex128)
        Hz = np.zeros((N, N), dtype=np.complex128)
    
        exp_core = lambda pha: np.exp(1j * (l * pha - beta * z))
        exp_pm = lambda pha: np.exp(1j * pha)
        exp_nm = lambda pha: np.exp(-1j * pha)
    
        # 核心区
        mask_core = rho < a
        r_c = rho[mask_core]
        theta_c = theta[mask_core]
        if LPxymap == 1:
            Ex[mask_core] = jv(l, h * r_c) * exp_core(theta_c)
            Ez[mask_core] = (1j * h / (2 * beta)) * (
                jv(l + 1, h * r_c) * exp_pm(theta_c) - jv(l - 1, h * r_c) * exp_nm(theta_c)
            ) * exp_core(theta_c)
            Hy[mask_core] = (beta / (w * mu)) * jv(l, h * r_c) * exp_core(theta_c)
            Hz[mask_core] = (h / (2 * w * mu)) * (
                jv(l + 1, h * r_c) * exp_pm(theta_c) + jv(l - 1, h * r_c) * exp_nm(theta_c)
            ) * exp_core(theta_c)
        else:
            Ey[mask_core] = jv(l, h * r_c) * exp_core(theta_c)
            Ez[mask_core] = (h / (2 * beta)) * (
                jv(l + 1, h * r_c) * exp_pm(theta_c) + jv(l - 1, h * r_c) * exp_nm(theta_c)
            ) * exp_core(theta_c)
            Hx[mask_core] = -(beta / (w * mu)) * jv(l, h * r_c) * exp_core(theta_c)
            Hz[mask_core] = -(1j * h / (2 * w * mu)) * (
                jv(l + 1, h * r_c) * exp_pm(theta_c) - jv(l - 1, h * r_c) * exp_nm(theta_c)
            ) * exp_core(theta_c)
    
        # 包层区
        mask_clad = rho > a
        r_k = rho[mask_clad]
        theta_k = theta[mask_clad]
        if LPxymap == 1:
            Ex[mask_clad] = B * kv(l, q * r_k) * exp_core(theta_k)
            Ez[mask_clad] = (1j * q * B / (2 * beta)) * (
                kv(l + 1, q * r_k) * exp_pm(theta_k) + kv(l - 1, q * r_k) * exp_nm(theta_k)
            ) * exp_core(theta_k)
            Hy[mask_clad] = (beta * B / (w * mu)) * kv(l, q * r_k) * exp_core(theta_k)
            Hz[mask_clad] = (q * B / (2 * w * mu)) * (
                kv(l + 1, q * r_k) * exp_pm(theta_k) - kv(l - 1, q * r_k) * exp_nm(theta_k)
            ) * exp_core(theta_k)
        else:
            Ey[mask_clad] = B * kv(l, q * r_k) * exp_core(theta_k)
            Ez[mask_clad] = (q * B / (2 * beta)) * (
                kv(l + 1, q * r_k) * exp_pm(theta_k) - kv(l - 1, q * r_k) * exp_nm(theta_k)
            ) * exp_core(theta_k)
            Hx[mask_clad] = -(beta * B / (w * mu)) * kv(l, q * r_k) * exp_core(theta_k)
            Hz[mask_clad] = -(1j * q * B / (2 * w * mu)) * (
                kv(l + 1, q * r_k) * exp_pm(theta_k) + kv(l - 1, q * r_k) * exp_nm(theta_k)
            ) * exp_core(theta_k)
    
        return Ex, Ey, Ez, Hx, Hy, Hz
    
    def MMFEHHEfields(self, epsilon, epsilon2, EHorHE, beta, l, rho, theta):
        """
        计算EH/HE模式下的矢量电磁场分布（Er, Ep, Ez, Hr, Hp, Hz）
        所有变量均为torch.tensor，rho和theta为2D网格张量
        """
    
        a = self.a
        n_core = self.n_core
        n_clad = self.n_clad
        k0 = self.k0
        w = self.w
        mu = self.mu
        z = 0  # 默认只考虑 z=0 面的分布
    
        h = np.sqrt((n_core * k0) ** 2 - beta ** 2)
        q = np.sqrt(beta ** 2 - (n_clad * k0) ** 2)
        ha = h * a
        qa = q * a
    
        # 初始化输出张量
        Er = np.zeros_like(rho, dtype=np.complex128)
        Ep = np.zeros_like(rho, dtype=np.complex128)
        Ez = np.zeros_like(rho, dtype=np.complex128)
        Hr = np.zeros_like(rho, dtype=np.complex128)
        Hp = np.zeros_like(rho, dtype=np.complex128)
        Hz = np.zeros_like(rho, dtype=np.complex128)
    
        # 导数项（避免 r=0）
        def Jd(r):
            return jv(l - 1, r) - (l / r) * jv(l, r)
    
        def Kd(r):
            return (l / r) * kv(l, r) - kv(l + 1, r)
    
        if l == 0:
            if EHorHE == 1:  # TM 模式
                A = 1.0
                B = 0.0
                C = A * (jv(0, ha) / kv(0, qa))
                D = 0.0
            else:  # TE 模式
                A = 0.0
                B = 1.0
                C = 0.0
                D = B * (jv(0, ha) / kv(0, qa))
        else:
            A = 1.0
            B = (1j * l * beta / (w * mu)) * (qa ** (-2) + ha ** (-2)) / \
                (Jd(ha) / (ha * jv(l, ha)) + Kd(qa) / (qa * kv(l, qa)))
            C = (jv(l, ha) / kv(l, qa)) * A
            D = B * C
    
        exp_phase = lambda r, pha: np.exp(1j * (l * pha - beta * z))
    
        # 核心区（core）
        def Ercore(r, pha):
            return -(1j * beta / h ** 2) * (
                A * h * Jd(h * r) + (1j * w * mu * l / (beta * r)) * B * jv(l, h * r)
            ) * exp_phase(r, pha)
    
        def Epcore(r, pha):
            return -(1j * beta / h ** 2) * (
                (1j * l / r) * A * jv(l, h * r) - (w * mu / beta) * B * h * Jd(h * r)
            ) * exp_phase(r, pha)
    
        def Ezcore(r, pha):
            return A * jv(l, h * r) * exp_phase(r, pha)
    
        def Hrcore(r, pha):
            return -(1j * beta / h ** 2) * (
                B * h * Jd(h * r) - (1j * w * epsilon * l / (beta * r)) * A * jv(l, h * r)
            ) * exp_phase(r, pha)
    
        def Hpcore(r, pha):
            return -(1j * beta / h ** 2) * (
                (1j * l / r) * B * jv(l, h * r) + (w * epsilon / beta) * A * h * Jd(h * r)
            ) * exp_phase(r, pha)
    
        def Hzcore(r, pha):
            return B * jv(l, h * r) * exp_phase(r, pha)
    
        # 包层（cladding）
        def Erclad(r, pha):
            return (1j * beta / q ** 2) * (
                C * q * Kd(q * r) + (1j * w * mu * l / (beta * r)) * D * kv(l, q * r)
            ) * exp_phase(r, pha)
    
        def Epclad(r, pha):
            return (1j * beta / q ** 2) * (
                (1j * l / r) * C * kv(l, q * r) - (w * mu / beta) * D * q * Kd(q * r)
            ) * exp_phase(r, pha)
    
        def Ezclad(r, pha):
            return C * kv(l, q * r) * exp_phase(r, pha)
    
        def Hrclad(r, pha):
            return (1j * beta / q ** 2) * (
                D * q * Kd(q * r) - (1j * w * epsilon2 * l / (beta * r)) * C * kv(l, q * r)
            ) * exp_phase(r, pha)
    
        def Hpclad(r, pha):
            return (1j * beta / q ** 2) * (
                (1j * l / r) * D * kv(l, q * r) + (w * epsilon2 / beta) * C * q * Kd(q * r)
            ) * exp_phase(r, pha)
    
        def Hzclad(r, pha):
            return D * kv(l, q * r) * exp_phase(r, pha)
    
        # 核心区域
        mask_core = (rho <= a) & (rho != 0)
        Er[mask_core] = Ercore(rho[mask_core], theta[mask_core])
        Ep[mask_core] = Epcore(rho[mask_core], theta[mask_core])
        Ez[mask_core] = Ezcore(rho[mask_core], theta[mask_core])
        Hr[mask_core] = Hrcore(rho[mask_core], theta[mask_core])
        Hp[mask_core] = Hpcore(rho[mask_core], theta[mask_core])
        Hz[mask_core] = Hzcore(rho[mask_core], theta[mask_core])
    
        # 包层区域
        mask_clad = (rho > a)
        Er[mask_clad] = Erclad(rho[mask_clad], theta[mask_clad])
        Ep[mask_clad] = Epclad(rho[mask_clad], theta[mask_clad])
        Ez[mask_clad] = Ezclad(rho[mask_clad], theta[mask_clad])
        Hr[mask_clad] = Hrclad(rho[mask_clad], theta[mask_clad])
        Hp[mask_clad] = Hpclad(rho[mask_clad], theta[mask_clad])
        Hz[mask_clad] = Hzclad(rho[mask_clad], theta[mask_clad])
    
        # 中心点 r == 0 的数值处理
        r_min = np.min(rho[rho != 0])
        center_mask = (rho == 0)
        neighbor_mask = (rho == r_min)
    
        def average_complex(tensor):
            amp = np.mean(np.abs(tensor[neighbor_mask]))
            phase = np.mean(np.angle(tensor[neighbor_mask]))
            return amp * np.exp(1j * phase)
    
        Er[center_mask] = average_complex(Er)
        Ep[center_mask] = average_complex(Ep)
        Ez[center_mask] = average_complex(Ez)
        Hr[center_mask] = average_complex(Hr)
        Hp[center_mask] = average_complex(Hp)
        Hz[center_mask] = average_complex(Hz)
    
        return Er, Ep, Ez, Hr, Hp, Hz
    
    def find_LP_root_indices(self, LP_values, threshold=3, margin=2):
        """
        查找 LP 模式中传播常数的粗略根索引。
        
        参数:
        - LP: numpy 数组，对应于某阶 Bessel 方程的函数值序列。
        - threshold: abs(LP) 小于该值才认为是可接受根。
        - margin: 为了后续使用 LPind ± margin，避免越界。
    
        返回:
        - LPind: 粗略根位置的索引数组（已升序排列）。
        """
        # 找出符号变化点
        sign_change = np.where(np.abs(np.diff(np.sign(LP_values))) > 0)[0]
        # 找出函数值小于 threshold 的点
        small_value = np.where(np.abs(LP_values) < threshold)[0]
        # 取交集并排序
        LPind = np.intersect1d(sign_change, small_value)
        
        # 排除前后越界情况（用于后续 LPind ± 2）
        LPind = LPind[(LPind >= margin) & (LPind < len(LP_values) - margin)]
    
        return LPind

    def solve_modes_LP(self):
        """
        求解所有LP模式及传输矩阵
        """
        Beta = np.linspace(self.n_core*self.k0, self.n_clad*self.k0, self.dbn)

        LPbeta = np.zeros((self.L+1, self.L))
        LPm = np.zeros(self.L+1)
        max_roots_per_mode = self.L

        for l in range(self.L+1):
            func_abs = lambda x: np.abs(self._LP_transcend_func(x, l))**2
        
            func_vals = self._LP_transcend_func(Beta, l)
            LPind = self.find_LP_root_indices(func_vals)
        
            betas_l = self.propconst_fine_search(func_abs, Beta[LPind], Beta[LPind+2], Beta[LPind-2])
            
            # 关键修改：只取前max_roots_per_mode个根
            if len(betas_l) > max_roots_per_mode:
                betas_l = betas_l[:max_roots_per_mode]
            
            # 确保不会超过预分配数组的大小
            actual_len = min(len(betas_l), LPbeta.shape[1])
            LPbeta[l, :actual_len] = betas_l[:actual_len]
            LPm[l] = actual_len

        LPlmax = np.max(np.where(LPm != 0)[0])

        NLP = np.sum(LPbeta[0, :] != 0) + 2 * np.sum(LPbeta[1:, :] != 0)
        NMode = 2*NLP

        lmap = []
        mmap = []
        for l in range(-LPlmax, LPlmax+1):
            count = int(LPm[abs(l)])
            for m in range(1, count+1):
                lmap.append(l)
                mmap.append(m)
        for l in range(-LPlmax, LPlmax+1):
            count = int(LPm[abs(l)])
            for m in range(1, count+1):
                lmap.append(l)
                mmap.append(m)

        lmap = np.array(lmap)
        mmap = np.array(mmap)

        LPxymap = np.concatenate((np.ones(NLP), np.zeros(NLP))).astype(int)

        Ex = np.zeros((self.N, self.N, NMode), dtype=np.complex128)
        Ey = np.zeros((self.N, self.N, NMode), dtype=np.complex128)
        Ez = np.zeros_like(Ex)
        Hx = np.zeros_like(Ex)
        Hy = np.zeros_like(Ex)
        Hz = np.zeros_like(Ex)

        for ii in range(NMode):
            l = lmap[ii]
            m = mmap[ii]
            pol = LPxymap[ii]
            beta_mode = LPbeta[abs(l), m-1]
            Ex[:, :, ii], Ey[:, :, ii], Ez[:, :, ii], Hx[:, :, ii], Hy[:, :, ii], Hz[:, :, ii] = \
                self._calc_mode_field(beta_mode, l, pol, self.rho, self.theta)

            poynting = np.sum(self.dx*self.dy * np.abs(Ex[:, :, ii] * np.conj(Hy[:, :, ii]) - Ey[:, :, ii] * np.conj(Hx[:, :, ii])))
            if poynting != 0:
                Ex[:, :, ii] /= np.sqrt(poynting)
                Ey[:, :, ii] /= np.sqrt(poynting)
                Ez[:, :, ii] /= np.sqrt(poynting)
                Hx[:, :, ii] /= np.sqrt(poynting)
                Hy[:, :, ii] /= np.sqrt(poynting)
                Hz[:, :, ii] /= np.sqrt(poynting)

        propconst = np.zeros(NMode)
        for ii in range(NMode):
            propconst[ii] = LPbeta[abs(lmap[ii]), mmap[ii]-1]

        n_segments = len(self.Length)
        T = np.eye(NMode, dtype=np.complex128)

        MDL = np.ones(NMode)  # 模依赖损耗全1，即无损

        sigma = 0.17
        xi = 1 - (1 - 2*sigma) * ((self.n_core - 1) / self.n_core)

        Ex_vec = Ex.reshape(self.N*self.N, NMode)
        Ey_vec = Ey.reshape(self.N*self.N, NMode)

        for seg_i in range(n_segments):
            length_seg = self.Length[seg_i]
            rho_seg = self.Rho[seg_i]
            theta_seg = self.Theta[seg_i]

            if np.isinf(rho_seg):
                T = np.diag(MDL * np.exp(1j * -propconst * length_seg)) @ T
            else:
                Rho_eff = rho_seg / xi
                X = (np.cos(theta_seg)*self.x + np.sin(theta_seg)*self.y).reshape(self.N*self.N, 1)

                A = self.area_pixel / self.imp_core * (
                    (Ex_vec.conj().T @ (X * Ex_vec)) + (Ey_vec.conj().T @ (X * Ey_vec))
                )
                B0 = np.diag(propconst)
                B = B0 - (self.n_core * self.k0 / Rho_eff) * A

                T = np.diag(MDL) @ expm(1j * -B * length_seg) @ T

        self.T = T
        self.NMode = NMode
        self.lmap = lmap
        self.mmap = mmap
        self.LPxymap = LPxymap
        self.propconst = propconst
        self.Ex = Ex
        self.Ey = Ey
        self.Ez = Ez
        self.Hx = Hx
        self.Hy = Hy
        self.Hz = Hz
        
    def solve_modes_PIM(self):
        """
        求解所有LP模式及传输矩阵
        """
        Beta = np.linspace(self.n_core*self.k0, self.n_clad*self.k0, self.dbn)

        EHbeta = np.zeros((self.L+1, self.L))
        EHm = np.zeros(self.L+1)
        HEbeta = np.zeros((self.L+1, self.L))
        HEm = np.zeros(self.L+1)

        for l in range(self.L+1):
            #EH模式
            func_abs_EH = lambda x: np.abs(self._EH_transcend_func(x, l))**2

            func_vals = self._EH_transcend_func(Beta, l)
            LPind = self.find_LP_root_indices(func_vals)

            betas_l = self.propconst_fine_search(func_abs_EH, Beta[LPind], Beta[LPind+2], Beta[LPind-2])
            EHbeta[l, :len(betas_l)] = betas_l
            EHm[l] = np.sum(betas_l != 0)
            
            #HE模式
            func_abs_HE = lambda x: np.abs(self._HE_transcend_func(x, l))**2

            func_vals = self._HE_transcend_func(Beta, l)
            LPind = self.find_LP_root_indices(func_vals)

            betas_l = self.propconst_fine_search(func_abs_HE, Beta[LPind], Beta[LPind+2], Beta[LPind-2])
            HEbeta[l, :len(betas_l)] = betas_l
            HEm[l] = np.sum(betas_l != 0)

        EHlmax = np.max(np.where(EHm != 0)[0])
        HElmax = np.max(np.where(HEm != 0)[0])

        NEH = np.sum(EHbeta[0, :] != 0) + 2 * np.sum(EHbeta[1:, :] != 0)
        NHE = np.sum(HEbeta[0, :] != 0) + 2 * np.sum(HEbeta[1:, :] != 0)
        NMode = NEH + NHE
        EHHEmap = np.concatenate((np.ones(NEH), np.zeros(NHE))).astype(int)

        lmap = []
        mmap = []
        for l in range(-EHlmax, EHlmax+1):
            count = int(EHm[np.abs(l)])
            for m in range(1, count+1):
                lmap.append(l)
                mmap.append(m)
        for l in range(-HElmax, HElmax+1):
            count = int(HEm[abs(l)])
            for m in range(1, count+1):
                lmap.append(l)
                mmap.append(m)

        lmap = np.array(lmap)
        mmap = np.array(mmap)

        Er = np.zeros((self.N, self.N, NMode), dtype=np.complex128)
        Ep = np.zeros((self.N, self.N, NMode), dtype=np.complex128)
        Ez = np.zeros_like(Er)
        Hr = np.zeros_like(Er)
        Hp = np.zeros_like(Er)
        Hz = np.zeros_like(Er)

        for ii in range(NMode):
            l = lmap[ii]
            m = mmap[ii]
            if EHHEmap[ii] == 1:
                mode_beta = EHbeta[ np.abs(l), m-1]
            else:
                mode_beta = HEbeta[ np.abs(l), m-1]
                
            Er[:, :, ii], Ep[:, :, ii], Ez[:, :, ii], Hr[:, :, ii], Hp[:, :, ii], Hz[:, :, ii] = \
                self.MMFEHHEfields(self.epsilon_core, self.epsilon_clad, EHHEmap[ii], mode_beta, l, self.rho, self.theta)
                

            poynting = np.sum(self.dx*self.dy * np.abs(Er[:, :, ii] * Hp[:, :, ii].conj() - Ep[:, :, ii] * Hr[:, :, ii].conj()))
            if poynting != 0:
                Er[:, :, ii] /= np.sqrt(poynting)
                Ep[:, :, ii] /= np.sqrt(poynting)
                Ez[:, :, ii] /= np.sqrt(poynting)
                Hr[:, :, ii] /= np.sqrt(poynting)
                Hp[:, :, ii] /= np.sqrt(poynting)
                Hz[:, :, ii] /= np.sqrt(poynting)

        propconst = np.zeros(NMode)
        for ii in range(NMode):
            l  = lmap[ii]
            m = mmap[ii]
            if EHHEmap[ii] == 1:
                propconst[ii] = EHbeta[ np.abs(l) , m-1]
            else:
                propconst[ii] = HEbeta[ np.abs(l) , m-1]

        n_segments = len(self.Length)
        T = np.eye(NMode, dtype=np.complex128)

        MDL = np.ones(NMode)  # 模依赖损耗全1，即无损

        sigma = 0.17
        xi = 1 - (1 - 2*sigma) * ((self.n_core - 1) / self.n_core)

        Er_vec = Er.reshape(self.N*self.N, NMode)
        Ep_vec = Ep.reshape(self.N*self.N, NMode)

        for seg_i in range(n_segments):
            length_seg = self.Length[seg_i]
            rho_seg = self.Rho[seg_i]
            theta_seg = self.Theta[seg_i]

            if np.isinf(rho_seg):
                T = np.diag(MDL * np.exp(1j * -propconst * length_seg)) @ T
            else:
                Rho_eff = rho_seg / xi
                X = (np.cos(theta_seg)*self.x + np.sin(theta_seg)*self.y).reshape(self.N*self.N, 1)

                A = self.area_pixel / self.imp_core * (
                    (Er_vec.conj().T @ (X * Er_vec)) + (Ep_vec.conj().T @ (X * Ep_vec))
                )
                B0 = np.diag(propconst)
                B = B0 - (self.n_core * self.k0 / Rho_eff) * A

                T = np.diag(MDL) @ expm(1j * -B * length_seg) @ T

        self.T = T
        self.NMode = NMode
        self.lmap = lmap
        self.mmap = mmap
        self.EHHEmap = EHHEmap
        self.propconst = propconst
        self.Er = Er
        self.Ep = Ep
        self.Ez = Ez
        self.Hr = Hr
        self.Hp = Hp
        self.Hz = Hz
        

    def propconst_fine_search(self, transcend_func_abs, beta_coarse, LB, UB):
        """
        精细搜索传播常数 beta。
        
        Parameters:
        - transcend_func_abs: 被最小化的目标函数 (|transcend_func|^2)
        - beta_coarse: 初始传播常数估计数组
        - LB: 对应的每个beta的搜索下限
        - UB: 对应的每个beta的搜索上限
    
        Returns:
        - beta_fine: 更精确求得的传播常数数组
        """
        beta_fine = np.zeros_like(beta_coarse)
    
        for i in range(len(beta_coarse)):
            res = minimize_scalar(
                transcend_func_abs,
                bounds=(LB[i], UB[i]),
                method='bounded',
                options={'xatol': 1e-12}
            )
            beta_fine[i] = res.x
    
        return beta_fine
    
    def compute_mode_coupling_T(self, prop_constants,
                                prop_dis,
                                mode_fields,
                                delta_eps_map,
                                dx,
                                dy,
                                beta_cutoff_r=0.2,
                                normalize_modes_flag=True,
                                absorb_self_to_beta=False,
                                n_steps=None,
                                prefactor_omega_over_4=False,
                                phase_threshold=1e-100):  # 阈值
        """
        计算模式耦合矩阵 C（单位：1/m）和对应的传播矩阵 T(z) = exp((-i B + iC) * z).
    
        参数（必填）:
        -----------
        prop_constants : array_like, shape (M,)
            模式的传播常数 β_p (单位 1/m)。
        prop_dis : float
            传播长度 z（单位 m）。
        mode_fields : ndarray, shape (nx, ny, M, 2) or (nx, ny, M)
            模式场。若最后一维为 2，则认为是矢量场 [Ex, Ey]；若没有最后一维，则按标量 Ex 处理。
            模式应为复值数组（complex）。
            每个模式可不必归一化，函数中可选择做归一化。
        delta_eps_map : ndarray, shape (nx, ny)
            空间上 Δε(x,y)（可以是 2 n Δn 或直接是 Δε，取决你如何构造），实值或复值。
        dx, dy : float
            网格采样间隔（m）。
        lambda0 : float
            工作波长（m），用于计算 ω = 2πc/λ。
        
        可选参数:
        -----------
        beta_cutoff : float or None
            若不为 None，则对 |β_p - β_q| > beta_cutoff 的模式对强制 C_{pq}=0（相位失配剪枝，单位 1/m）。
        normalize_modes_flag : bool
            是否先把每个模式按功率归一化（推荐 True）。
        absorb_self_to_beta : bool
            若 True，则把 C 的自耦合项（p==q）吸收到 prop_constants（修改后的有效 β）
            并在返回的 C 中把对角清零（常用于把自项视为传播常数修正）。
        n_steps : int
            分段传播步数（用于数值稳定）。若 n_steps > 1，则用 expm(K*dz) 连乘 n_steps 次。
        prefactor_omega_over_4 : bool
            若 True，使用耦合前因子 (ω/4)；若 False，可自行替换其他常数（某些文献用 ω/2 等）。
        phase_threshold : float, optional
            当 T 矩阵元素的模长小于此阈值时，将其设为0。默认值 1e-10。
        
        返回:
        -------
        C : ndarray, shape (M, M)
            模式耦合矩阵（complex）。
        T : ndarray, shape (M, M)
            传播矩阵 T(z) = exp((i B + C) z)（complex）。
            
        备注 / 物理说明:
        ----------------
        - 推荐 delta_eps_map 为物理的 Δε(x,y)（例如 2 n Δn），且其幅度要小（例如 Δn ~ 1e-6..1e-4）。
        - 若你希望把自耦合项作为 β 的修正，请设置 absorb_self_to_beta=True。
        - 若计算结果产生极大数值或 NaN，请检查 delta_eps_map 和模式归一化（以及 n_steps>1）。
        """
        # --- 验证形状 ---
        prop_constants = np.asarray(prop_constants)
        M = prop_constants.size
    
        mode_fields = np.asarray(mode_fields)
        if mode_fields.ndim not in (3, 4):
            raise ValueError("mode_fields must be shape (nx,ny,M) or (nx,ny,M,2)")
    
        if mode_fields.ndim == 3:
            # make shape (nx,ny,M,1) for uniform handling (scalar Ex)
            mode_fields = mode_fields[..., np.newaxis]
    
        nx, ny, M_fields, comps = mode_fields.shape
        if M_fields != M:
            raise ValueError(f"prop_constants length {M} != mode_fields modes {M_fields}")
    
        if delta_eps_map.shape != (nx, ny):
            raise ValueError(f"delta_eps_map shape {delta_eps_map.shape} must match mode_fields spatial shape {(nx,ny)}")
    
        # --- 物理常数 ---
        omega = 2.0 * np.pi * self.c / float(self.lambda_)
        coeff = (omega / 4.0) if prefactor_omega_over_4 else (omega / 2.0)
    
        # --- 归一化模式（按功率） ---
        if normalize_modes_flag:
            for p in range(M):
                Ex = mode_fields[:, :, p, 0]
                Ey = mode_fields[:, :, p, 1] if comps > 1 else 0.0 * Ex
                # 修正：先乘dx*dy再求和
                power = np.real(np.sum((np.abs(Ex)**2 + np.abs(Ey)**2) * dx * dy))
                if power <= 0:
                    raise ValueError(f"mode {p} has non-positive power {power}")
                mode_fields[:, :, p, 0] = mode_fields[:, :, p, 0] / np.sqrt(power)
                if comps > 1:
                    mode_fields[:, :, p, 1] = mode_fields[:, :, p, 1] / np.sqrt(power)
    
        # --- 计算 C_{pq} ---
        # 改为复数类型，因为delta_eps_map可能是复数
        C = np.zeros((M, M), dtype=float)
        weight = dx * dy
        
        # 提前计算所有模式的电场，避免重复计算
        mode_Ex = mode_fields[:, :, :, 0]
        mode_Ey = mode_fields[:, :, :, 1] if comps > 1 else np.zeros_like(mode_Ex)
        
        # 计算传播常数的范围
        beta_min = np.min(prop_constants)
        beta_max = np.max(prop_constants)
        beta_range = beta_max - beta_min
        
        # 设置为范围的某个百分比
        beta_cutoff = beta_cutoff_r * beta_range  # 10%的范围
        
        for p in range(M):
            Ex_p = mode_Ex[:, :, p]
            Ey_p = mode_Ey[:, :, p]
            for q in range(M):
                # 应用beta_cutoff筛选
                if beta_cutoff is not None:
                    beta_diff = abs(prop_constants[p] - prop_constants[q])
                    if beta_diff > beta_cutoff:
                        C[p, q] = 0.0
                        continue
                
                Ex_qc = np.conj(mode_Ex[:, :, q])
                Ey_qc = np.conj(mode_Ey[:, :, q])
                
                # 计算重叠积分
                integrand = delta_eps_map * (Ex_p * Ex_qc + Ey_p * Ey_qc)* weight
                C[p, q] = coeff * np.sum(integrand) 
    
        # --- 处理对角项（可选：吸收到 beta） ---
        betas = prop_constants.astype(complex).copy()  # 改为复数以容纳可能的复数修正
        
        if absorb_self_to_beta:
            # 计算对角自耦合项并吸收到betas
            diag_self = np.diag(C)  # 直接取复数对角元素
            # 添加到betas（传播常数修正）
            betas = betas + diag_self
            np.fill_diagonal(C, 0.0)
        else:
            # 保持C的对角为零（通常我们不希望自耦合出现在C）
            np.fill_diagonal(C, 0.0)
    
        # --- 构造 K = -i B + jC 并计算矩阵指数（可分段） ---
        # 根据物理推导：dA/dz = -j*B*A + j*C*A = j*(C - B)*A
        # 所以 K = j*(C - B)
        B = np.diag(betas)
        K = 1j * (C - B)  # 修正：应该是 j*(C - B)
    
        # 数值检查：若 K 的谱包含非常大的正实特征值，给出警告（可能导致 expm 溢出）
        evals = np.linalg.eigvals(K)
        max_real = np.max(np.real(evals))
        if max_real * prop_dis > 50:  # 调整为50
            import warnings
            warnings.warn(f"Large real part in eig(K)*z: max Re(eig)*z = {max_real*prop_dis:.3e}. "
                          "expm may overflow or produce nonphysical growth. "
                          "Check delta_eps_map magnitude, normalization, or use smaller n_steps.")
    
        # 计算传播矩阵T
        if n_steps is None or n_steps <= 1:
            T = expm(K * prop_dis)
        else:
            dz = prop_dis / float(n_steps)
            T = np.eye(M, dtype=complex)
            # 分段计算以提高稳定性
            Keachdz = K * dz
            E_step = expm(Keachdz)
            for _ in range(n_steps):
                T = E_step @ T
    
        # # --- 阈值处理：当模长小于阈值时，元素设为0 ---
        # amplitudes = np.abs(T)
        # small_amplitude_mask = amplitudes < phase_threshold
        # T_processed = T.copy()
        # T_processed[small_amplitude_mask] = 0.0
        
        # # 可选：打印一些统计信息（不会太多）
        # n_small = np.sum(small_amplitude_mask)
        # if n_small > 0:
        #     print(f"清零处理: {n_small}/{M*M} 个元素模长 < {phase_threshold:.1e}")
    
        self.T_C = T
        self.C = C
    
        return C, T

    def plot_prop_constants(self):
    
        if self.lmap is None or self.propconst is None:
            raise ValueError("请先调用 solve_modes_LP() 方法")
    
        LPlmax = int(np.max(np.abs(self.lmap)))
        plt.figure(figsize=(8, 3))
        for l in range(0, LPlmax+1):
            beta_vals = self.propconst[self.lmap == l]
            plt.scatter([l] * len(beta_vals), beta_vals, s=20, c='black')
        plt.grid(True)
        plt.xlabel("orbital angular momentum (l index)")
        plt.ylabel("propagation const. (β in m⁻¹)")
        plt.title("LP Modes Propagation Constants")
        plt.tight_layout()
        plt.show()
        
    def plot_PIM_modes(self):
    
        if self.Er is None or self.lmap is None or self.mmap is None:
            raise ValueError("请先调用 solve_modes_PIM() 方法")
    
        PIMlmax = int(np.max(np.abs(self.lmap)))
        max_EH = int(np.max(self.mmap[self.EHHEmap == 1]))
        max_HE = int(np.max(self.mmap[self.EHHEmap == 0]))
        cols = 2 * PIMlmax + 1
    
        plt.figure(figsize=(cols * 2, max_EH * 2))
        for ii in range(self.NMode // 2):  # 只显示 LPx 模式
            if self.EHHEmap[ii]==1:
                temp = self.PIM_field[:, :, ii]
                l = self.lmap[ii]
                m = self.mmap[ii]
                row = m - 1
                col = l + PIMlmax
                idx = row * cols + col + 1
        
                plt.subplot(max_EH, cols, idx)
                plt.imshow(np.abs(temp), cmap='twilight')
                plt.title(f"PIM EH l={l}, m={m}")
                plt.axis('off')
            else:
                continue
            
        plt.figure(figsize=(cols * 2, max_HE * 2))
        for ii in range(self.NMode // 2):  # 只显示 LPx 模式
            if self.EHHEmap[ii]==0:
                temp = self.PIM_field[:, :, ii]
                l = self.lmap[ii]
                m = self.mmap[ii]
                row = m - 1
                col = l + PIMlmax
                idx = row * cols + col + 1
        
                plt.subplot(max_HE, cols, idx)
                plt.imshow(np.abs(temp), cmap='twilight')
                plt.title(f"PIM HE l={l}, m={m}")
                plt.axis('off')
            else:
                continue
        plt.tight_layout()
        plt.show()
        
    def plot_LP_modes(self):
    
        if self.Ex is None or self.lmap is None or self.mmap is None:
            raise ValueError("请先调用 solve_modes_LP() 方法")
    
        LPlmax = int(np.max(np.abs(self.lmap)))
        max_m = int(np.max(self.mmap[self.LPxymap == 1]))
        cols = 2 * LPlmax + 1
    
        plt.figure(figsize=(cols * 2, max_m * 2))
        for ii in range(self.NMode // 2):  # 只显示 LPx 模式
            temp = self.Ex[:, :, ii]
            l = self.lmap[ii]
            m = self.mmap[ii]
            row = m - 1
            col = l + LPlmax
            idx = row * cols + col + 1
    
            plt.subplot(max_m, cols, idx)
            plt.imshow(np.abs(temp), cmap='hot')
            plt.title(f"LPx l={l}, m={m}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()
        
        
    def plot_LG_modes(self):
     if self.Ex is None or self.lmap is None or self.LPxymap is None:
         raise ValueError("请先调用 solve_modes_LP() 方法")
 
     LPlmax = int(np.max(np.abs(self.lmap)))
     max_m = int(np.max(self.mmap))
     cols = LPlmax + 1
 
     plt.figure(figsize=(cols * 2.5, max_m * 2.5))
     for l in range(0, LPlmax + 1):
         idx_pm = np.where((self.lmap == l) & (self.LPxymap == 1))[0]
         idx_nm = np.where((self.lmap == -l) & (self.LPxymap == 1))[0]
 
         for ii in range(min(len(idx_pm), len(idx_nm))):
             field_plus = self.Ex[:, :, idx_pm[ii]]
             field_minus = self.Ex[:, :, idx_nm[ii]]
             field_sum = field_plus + field_minus
             img = np.abs(field_sum)
             img /= np.max(img)
 
             row = ii
             col = l
             idx = row * cols + col + 1
             plt.subplot(max_m, cols, idx)
             plt.imshow(img, cmap='hot')
             plt.title(f"LG({ii},{l})")
             plt.axis('off')
     plt.tight_layout()
     plt.show() 
     
    def get_PIM_total_field(self, normalize=True, pol = 'x'):
        """
        获取每个PIM模式的复合横向电场，可作为传输矩阵基底使用。
    
        参数：
        - normalize: 是否对每个模式能量归一化（默认 True）
    
        返回：
        - PIM_field: 复数张量 (H, W, NMode)，表示 LP 模式复合场 Ex + Ey
        """
        if self.Er is None or self.Ep is None:
            raise ValueError("请先调用 solve_modes_PIM() 方法生成 Er 和 Ep")
            
        Ex, Ey = self.ErEp_2_ExEy()
            
        if pol == 'x': 
            self.PIM_field = Ex
        else:
            if pol == 'y': 
                self.PIM_field = Ey
            else:
                self.PIM_field = self.Er # shape: (H, W, NMode)
    
    
        if normalize:
            H, W, NMode = self.PIM_field.shape
            for i in range(NMode):
                mode = self.PIM_field[:, :, i]
                energy = np.sum(np.abs(mode)**2)
                self.LP_mode_indices = [(int(self.lmap[i]), int(self.mmap[i])) for i in range(len(self.lmap))]
                if energy > 1:
                    self.PIM_field[:, :, i] = mode / np.sqrt(energy)
    
    
        return self.PIM_field
     
    def get_LP_total_field(self, normalize=True, pol = 'x'):
        """
        获取每个 LP 模式的复合横向电场（Ex + Ey），可作为传输矩阵基底使用。
    
        参数：
        - normalize: 是否对每个模式能量归一化（默认 True）
    
        返回：
        - LP_field: 复数张量 (H, W, NMode)，表示 LP 模式复合场 Ex + Ey
        """
        if self.Ex is None or self.Ey is None:
            raise ValueError("请先调用 solve_modes_LP() 方法生成 Ex 和 Ey")
            
        if pol == 'x': 
            self.LP_field = self.Ex
        else:
            if pol == 'y': 
                self.LP_field = self.Ey
            else:
                self.LP_field = self.Ex + self.Ey # shape: (H, W, NMode)
    
    
        if normalize:
            H, W, NMode = self.LP_field.shape
            for i in range(NMode):
                mode = self.LP_field[:, :, i]
                energy = np.sum(np.abs(mode)**2)
                self.LP_mode_indices = [(int(self.lmap[i]), int(self.mmap[i])) for i in range(len(self.lmap))]
                if energy > 1:
                    self.LP_field[:, :, i] = mode / np.sqrt(energy)

    
        return self.LP_field
    
    def get_LG_total_field(self, normalize=True, pol='x', sort_by_beta=True):
        """
        计算LG模式的总场分布
        
        参数:
            normalize: 是否归一化模式场
            pol: 偏振方向 ('x', 'y', 或其他表示总场)
            sort_by_beta: 是否按传播常数从小到大排序 (True/False)
        """
        if self.Ex is None or self.lmap is None or self.LPxymap is None or self.NMode is None:
            raise ValueError("请先调用 solve_modes_LP() 计算LP模式相关数据")
        
        LPlmax = int(np.max(np.abs(self.lmap)))
        
        # 先计算总的LG模式数目，等于所有 l 从0到LPlmax，每个 l 对应 min(正负 l 模式数)
        LG_mode_count = 0
        for l in range(LPlmax + 1):
            idx_pm = np.where((self.lmap == l) & (self.LPxymap == 1))[0]
            idx_nm = np.where((self.lmap == -l) & (self.LPxymap == 1))[0]
            LG_mode_count += min(len(idx_pm), len(idx_nm))
        
        H, W, _ = self.Ex.shape
        LG_Ex = np.zeros((H, W, LG_mode_count), dtype=np.complex128)
        LG_Ey = np.zeros((H, W, LG_mode_count), dtype=np.complex128)
        self.LG_mode_indices = []  # 存储(l, m)
        self.propconst_LG = np.zeros(LG_mode_count, dtype=np.float64)  # 存储传播常数
        self.LG_LP_indices = []  # 存储对应的LP模式索引对 (idx_pm, idx_nm)
        
        # 临时存储模式信息，以便排序
        mode_info_list = []
        
        count = 0
        for l in range(LPlmax + 1):
            idx_pm = np.where((self.lmap == l) & (self.LPxymap == 1))[0]
            idx_nm = np.where((self.lmap == -l) & (self.LPxymap == 1))[0]
            
            mode_num = min(len(idx_pm), len(idx_nm))
            for ii in range(mode_num):
                # 获取对应的LP模式索引
                lp_idx_plus = idx_pm[ii]
                lp_idx_minus = idx_nm[ii]
                
                # 获取场分布
                Ex_plus = self.Ex[:, :, lp_idx_plus]
                Ex_minus = self.Ex[:, :, lp_idx_minus]
                Ey_plus = self.Ey[:, :, lp_idx_plus]
                Ey_minus = self.Ey[:, :, lp_idx_minus]
                
                # 获取模式参数
                lg_l = int(self.lmap[lp_idx_plus])
                lg_m = int(self.mmap[lp_idx_plus])
                
                # 获取传播常数（假设LP模式±l的传播常数相同，取其中一个）
                if hasattr(self, 'propconst'):
                    beta = self.propconst[lp_idx_plus]
                else:
                    # 如果没有betas属性，设置默认值或尝试从其他属性获取
                    beta = 0.0
                
                # 存储模式信息
                mode_info = {
                    'index': count,
                    'l': lg_l,
                    'm': lg_m,
                    'beta': beta,
                    'lp_indices': (lp_idx_plus, lp_idx_minus),
                    'Ex_plus': Ex_plus,
                    'Ex_minus': Ex_minus,
                    'Ey_plus': Ey_plus,
                    'Ey_minus': Ey_minus
                }
                mode_info_list.append(mode_info)
                
                count += 1
        
        # 如果要求按传播常数排序
        if sort_by_beta:
            # 按传播常数从小到大排序
            # 注意：基模（传播常数最大）通常排最后
            # 如果希望基模排第一，可以改为：key=lambda x: -np.real(x['beta'])
            mode_info_list.sort(key=lambda x: -np.real(x['beta']))
        
        # 按排序后的顺序构建LG模式
        for count, info in enumerate(mode_info_list):
            # 构建LG模式场
            LG_Ex[:, :, count] = info['Ex_plus'] + info['Ex_minus']
            LG_Ey[:, :, count] = info['Ey_plus'] + info['Ey_minus']
            
            # 存储模式参数
            self.LG_mode_indices.append((info['l'], info['m']))
            self.propconst_LG[count] = info['beta']
            self.LG_LP_indices.append(info['lp_indices'])
        
        # 选择偏振分量
        if pol == 'x': 
            self.LG = LG_Ex
        elif pol == 'y': 
            self.LG = LG_Ey
        else:
            self.LG = LG_Ex + LG_Ey  # 或其他组合方式
        
        # 归一化
        if normalize:
            H, W, NMode = self.LG.shape
            for i in range(NMode):
                mode = self.LG[:, :, i]
                energy = np.sum(np.abs(mode)**2)
                if energy > 1:
                    self.LG[:, :, i] = mode / np.sqrt(energy)
        
        return self.LG
