import numpy as np
import torch

def sort_lp_modes(lm_indices, field_distributions, T=None):
    """
    将LP模式按照标准顺序排序
    
    参数:
    lm_indices: 列表或数组，包含每个模式的(l, m)索引
    field_distributions: 三维数组，形状为 (x_dim, y_dim, num_modes)，包含每个模式的场分布
    T: 传输矩阵，形状为 (num_modes, num_modes)，如果为None则不处理
    
    返回:
    sorted_lm_indices: 排序后的(l, m)索引列表
    sorted_field_distributions: 排序后的场分布数组
    sorted_T: 排序后的传输矩阵（如果输入了T）
    """
    # 将索引转换为列表以便处理
    if not isinstance(lm_indices, list):
        lm_list = [tuple(idx) for idx in lm_indices]
    else:
        lm_list = lm_indices.copy()
    
    # 获取模式数量
    num_modes = len(lm_list)
    
    # 创建索引列表
    indices = list(range(num_modes))
    
    # 按照LP模式标准排序规则排序:
    # 1. 先按l的绝对值从小到大
    # 2. 对于相同|l|的模式，按m从小到大
    # 3. 对于相同|l|和m的模式，按l从小到大（负值在前）
    indices.sort(key=lambda i: (abs(lm_list[i][0]), lm_list[i][1], lm_list[i][0]))
    
    # 重新排序场分布
    sorted_field_distributions = field_distributions[:, indices]
    
    # 重新排序索引
    sorted_lm_indices = [lm_list[i] for i in indices]
    
    # 如果有传输矩阵T，也进行相应的排序
    if T is not None:
        # 确保T是二维方阵
        if T.ndim == 2 and T.shape[0] == T.shape[1] == num_modes:
            # 使用排序索引对T的行和列进行重排
            sorted_T = T[np.ix_(indices, indices)]
        else:
            print(f"警告: 传输矩阵T的形状{T.shape}与模式数量{num_modes}不匹配")
            sorted_T = T
    else:
        sorted_T = None
    
    # 根据是否有T返回不同的结果
    if T is not None:
        return sorted_lm_indices, sorted_field_distributions, sorted_T
    else:
        return sorted_lm_indices, sorted_field_distributions
    
def clean_zero_modes(PIM, lm_index, pol='x'):
    """
    清除 PIM 中全为零的模式，并同步更新 lm_index。
    参数：
        PIM: torch.Tensor, shape = (N_pixel, N_modes)
        lm_index: list[(l, m)] 模式序号
        pol: 'x' 或 'y'，表示选择哪个偏振
    返回：
        PIM_clean, lm_index_clean
    """
    # 计算每个模式能量（复振幅平方和）
    power = torch.sum(torch.abs(PIM)**2, dim=0)
    nonzero_mask = power > 1e-12  # 能量非零的模式

    # 过滤
    PIM_clean = PIM[:, nonzero_mask]
    lm_index_clean = [lm_index[i] for i in range(len(lm_index)) if nonzero_mask[i]]

    print(f"已移除 {torch.sum(~nonzero_mask).item()} 个零模式，保留 {PIM_clean.shape[1]} 个有效模式。")
    return PIM_clean, lm_index_clean