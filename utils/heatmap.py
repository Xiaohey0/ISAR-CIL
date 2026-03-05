import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def save_feature_heatmap_on_image(original_img, feature_map, save_path, alpha=0.5, colormap='jet', reduce_mode='mean', select_idx=0):
    """
    将特征图叠加到原图上，并保存热度图

    Args:
        original_img (Tensor or ndarray): 原始图像 (C,H,W) or (H,W,C)
        feature_map (Tensor or ndarray): 特征图 (H,W)、(1,H,W)或(C_feat,H,W)
        save_path (str): 保存热度图的路径
        alpha (float): 热度图透明度 默认0.5
        colormap (str): 叠加用的颜色映射 默认'jet'
        reduce_mode (str): 'mean', 'max', or 'select' 特征融合方式
        select_idx (int): 如果reduce_mode='select'，选择第几个通道
    """
    # 将输入都转为numpy
    if isinstance(original_img, torch.Tensor):
        original_img = original_img.detach().cpu().numpy()
    if isinstance(feature_map, torch.Tensor):
        feature_map = feature_map.detach().cpu()
    else:
        feature_map = torch.from_numpy(feature_map)
    
    # 处理原图
    if original_img.ndim == 3:
        if original_img.shape[0] == 3 or original_img.shape[0] == 1:  # (C,H,W) -> (H,W,C)
            original_img = np.transpose(original_img, (1,2,0))
    if original_img.shape[-1] == 1:
        original_img = np.repeat(original_img, 3, axis=-1)  # 灰度图转RGB

    H, W = original_img.shape[:2]

    # 处理特征图
    if feature_map.ndim == 2:
        feature_map = feature_map.unsqueeze(0).unsqueeze(0)  # (H,W) -> (1,1,H,W)
    elif feature_map.ndim == 3:
        if feature_map.shape[0] == 1:  
            feature_map = feature_map.unsqueeze(0)  # (1,H,W) -> (1,1,H,W)
        else:
            feature_map = feature_map.unsqueeze(0)  # (C_feat,H,W) -> (1,C_feat,H,W)

    # 调整特征图尺寸
    feature_map = F.interpolate(feature_map, size=(H, W), mode='bilinear', align_corners=False)
    feature_map = feature_map.squeeze(0)  # 变成 (C_feat, H, W)

    # 多通道融合处理 
    if feature_map.dim() == 3:
        if reduce_mode == 'mean':
            feature_map = feature_map.mean(dim=0)
        elif reduce_mode == 'max':
            feature_map = feature_map.max(dim=0)[0]
        elif reduce_mode == 'select':
            feature_map = feature_map[select_idx]
        else:
            raise ValueError(f"Invalid reduce_mode {reduce_mode}. Choose from ['mean', 'max', 'select'].")

    # 现在feature_map一定是 (H,W)

    feature_map = feature_map.cpu().numpy()

    # 标准化特征图
    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)

    # 应用colormap
    heatmap = cm.get_cmap(colormap)(feature_map)[:, :, :3]  # 取RGB通道

    # 叠加原图和热度图
    overlay = (1 - alpha) * original_img + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)  # 保证数值在[0,1]

    # 绘制并保存
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(overlay)
    plt.tight_layout(pad=0)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
