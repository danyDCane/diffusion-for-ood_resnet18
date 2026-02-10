import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm

from dood.models.resnet18_cifar10 import ResNet18_CIFAR10
from dood.utils.diffusion import get_diffusion_model, get_diffusion_scores, load_diffusion_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate OOD detection on CIFAR-10')
    
    # 模型相关
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint file')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    
    # 数据相关
    parser.add_argument('--data_root', type=str, default='/home/server5090/Desktop/M11307320/datasets', help='Root directory for datasets')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # OOD数据集相关
    parser.add_argument('--ood_dataset', type=str, default='LSUN', 
                       choices=['cifar100', 'LSUN', 'SVHN'],
                       help='OOD dataset to use')
    
    # Diffusion相关
    parser.add_argument('--diffusion_denoiser_channels', type=int, default=512, help='Diffusion denoiser channels')
    parser.add_argument('--num_diffusion_steps', type=int, default=1000, help='Number of diffusion steps')
    parser.add_argument('--ood_eval_scores_type', type=str, default='eps_mse',
                       choices=['eps_mse', 'eps_cos', 'recon_mse', 'bpd'],
                       help='Type of OOD scoring function')
    parser.add_argument('--num_eval_steps', type=int, default=25, help='Number of diffusion steps for evaluation')
    
    # 其他
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    
    # 噪声OOD评估相关
    parser.add_argument('--use_noise_ood', action='store_true', 
                       help='Use noise as OOD dataset for evaluation')
    parser.add_argument('--noise_samples', type=int, default=10000, 
                       help='Number of noise samples to generate for OOD evaluation')
    
    return parser.parse_args()


def compute_auroc(id_scores, ood_scores):
    """计算AUROC"""
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros_like(id_scores), np.ones_like(ood_scores)])
    auroc = roc_auc_score(labels, scores)
    return auroc


def compute_fpr_at_tpr(id_scores, ood_scores, tpr=0.95):
    """计算FPR@TPR (False Positive Rate at True Positive Rate)，使用 sklearn 的 roc_curve（更精确）"""
    # 1. 建立標籤：ID=0, OOD=1
    y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    y_scores = np.concatenate([id_scores, ood_scores])
    
    # 2. 處理分數方向：確保 OOD 分數比 ID 高
    if np.mean(id_scores) > np.mean(ood_scores):
        y_scores = -y_scores
    
    # 3. 計算 ROC curve
    fpr, tpr_array, thresholds = roc_curve(y_true, y_scores)
    
    # 4. 找到 TPR >= tpr 的第一個點
    idx = np.searchsorted(tpr_array, tpr)
    
    # 5. 處理邊界情況
    if idx == 0:
        # 如果第一個點的 TPR 就已經 >= tpr，返回該點的 FPR
        return fpr[0]
    elif idx >= len(fpr):
        # 如果所有點的 TPR 都 < tpr，返回最後一個點的 FPR
        return fpr[-1]
    
    # 6. 線性插值（更精確）
    # 如果 tpr_array[idx-1] < tpr < tpr_array[idx]，進行插值
    if idx > 0 and tpr_array[idx-1] < tpr < tpr_array[idx]:
        # 線性插值
        tpr_diff = tpr_array[idx] - tpr_array[idx-1]
        if tpr_diff > 0:
            weight = (tpr - tpr_array[idx-1]) / tpr_diff
            fpr_interpolated = fpr[idx-1] + weight * (fpr[idx] - fpr[idx-1])
            return fpr_interpolated
    
    # 7. 如果恰好等於，直接返回
    return fpr[idx]


def get_cifar10_loader(data_root, batch_size, num_workers, train=False):
    """加载CIFAR-10数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    dataset = datasets.CIFAR10(
        root=data_root,
        train=train,
        download=True,
        transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader


def get_cifar100_loader(data_root, batch_size, num_workers):
    """加载CIFAR-100数据集作为OOD"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    dataset = datasets.CIFAR100(
        root=data_root,
        train=False,
        download=True,
        transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader


def get_svhn_loader(data_root, batch_size, num_workers):
    """加载SVHN数据集作为OOD"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    ])
    
    dataset = datasets.SVHN(
        root=data_root,
        split='test',
        download=True,
        transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader


def get_lsun_loader(data_root, batch_size, num_workers):
    """加载LSUN数据集作为OOD
    
    处理方式参考FOOGD-main项目：
    - 使用ImageFolder加载LSUN数据集（需要子目录结构）
    - 使用RandomCrop(32, padding=4)来将LSUN图片裁剪到32x32
    - 使用CIFAR-10的normalization参数，因为模型是用CIFAR-10训练的
    """
    # 使用FOOGD-main的normalization参数（与CIFAR-10标准参数相同，只是表示方式不同）
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]
    
    # 按照FOOGD-main的方式处理LSUN（注意transform顺序）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomCrop(32, padding=4),  # RandomCrop在tensor上操作
    ])
    
    # LSUN数据集路径
    lsun_path = os.path.join(data_root, 'LSUN')
    
    # 检查路径是否存在
    if not os.path.exists(lsun_path):
        # 尝试其他可能的路径
        alternative_paths = [
            os.path.join(data_root, 'lsun'),
            os.path.join(data_root, 'lsun_resize'),
            data_root,  # 直接使用data_root
        ]
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                lsun_path = alt_path
                break
        else:
            raise FileNotFoundError(
                f'LSUN dataset not found. Please ensure LSUN dataset is located at one of:\n'
                f'  - {os.path.join(data_root, "LSUN")}\n'
                f'  - {os.path.join(data_root, "lsun")}\n'
                f'  - {os.path.join(data_root, "lsun_resize")}\n'
                f'  - {data_root}\n'
                f'Or specify the correct path using --data_root argument.'
            )
    
    # 使用ImageFolder加载LSUN数据集（需要子目录结构）
    dataset = datasets.ImageFolder(
        root=lsun_path,
        transform=transform
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f'Loaded LSUN dataset from {lsun_path} with {len(dataset)} images')
    
    return loader


def get_ood_loader(args):
    """根据参数获取OOD数据集loader"""
    if args.ood_dataset == 'cifar100':
        return get_cifar100_loader(args.data_root, args.batch_size, args.num_workers)
    elif args.ood_dataset == 'svhn':
        return get_svhn_loader(args.data_root, args.batch_size, args.num_workers)
    elif args.ood_dataset == 'LSUN':
        return get_lsun_loader(args.data_root, args.batch_size, args.num_workers)
    else:
        raise NotImplementedError(f'OOD dataset {args.ood_dataset} not implemented')


class NoiseDataset(Dataset):
    """生成随机噪声图像作为OOD数据集（适配CIFAR-10）"""
    def __init__(self, num_samples, image_size=(32, 32), num_channels=3):
        """
        Args:
            num_samples: 生成的噪声样本数量
            image_size: 图像尺寸 (height, width)，默认 (32, 32) 用于CIFAR-10
            num_channels: 图像通道数，默认3（RGB）
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_channels = num_channels
        
        # CIFAR-10的normalization参数
        self.normalize = transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465), 
            std=(0.2023, 0.1994, 0.2010)
        )
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 生成随机噪声图像 [0, 1] 范围（ToTensor后的范围）
        # 使用均匀分布生成随机噪声
        noise_image = torch.rand(self.num_channels, self.image_size[0], self.image_size[1])
        
        # 应用normalization（与CIFAR-10测试数据一致）
        noise_image = self.normalize(noise_image)
        
        # 返回噪声图像和dummy标签（OOD不需要真实标签）
        return noise_image, 0


def get_noise_loader(num_samples, batch_size, num_workers, image_size=(32, 32)):
    """创建噪声数据集的DataLoader（适配CIFAR-10）"""
    noise_dataset = NoiseDataset(num_samples=num_samples, image_size=image_size)
    
    loader = DataLoader(
        noise_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader


def evaluate_ood_detection(args):
    """评估OOD检测性能"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 加载模型
    print('Loading models...')
    backbone = ResNet18_CIFAR10(
        pretrained=False,  # 不使用预训练，因为会从checkpoint加载
        num_classes=args.num_classes
    ).to(device)
    
    diffusion_model = get_diffusion_model(
        ft_size=512,
        denoiser_type="unet0d",
        diffusion_denoiser_channels=args.diffusion_denoiser_channels,
        num_diffusion_steps=args.num_diffusion_steps,
    ).to(device)
    
    # 加载checkpoint
    print(f'Loading checkpoint from {args.checkpoint}...')
    checkpoint = torch.load(args.checkpoint, map_location=device)
    backbone.load_state_dict(checkpoint['backbone_state_dict'])
    diffusion_model.load_state_dict(checkpoint['diffusion_state_dict'])
    
    backbone.eval()
    diffusion_model.eval()
    
    # 准备diffusion steps
    diffusion_steps = list(range(args.num_eval_steps))
    
    # 加载数据集
    print('Loading datasets...')
    id_loader = get_cifar10_loader(args.data_root, args.batch_size, args.num_workers, train=False)
    
    # 根据参数决定使用哪个OOD数据集
    if args.use_noise_ood:
        # 如果启用噪声OOD，使用噪声作为OOD数据集
        ood_loader = get_noise_loader(
            args.noise_samples,
            args.batch_size,
            args.num_workers,
            image_size=(32, 32)  # CIFAR-10尺寸
        )
        ood_dataset_name = 'Noise'
    else:
        # 否则使用常规OOD数据集
        ood_loader = get_ood_loader(args)
        ood_dataset_name = args.ood_dataset
    
    print(f'Evaluating on ID dataset (CIFAR-10 test set)...')
    id_scores = []
    with torch.no_grad():
        for data, _ in tqdm(id_loader, desc='ID samples'):
            data = data.to(device)
            latents = backbone.intermediate_forward(data)
            
            # 获取diffusion scores
            scores, _ = get_diffusion_scores(
                latents,
                diffusion_model,
                diffusion_steps,
                args.ood_eval_scores_type,
                normalize=True,
                dtype=torch.float32
            )
            # scores可能是标量或tensor，确保转换为numpy数组
            if isinstance(scores, torch.Tensor):
                scores_np = scores.cpu().numpy()
            else:
                scores_np = np.array([scores])
            # 如果是多维，展平
            id_scores.append(scores_np.flatten())
    
    id_scores = np.concatenate(id_scores)
    id_scores_original = id_scores.copy()  # 保存原始ID分数
    
    # 评估OOD数据集
    print(f'Evaluating on OOD dataset ({ood_dataset_name})...')
    ood_scores = []
    with torch.no_grad():
        for data, _ in tqdm(ood_loader, desc='OOD samples'):
            data = data.to(device)
            latents = backbone.intermediate_forward(data)
            
            # 获取diffusion scores
            scores, _ = get_diffusion_scores(
                latents,
                diffusion_model,
                diffusion_steps,
                args.ood_eval_scores_type,
                normalize=True,
                dtype=torch.float32
            )
            # scores可能是标量或tensor，确保转换为numpy数组
            if isinstance(scores, torch.Tensor):
                scores_np = scores.cpu().numpy()
            else:
                scores_np = np.array([scores])
            # 如果是多维，展平
            ood_scores.append(scores_np.flatten())
    
    ood_scores = np.concatenate(ood_scores)
    
    # 计算指标和统计信息
    print('\nComputing metrics...')
    
    # 如果ID分数更高，需要反转（用于计算指标）
    id_scores_for_metric = id_scores_original.copy()
    ood_scores_for_metric = ood_scores.copy()
    if np.mean(id_scores_original) > np.mean(ood_scores):
        print('Warning: ID scores are higher than OOD scores. Inverting scores for metric calculation.')
        id_scores_for_metric = -id_scores_original
        ood_scores_for_metric = -ood_scores
    
    # 计算指标
    auroc = compute_auroc(id_scores_for_metric, ood_scores_for_metric)
    fpr95 = compute_fpr_at_tpr(id_scores_for_metric, ood_scores_for_metric, tpr=0.95)
    
    # 计算OOD统计信息（基于原始分数）
    ood_mean = np.mean(ood_scores)
    ood_std = np.std(ood_scores)
    ood_min = np.min(ood_scores)
    ood_max = np.max(ood_scores)
    ood_num_samples = len(ood_scores)
    
    # ID统计信息
    id_mean = np.mean(id_scores_original)
    id_std = np.std(id_scores_original)
    id_min = np.min(id_scores_original)
    id_max = np.max(id_scores_original)
    id_num_samples = len(id_scores_original)
    
    # 打印详细结果
    print(f'\n{"="*80}')
    print(f'Results Summary:')
    print(f'{"="*80}')
    print(f'{"Dataset":<20} {"Mean Score":<15} {"Std":<15} {"Min":<15} {"Max":<15} {"Num Samples":<15}')
    print(f'{"-"*80}')
    print(f'{"ID (CIFAR-10)":<20} {id_mean:<15.6f} {id_std:<15.6f} {id_min:<15.6f} {id_max:<15.6f} {id_num_samples:<15}')
    print(f'{"OOD (" + ood_dataset_name + ")":<20} {ood_mean:<15.6f} {ood_std:<15.6f} {ood_min:<15.6f} {ood_max:<15.6f} {ood_num_samples:<15}')
    print(f'{"-"*80}')
    print(f'\nMetrics:')
    print(f'  AUROC: {auroc:.6f}')
    print(f'  FPR@95%TPR: {fpr95:.6f}')
    print(f'{"="*80}')
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存统计信息到CSV
    results_dict = {
        'dataset': ['ID (CIFAR-10)', f'OOD ({ood_dataset_name})'],
        'mean_score': [id_mean, ood_mean],
        'std_score': [id_std, ood_std],
        'min_score': [id_min, ood_min],
        'max_score': [id_max, ood_max],
        'num_samples': [id_num_samples, ood_num_samples],
        'score_type': [args.ood_eval_scores_type, args.ood_eval_scores_type],
        'auroc': [auroc, auroc],  # 两个数据集共享同一个AUROC
        'fpr95': [fpr95, fpr95]   # 两个数据集共享同一个FPR95
    }
    
    df = pd.DataFrame(results_dict)
    csv_filename = f'ood_scores_{ood_dataset_name.lower()}_{args.ood_eval_scores_type}.csv'
    csv_path = os.path.join(args.output_dir, csv_filename)
    df.to_csv(csv_path, index=False)
    print(f'\nResults saved to CSV: {csv_path}')
    
    # 保存完整的分数数组到numpy文件
    npz_filename = f'ood_scores_{ood_dataset_name.lower()}_{args.ood_eval_scores_type}.npz'
    npz_path = os.path.join(args.output_dir, npz_filename)
    np.savez(
        npz_path,
        id_scores=id_scores_original,  # 保存原始ID分数（未反转）
        ood_scores=ood_scores,  # 保存原始OOD分数（未反转）
        id_mean=id_mean,
        id_std=id_std,
        id_min=id_min,
        id_max=id_max,
        ood_mean=ood_mean,
        ood_std=ood_std,
        ood_min=ood_min,
        ood_max=ood_max,
        auroc=auroc,
        fpr95=fpr95,
        ood_dataset=ood_dataset_name,
        score_type=args.ood_eval_scores_type,
        num_eval_steps=args.num_eval_steps
    )
    print(f'Full scores saved to NPZ: {npz_path}')


if __name__ == '__main__':
    args = parse_args()
    evaluate_ood_detection(args)

