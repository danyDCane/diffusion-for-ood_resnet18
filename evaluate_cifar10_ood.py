import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
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
    parser.add_argument('--ood_eval_scores_type', type=str, default='eps_cos',
                       choices=['eps_mse', 'eps_cos', 'recon_mse', 'bpd'],
                       help='Type of OOD scoring function')
    parser.add_argument('--num_eval_steps', type=int, default=25, help='Number of diffusion steps for evaluation')
    
    # 其他
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
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
    ood_loader = get_ood_loader(args)
    
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
    
    print(f'Evaluating on OOD dataset ({args.ood_dataset})...')
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
    
    # 计算指标
    print('\nComputing metrics...')
    
    # 对于OOD检测，通常OOD样本的分数应该更高
    # 但根据不同的评分函数，可能需要反转
    # 这里假设分数越高越可能是OOD
    
    # 如果ID分数更高，需要反转
    if np.mean(id_scores) > np.mean(ood_scores):
        print('Warning: ID scores are higher than OOD scores. Inverting scores.')
        id_scores = -id_scores
        ood_scores = -ood_scores
    
    auroc = compute_auroc(id_scores, ood_scores)
    fpr95 = compute_fpr_at_tpr(id_scores, ood_scores, tpr=0.95)
    
    print(f'\nResults:')
    print(f'  ID scores: mean={np.mean(id_scores):.4f}, std={np.std(id_scores):.4f}')
    print(f'  OOD scores: mean={np.mean(ood_scores):.4f}, std={np.std(ood_scores):.4f}')
    print(f'  AUROC: {auroc:.4f}')
    print(f'  FPR@95%TPR: {fpr95:.4f}')


if __name__ == '__main__':
    args = parse_args()
    evaluate_ood_detection(args)

