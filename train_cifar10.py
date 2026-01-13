import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import wandb

from dood.models.resnet18_cifar10 import ResNet18_CIFAR10
from dood.utils.diffusion import get_diffusion_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train ResNet18 with Diffusion OOD detection on CIFAR-10')
    
    # 数据相关
    parser.add_argument('--data_root', type=str, default='../datasets', help='Root directory for CIFAR-10 dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # 模型相关
    parser.add_argument('--pretrained', action='store_true', default=True, help='Use ImageNet pretrained weights')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes (CIFAR-10: 10)')
    
    # 训练相关
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr_backbone', type=float, default=0.1, help='Learning rate for backbone')
    parser.add_argument('--lr_diffusion', type=float, default=5e-5, help='Learning rate for diffusion model')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--lambda_diff', type=float, default=1.0, help='Weight for diffusion loss')
    
    # Diffusion模型相关
    parser.add_argument('--diffusion_denoiser_channels', type=int, default=512, help='Diffusion denoiser channels')
    parser.add_argument('--num_diffusion_steps', type=int, default=1000, help='Number of diffusion steps')
    
    # 其他
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--save_freq', type=int, default=10, help='Frequency to save checkpoints (epochs)')
    parser.add_argument('--eval_freq', type=int, default=5, help='Frequency to evaluate (epochs)')
    
    # Wandb相关
    parser.add_argument('--use_wandb', action='store_true', default=True, help='Use wandb for logging')
    parser.add_argument('--wandb_project', type=str, default='diffusion-ood-cifar10', help='Wandb project name')
    parser.add_argument('--wandb_name', type=str, default=None, help='Wandb run name (default: auto-generated)')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Wandb entity/team name')
    
    return parser.parse_args()


def get_cifar10_loaders(data_root, batch_size, num_workers):
    """加载CIFAR-10数据集"""
    # 训练集transform
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 测试集transform
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 加载数据集
    train_dataset = datasets.CIFAR10(
        root=data_root, 
        train=True, 
        download=True, 
        transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root=data_root, 
        train=False, 
        download=True, 
        transform=transform_test
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def evaluate_classification(model, test_loader, device):
    """评估分类准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            logits = model(data)
            _, predicted = torch.max(logits.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def train(args):
    """主训练函数"""
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 初始化 wandb
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            entity=args.wandb_entity,
            config=vars(args),
            reinit=True
        )
        print(f'Wandb initialized: project={args.wandb_project}, name={wandb.run.name}')
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 加载数据
    print('Loading CIFAR-10 dataset...')
    train_loader, test_loader = get_cifar10_loaders(
        args.data_root, 
        args.batch_size, 
        args.num_workers
    )
    
    # 初始化模型
    print('Initializing models...')
    backbone = ResNet18_CIFAR10(
        pretrained=args.pretrained, 
        num_classes=args.num_classes
    ).to(device)
    
    diffusion_model = get_diffusion_model(
        ft_size=512,  # ResNet18 intermediate features dimension
        denoiser_type="unet0d",
        diffusion_denoiser_channels=args.diffusion_denoiser_channels,
        num_diffusion_steps=args.num_diffusion_steps,
    ).to(device)
    
    # 优化器
    optimizer_backbone = torch.optim.SGD(
        backbone.parameters(), 
        lr=args.lr_backbone, 
        momentum=args.momentum, 
        weight_decay=args.weight_decay
    )
    optimizer_diffusion = torch.optim.Adam(
        diffusion_model.parameters(), 
        lr=args.lr_diffusion
    )
    
    # 学习率调度器（可选）
    scheduler_backbone = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_backbone, 
        T_max=args.epochs
    )
    
    # 训练循环
    print('Starting training...')
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        backbone.train()
        diffusion_model.train()
        
        epoch_loss_cls = 0.0
        epoch_loss_diff = 0.0
        epoch_total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch_idx, (data, targets) in enumerate(pbar):
            data, targets = data.to(device), targets.to(device)
            
            # 提取特征
            latents = backbone.intermediate_forward(data)
            
            # 分类损失
            logits = backbone.fc(latents)
            loss_cls = F.cross_entropy(logits, targets)
            
            # 先训练 backbone（分类任务）
            optimizer_backbone.zero_grad()
            loss_cls.backward()
            optimizer_backbone.step()
            
            # 然后单独训练 diffusion model
            # 断开梯度连接，避免影响已经更新过的 backbone
            latents_for_diff = latents.detach().requires_grad_(True)
            # 归一化特征（这会自动更新归一化统计信息）
            latents_normalized = diffusion_model.normalize(latents_for_diff)

            # 检查归一化后的特征范围（Normalized Feature Statistics）
            if batch_idx % 50 == 0:
                norm_mean = latents_normalized.mean().item()
                norm_std = latents_normalized.std().item()
                norm_min = latents_normalized.min().item()
                norm_max = latents_normalized.max().item()
                
                # 判断归一化是否正常
                scale_to_unit_gauss = diffusion_model.normalization.scale_to_unit_gauss
                if scale_to_unit_gauss:
                    # 如果使用 scale_to_unit_gauss=True：Std 应该接近 1.0
                    std_status = "OK" if abs(norm_std - 1.0) < 0.2 else "WARNING (std not ~1.0!)"
                else:
                    # 如果使用 MinMax：Std 应该适中 (0.2 ~ 0.6)，不能太小 (0.001)
                    if norm_std < 0.001:
                        std_status = "WARNING (std too small!)"
                    elif 0.2 <= norm_std <= 0.6:
                        std_status = "OK"
                    else:
                        std_status = "WARNING (std out of range!)"
                
                print(f'Normalized features - min: {norm_min:.4f}, '
                      f'max: {norm_max:.4f}, '
                      f'mean: {norm_mean:.4f}, '
                      f'std: {norm_std:.4f} [{std_status}]')
                
                # 记录到 wandb（只记录关键指标）
                if args.use_wandb:
                    wandb.log({
                        'normalized_features/std': norm_std,  # 最重要的指标
                        'normalized_features/mean': norm_mean,
                    })

            loss_diff = diffusion_model.get_loss_iter(latents_normalized)
            
            # 从 p_losses 中获取监控信息（每50个batch记录一次以减少wandb开销）
            if batch_idx % 50 == 0 and hasattr(diffusion_model.diffusion_process, '_last_snr_info'):
                monitor_info = diffusion_model.diffusion_process._last_snr_info
                if args.use_wandb:
                    # 只记录关键指标
                    log_dict = {
                        'snr/snr': monitor_info['snr'],  # 最重要的SNR指标
                    }
                    
                    # 添加 Identity Correlation 指标（只记录平均值）
                    if 'identity_correlation' in monitor_info:
                        log_dict['trivial_solution/identity_correlation'] = monitor_info['identity_correlation']
                    
                    wandb.log(log_dict)
                
                # 判断SNR是否正常（应该接近1.0，如果小于0.1则警告）
                snr_status = "OK" if monitor_info['snr'] > 0.1 else "WARNING (too low!)"
                print(f'SNR: {monitor_info["snr"]:.4f} [{snr_status}], '
                      f'||x_0||: {monitor_info["x0_norm"]:.4f} ± {monitor_info["x0_norm_std"]:.4f}, '
                      f'||epsilon||: {monitor_info["noise_norm"]:.4f} ± {monitor_info["noise_norm_std"]:.4f}, '
                      f'Expected: {monitor_info["expected_norm"]:.4f}')
                
                # 打印 Identity Correlation 信息
                if 'identity_correlation' in monitor_info:
                    trivial_status = "WARNING (Trivial Solution!)" if monitor_info['is_trivial_solution'] else "OK"
                    print(f'Identity Correlation: {monitor_info["identity_correlation"]:.4f} ± {monitor_info["identity_correlation_std"]:.4f} [{trivial_status}], '
                          f'Max: {monitor_info["identity_correlation_max"]:.4f}')
            
            # 反向传播 diffusion loss
            optimizer_diffusion.zero_grad()
            loss_diff.backward()
            # 检查梯度
            total_norm = 0
            param_count = 0
            for name, param in diffusion_model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
                else:
                    print(f"Warning: {name} has no gradient!")

            if param_count > 0:
                total_norm = total_norm ** (1. / 2)
                if batch_idx % 100 == 0:  # 每50个batch打印一次
                    print(f'Diffusion gradient norm: {total_norm:.6f}, params with grad: {param_count}')
            optimizer_diffusion.step()
            
            # 记录损失
            epoch_loss_cls += loss_cls.item()
            epoch_loss_diff += loss_diff.item()
            epoch_total_loss += loss_cls.item() + args.lambda_diff * loss_diff.item()
            
            # 移除每个batch的wandb记录，只保留epoch级别的记录以减少开销
            
            # 更新进度条
            pbar.set_postfix({
                'loss_cls': f'{loss_cls.item():.4f}',
                'loss_diff': f'{loss_diff.item():.4f}',
                'total': f'{loss_cls.item() + args.lambda_diff * loss_diff.item():.4f}'
            })
        
        # 更新学习率
        scheduler_backbone.step()
        
        # 计算平均损失
        avg_loss_cls = epoch_loss_cls / len(train_loader)
        avg_loss_diff = epoch_loss_diff / len(train_loader)
        avg_total_loss = epoch_total_loss / len(train_loader)
        
        # 获取学习率
        current_lr_backbone = optimizer_backbone.param_groups[0]['lr']
        current_lr_diffusion = optimizer_diffusion.param_groups[0]['lr']
        
        # 记录epoch级别的指标到wandb
        log_dict = {
            'train/epoch_loss_cls': avg_loss_cls,
            'train/epoch_loss_diff': avg_loss_diff,
            'train/epoch_total_loss': avg_total_loss,
            'train/lr_backbone': current_lr_backbone,
            'train/lr_diffusion': current_lr_diffusion,
            'epoch': epoch + 1,
        }
        
        # 添加监控指标到epoch日志（只记录关键指标）
        if hasattr(diffusion_model.diffusion_process, '_last_snr_info'):
            monitor_info = diffusion_model.diffusion_process._last_snr_info
            log_dict.update({
                'snr/snr': monitor_info['snr'],  # 只保留最重要的SNR指标
            })
            
            # 添加 Identity Correlation 指标（只记录平均值）
            if 'identity_correlation' in monitor_info:
                log_dict['trivial_solution/identity_correlation'] = monitor_info['identity_correlation']
        
        # 添加归一化特征统计信息到epoch日志（只记录std）
        if args.use_wandb:
            try:
                with torch.no_grad():
                    sample_data, _ = next(iter(train_loader))
                    sample_data = sample_data.to(device)
                    sample_latents = backbone.intermediate_forward(sample_data)
                    sample_latents_normalized = diffusion_model.normalize(sample_latents.detach())
                    
                    norm_std = sample_latents_normalized.std().item()
                    log_dict['normalized_features/std'] = norm_std  # 只记录最重要的std
            except:
                pass
        
        print(f'\nEpoch {epoch+1}/{args.epochs}:')
        print(f'  Loss_cls: {avg_loss_cls:.4f}, Loss_diff: {avg_loss_diff:.4f}, Total: {avg_total_loss:.4f}')
        
        # 评估
        if (epoch + 1) % args.eval_freq == 0:
            accuracy = evaluate_classification(backbone, test_loader, device)
            print(f'  Test Accuracy: {accuracy:.2f}%')
            log_dict['test/accuracy'] = accuracy
            
            # 保存最佳模型
            if accuracy > best_acc:
                best_acc = accuracy
                print(f'  New best accuracy: {best_acc:.2f}%')
                log_dict['test/best_accuracy'] = best_acc
        
        # 记录到wandb
        if args.use_wandb:
            wandb.log(log_dict)
        
        # 保存checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'backbone_state_dict': backbone.state_dict(),
                'diffusion_state_dict': diffusion_model.state_dict(),
                'optimizer_backbone_state_dict': optimizer_backbone.state_dict(),
                'optimizer_diffusion_state_dict': optimizer_diffusion.state_dict(),
                'best_acc': best_acc,
            }
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f'  Checkpoint saved to {checkpoint_path}')
    
    # 保存最终模型
    final_checkpoint = {
        'epoch': args.epochs,
        'backbone_state_dict': backbone.state_dict(),
        'diffusion_state_dict': diffusion_model.state_dict(),
        'optimizer_backbone_state_dict': optimizer_backbone.state_dict(),
        'optimizer_diffusion_state_dict': optimizer_diffusion.state_dict(),
        'best_acc': best_acc,
    }
    final_path = os.path.join(args.save_dir, 'final_model.pth')
    torch.save(final_checkpoint, final_path)
    print(f'\nFinal model saved to {final_path}')
    print(f'Best accuracy: {best_acc:.2f}%')
    
    # 记录最终结果到wandb
    if args.use_wandb:
        wandb.log({'final/best_accuracy': best_acc})
        wandb.finish()


if __name__ == '__main__':
    args = parse_args()
    train(args)


