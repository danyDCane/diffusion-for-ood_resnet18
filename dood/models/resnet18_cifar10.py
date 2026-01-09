import torch
import torch.nn as nn
import torchvision.models as models
import types


def ResNet18_CIFAR10(pretrained=True, num_classes=10):
    """ResNet18适配CIFAR-10，使用ImageNet预训练权重
    
    Args:
        pretrained (bool): 是否使用ImageNet预训练权重
        num_classes (int): 分类类别数，CIFAR-10为10
    
    Returns:
        model: 适配CIFAR-10的ResNet18模型，包含intermediate_forward方法
    """
    # 加载预训练ResNet18
    # 兼容新旧版本的torchvision
    try:
        # 新版本torchvision (>=0.13)
        if pretrained:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet18(weights=None)
    except (AttributeError, TypeError):
        # 旧版本torchvision (<0.13)
        model = models.resnet18(pretrained=pretrained)
    
    # 修改conv1以适应32x32输入
    # 从7x7 stride=2改为3x3 stride=1，padding=1
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # 重新初始化conv1权重（因为kernel size不同，无法直接使用预训练权重）
    nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
    
    # 移除maxpool（用Identity替代，避免过度下采样）
    model.maxpool = nn.Identity()
    
    # 修改fc层
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # 添加intermediate_forward方法，返回512维特征向量（在fc之前）
    def intermediate_forward(self, x):
        """提取中间特征，返回512维特征向量
        
        Args:
            x: 输入图像 (B, 3, 32, 32)
        
        Returns:
            features: 展平后的特征向量 (B, 512)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # 跳过maxpool（已经是Identity）
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    
    # 绑定方法到模型实例
    model.intermediate_forward = types.MethodType(intermediate_forward, model)
    
    return model

