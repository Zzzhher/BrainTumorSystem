import torch
import torch.nn as nn
from torchvision import models, transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import time
import yaml

# 加载配置文件
def load_config(config_path='config.yml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# 加载配置
config = load_config()

# 类别标签
CLASSES = config['classes']

# 数据预处理transform
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 统一输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 模型定义
def get_model(model_name, num_classes=4):
    if model_name == 'resnet50':
        model = models.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'resnet34':
        model = models.resnet34(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'efficientnet_b1':
        model = models.efficientnet_b1(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model

# 加载模型
def load_model(model_path, model_name, num_classes=4):
    model = get_model(model_name, num_classes)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

# 绘制置信度条形图
def plot_confidence_bar(pred_probs):
    plt.figure(figsize=(10, 6))
    bars = plt.bar(CLASSES, pred_probs, color=['blue', 'green', 'red', 'purple'])
    plt.xlabel('类别')
    plt.ylabel('置信度')
    plt.title('预测置信度分布')
    plt.ylim(0, 1)
    
    # 在条形上方添加数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', ha='center', va='bottom')
    
    return plt

# 保存日志
def save_log(log_dir, content):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'log_{time.strftime("%Y%m%d_%H%M%S")}.txt')
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(content)
    return log_file

# GradCAM 实现
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        self.hook_handles = []
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        
        # 查找目标层并注册钩子
        found = False
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                self.hook_handles.append(module.register_forward_hook(forward_hook))
                # 使用register_full_backward_hook替代register_backward_hook
                self.hook_handles.append(module.register_full_backward_hook(backward_hook))
                found = True
                break
        
        if not found:
            # 如果找不到指定层，使用模型的最后一个卷积层
            last_conv_layer = None
            for name, module in reversed(list(self.model.named_modules())):
                if isinstance(module, nn.Conv2d):
                    last_conv_layer = name
                    break
            
            if last_conv_layer:
                for name, module in self.model.named_modules():
                    if name == last_conv_layer:
                        self.hook_handles.append(module.register_forward_hook(forward_hook))
                        # 使用register_full_backward_hook替代register_backward_hook
                        self.hook_handles.append(module.register_full_backward_hook(backward_hook))
                        break
    
    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
    
    def __call__(self, x, class_idx=None):
        # 前向传播
        outputs = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(outputs, dim=1)
        
        # 计算目标类的梯度
        self.model.zero_grad()
        one_hot = torch.zeros_like(outputs)
        one_hot[0, class_idx] = 1
        outputs.backward(gradient=one_hot, retain_graph=True)
        
        # 检查梯度和激活是否存在
        if self.gradients is None or self.activations is None:
            raise ValueError("无法获取梯度或激活值，请检查模型结构和目标层设置")
        
        # 计算权重
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1)
        cam = torch.relu(cam)
        
        # 归一化
        cam = cam / (torch.max(cam) + 1e-8)
        
        # 调整尺寸
        cam = torch.nn.functional.interpolate(
            cam.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
        ).squeeze()
        
        return cam.cpu().numpy()

# 生成GradCAM热力图
def generate_gradcam(model, image_tensor, class_idx=None):
    # 根据模型类型选择目标层
    if 'resnet' in model.__class__.__name__.lower():
        # ResNet50的最后一个卷积层
        target_layer = 'layer4.2.conv3'
    elif 'efficientnet' in model.__class__.__name__.lower():
        target_layer = 'features.8.0.conv'
    else:
        # 对于其他模型，尝试使用最后一个卷积层
        target_layer = None
    
    gradcam = GradCAM(model, target_layer)
    cam = gradcam(image_tensor, class_idx)
    gradcam.remove_hooks()
    return cam

# 叠加热力图到原图
def overlay_heatmap(original_image, cam):
    # 将PIL图像转换为numpy数组
    img = np.array(original_image.resize((224, 224)))
    
    # 生成热力图
    heatmap = np.uint8(255 * cam)
    heatmap = np.stack([heatmap, heatmap, heatmap], axis=-1)
    
    # 应用颜色映射
    import cv2
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    
    # 叠加
    superimposed_img = heatmap * 0.4 + np.float32(img) / 255
    superimposed_img = np.uint8(255 * superimposed_img)
    
    return Image.fromarray(superimposed_img)