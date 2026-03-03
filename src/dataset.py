# 数据加载与预处理
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# 类别标签
CLASSES = [' glioma', ' meningioma', ' no_tumor', ' pituitary']

# 训练数据增强transform
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 统一输入尺寸
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomVerticalFlip(),  # 随机垂直翻转
    transforms.RandomRotation(10),  # 随机旋转
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 色彩调整
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # 高斯模糊
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 测试数据transform
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 统一输入尺寸
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=train_transform, balance=False):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 遍历数据集目录
        class_images = {i: [] for i in range(len(CLASSES))}
        for class_idx, class_name in enumerate(CLASSES):
            class_dir = os.path.join(root_dir, class_name.strip())
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        class_images[class_idx].append(os.path.join(class_dir, img_name))
        
        # 数据均衡处理
        if balance:
            # 计算每个类别的样本数量
            class_counts = [len(images) for images in class_images.values()]
            max_count = max(class_counts)
            
            # 对每个类别进行采样，使其样本数量达到最大值
            for class_idx, images in class_images.items():
                if len(images) < max_count:
                    # 重复采样
                    repeated_images = images * (max_count // len(images))
                    remaining = max_count - len(repeated_images)
                    if remaining > 0:
                        repeated_images.extend(images[:remaining])
                    self.image_paths.extend(repeated_images)
                    self.labels.extend([class_idx] * max_count)
                else:
                    self.image_paths.extend(images)
                    self.labels.extend([class_idx] * len(images))
        else:
            # 不进行均衡处理
            for class_idx, images in class_images.items():
                self.image_paths.extend(images)
                self.labels.extend([class_idx] * len(images))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 数据加载器
def get_dataloader(root_dir, batch_size=32, shuffle=True, is_test=False, balance=False):
    transform = test_transform if is_test else train_transform
    dataset = BrainTumorDataset(root_dir, transform=transform, balance=balance)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
