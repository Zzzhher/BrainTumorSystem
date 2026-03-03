import torch
from PIL import Image
import numpy as np
import sys
import os

# 添加 src 目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from .utils import load_model, CLASSES, inference_transform

class InferenceEngine:
    def __init__(self):
        self.models = {}
    
    def load_model(self, model_path, model_name):
        """加载模型"""
        model = load_model(model_path, model_name)
        self.models[model_name] = model
        return model
    
    def preprocess_image(self, image_path):
        """预处理图像"""
        image = Image.open(image_path).convert('RGB')
        image_tensor = inference_transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # 添加批次维度
        return image_tensor, image
    
    def predict(self, model_name, image_path):
        """进行预测"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.models[model_name]
        image_tensor, original_image = self.preprocess_image(image_path)
        
        # 进行预测
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1).squeeze().numpy()
            predicted_class = np.argmax(probabilities)
        
        result = {
            'class_name': CLASSES[predicted_class].strip(),
            'confidence': float(probabilities[predicted_class]),
            'probabilities': probabilities.tolist(),
            'predicted_class': predicted_class
        }
        
        return result, original_image
    
    def get_available_models(self):
        """获取已加载的模型列表"""
        return list(self.models.keys())

# 示例用法
if __name__ == "__main__":
    engine = InferenceEngine()
    # 加载模型
    import os
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    model_path = os.path.join(models_dir, 'resnet50_best.pth')
    engine.load_model(model_path, 'resnet50')
    # 进行预测
    result, _ = engine.predict('resnet50', 'path_to_image.jpg')
    print(f"预测结果: {result['class_name']}")
    print(f"置信度: {result['confidence']:.4f}")
    print(f"各类别概率: {result['probabilities']}")