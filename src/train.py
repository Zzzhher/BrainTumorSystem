import os
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloader
from utils import get_model, save_log, load_config
import time
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser(description='训练脑肿瘤识别模型')
parser.add_argument('--model', type=str, default=None, help='要训练的模型名称 (resnet50, resnet34, efficientnet_b0, efficientnet_b1)')
args = parser.parse_args()

# 加载配置
config = load_config()

# 训练配置
train_config = {
    'dataset_path': config['paths']['dataset_path'],
    'model_save_path': config['paths']['model_save_path'],
    'log_path': config['paths']['log_path'],
    'model_name': args.model if args.model else config['model']['default_model'],
    'batch_size': config['training']['batch_size'],
    'num_epochs': config['training']['num_epochs'],
    'learning_rate': config['training']['learning_rate'],
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# 验证模型名称是否有效
supported_models = ['resnet50', 'resnet34', 'efficientnet_b0', 'efficientnet_b1']
if train_config['model_name'] not in supported_models:
    raise ValueError(f"不支持的模型名称: {train_config['model_name']}。支持的模型: {', '.join(supported_models)}")

def evaluate(model, test_loader, device):
    """评估模型在测试集上的性能"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        with tqdm(test_loader, desc='Evaluating', unit='batch') as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # 统计
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 收集预测和标签
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # 更新进度条
                pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'accuracy': 100. * correct / total})
    
    # 计算指标
    test_loss = running_loss / len(test_loader)
    test_accuracy = 100. * correct / total
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    precision_class, recall_class, f1_class, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
    
    # 生成混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'loss': test_loss,
        'accuracy': test_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_class': precision_class,
        'recall_class': recall_class,
        'f1_class': f1_class,
        'confusion_matrix': cm,
        'all_labels': all_labels,
        'all_preds': all_preds
    }

def train():
    # 创建保存目录
    os.makedirs(train_config['model_save_path'], exist_ok=True)
    os.makedirs(train_config['log_path'], exist_ok=True)
    os.makedirs(config['paths']['result_path'], exist_ok=True)
    
    # 显示设备信息
    print(f"训练设备: {train_config['device']}")
    if train_config['device'] == 'cuda':
        print(f"GPU型号: {torch.cuda.get_device_name(0)}")
    
    # 加载数据
    train_loader = get_dataloader(train_config['dataset_path'], batch_size=train_config['batch_size'], shuffle=True, balance=True)
    test_loader = get_dataloader('dataset/Testing', batch_size=train_config['batch_size'], shuffle=False, is_test=True)
    
    # 初始化模型
    model = get_model(train_config['model_name'], num_classes=4)
    model.to(train_config['device'])
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=train_config['learning_rate'])
    
    # 训练循环
    start_time = time.time()
    best_accuracy = 0.0
    patience = 10  # 早停耐心值
    patience_counter = 0  # 早停计数器
    
    # 记录训练过程中的指标
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    learning_rates = []
    epoch_times = []
    
    log_content = f"模型: {train_config['model_name']}\n"
    log_content += f"批量大小: {train_config['batch_size']}\n"
    log_content += f"学习率: {train_config['learning_rate']}\n"
    log_content += f"训练轮数: {train_config['num_epochs']}\n"
    log_content += f"设备: {train_config['device']}\n"
    log_content += "-" * 50 + "\n"
    
    for epoch in range(train_config['num_epochs']):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        # 使用 tqdm 显示进度
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{train_config["num_epochs"]}', unit='batch') as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(train_config['device']), labels.to(train_config['device'])
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 统计
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 收集预测和标签
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # 更新进度条
                pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'accuracy': 100. * correct / total})
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100. * correct / total
        
        # 计算精确率、召回率和F1分数
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
        precision_class, recall_class, f1_class, _ = precision_recall_fscore_support(all_labels, all_preds, average=None, zero_division=0)
        
        # 构建详细的日志信息
        detailed_info = f"Epoch [{epoch+1}/{train_config['num_epochs']}]:\n"
        detailed_info += f"  Loss: {epoch_loss:.4f}\n"
        detailed_info += f"  Accuracy: {epoch_accuracy:.2f}%\n"
        detailed_info += f"  Precision (macro): {precision:.4f}\n"
        detailed_info += f"  Recall (macro): {recall:.4f}\n"
        detailed_info += f"  F1 Score (macro): {f1:.4f}\n"
        
        # 每个类别的指标
        for i, class_name in enumerate(config['classes']):
            detailed_info += f"  {class_name}: Precision={precision_class[i]:.4f}, Recall={recall_class[i]:.4f}, F1={f1_class[i]:.4f}\n"
        
        # 写入日志并打印
        log_content += detailed_info
        print(detailed_info)
        
        # 测试集评估
        print("\n测试集评估:")
        test_results = evaluate(model, test_loader, train_config['device'])
        
        # 构建测试集评估日志
        test_info = f"测试集评估结果:\n"
        test_info += f"  Loss: {test_results['loss']:.4f}\n"
        test_info += f"  Accuracy: {test_results['accuracy']:.2f}%\n"
        test_info += f"  Precision (macro): {test_results['precision']:.4f}\n"
        test_info += f"  Recall (macro): {test_results['recall']:.4f}\n"
        test_info += f"  F1 Score (macro): {test_results['f1']:.4f}\n"
        
        # 每个类别的指标
        for i, class_name in enumerate(config['classes']):
            test_info += f"  {class_name}: Precision={test_results['precision_class'][i]:.4f}, Recall={test_results['recall_class'][i]:.4f}, F1={test_results['f1_class'][i]:.4f}\n"
        
        # 写入日志并打印
        log_content += test_info
        print(test_info)
        
        # 记录指标
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        val_losses.append(test_results['loss'])
        val_accuracies.append(test_results['accuracy'])
        # 记录学习率（这里假设学习率不变）
        learning_rates.append(train_config['learning_rate'])
        # 记录每个epoch的训练时间
        epoch_times.append(time.time() - epoch_start_time)
        
        # 早停机制
        if test_results['accuracy'] > best_accuracy:
            best_accuracy = test_results['accuracy']
            model_save_name = f"{train_config['model_name']}_best.pth"
            model_save_full_path = os.path.join(train_config['model_save_path'], model_save_name)
            torch.save(model.state_dict(), model_save_full_path)
            print(f"最佳模型已保存到: {model_save_full_path}")
            patience_counter = 0  # 重置计数器
        else:
            patience_counter += 1
            print(f"早停计数器: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("早停触发，停止训练")
                break
    
    # 最终测试集评估
    print("\n最终测试集评估:")
    final_test_results = evaluate(model, test_loader, train_config['device'])
    
    # 生成混淆矩阵
    plt.figure(figsize=(10, 8))
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 设置中文字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    # 确保类标签正确显示
    class_labels = [label.strip() for label in config['classes']]
    sns.heatmap(final_test_results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    
    # 保存混淆矩阵
    confusion_matrix_path = os.path.join(config['paths']['result_path'], f"confusion_matrix_{train_config['model_name']}.png")
    plt.savefig(confusion_matrix_path)
    plt.close()
    
    # 生成训练过程指标图表
    plt.figure(figsize=(16, 12))
    
    # 子图1: 训练和验证损失
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='训练损失')
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r-', label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和验证损失')
    plt.legend()
    plt.grid(True)
    
    # 子图2: 训练和验证准确率
    plt.subplot(2, 2, 2)
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, 'b-', label='训练准确率')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, 'r-', label='验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('训练和验证准确率')
    plt.legend()
    plt.grid(True)
    
    # 子图3: 学习率变化
    plt.subplot(2, 2, 3)
    plt.plot(range(1, len(learning_rates) + 1), learning_rates, 'g-')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('学习率变化')
    plt.grid(True)
    
    # 子图4: 每个Epoch的训练时间
    plt.subplot(2, 2, 4)
    plt.plot(range(1, len(epoch_times) + 1), epoch_times, 'm-')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.title('每个Epoch的训练时间')
    plt.grid(True)
    
    # 调整子图间距
    plt.tight_layout()
    
    # 保存指标图表
    metrics_chart_path = os.path.join(config['paths']['result_path'], f"training_metrics_{train_config['model_name']}.png")
    plt.savefig(metrics_chart_path)
    plt.close()
    
    # 计算训练时间
    end_time = time.time()
    training_time = end_time - start_time
    log_content += "-" * 50 + "\n"
    log_content += f"训练完成!\n"
    log_content += f"最佳准确率: {best_accuracy:.2f}%\n"
    log_content += f"最终测试集准确率: {final_test_results['accuracy']:.2f}%\n"
    log_content += f"最终测试集F1分数: {final_test_results['f1']:.4f}\n"
    log_content += f"训练时间: {training_time:.2f}秒\n"
    log_content += f"混淆矩阵已保存到: {confusion_matrix_path}\n"
    log_content += f"训练指标图表已保存到: {metrics_chart_path}\n"
    
    # 保存日志
    log_file = save_log(train_config['log_path'], log_content)
    print(f"训练日志已保存到: {log_file}")
    print(f"混淆矩阵已保存到: {confusion_matrix_path}")

if __name__ == "__main__":
    train()