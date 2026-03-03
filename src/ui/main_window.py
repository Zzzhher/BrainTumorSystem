import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QComboBox, QFileDialog, QTabWidget, 
    QGroupBox, QTextBrowser, QProgressBar, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage, QFont, QPalette, QColor, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
from PIL import Image

# 导入自定义模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.inference import InferenceEngine
from src.utils import plot_confidence_bar, generate_gradcam, overlay_heatmap, CLASSES, load_config
from src.ui.styles import Styles

# 加载配置
config = load_config()

class InferenceThread(QThread):
    """推理线程"""
    finished = pyqtSignal(dict, object)  # 结果和原始图像
    error = pyqtSignal(str)
    
    def __init__(self, engine, model_name, image_path):
        super().__init__()
        self.engine = engine
        self.model_name = model_name
        self.image_path = image_path
    
    def run(self):
        try:
            result, original_image = self.engine.predict(self.model_name, self.image_path)
            self.finished.emit(result, original_image)
        except Exception as e:
            self.error.emit(str(e))

class GradCAMThread(QThread):
    """GradCAM热力图生成线程"""
    finished = pyqtSignal(object)  # 热力图图像
    error = pyqtSignal(str)
    
    def __init__(self, engine, model_name, image_path, class_idx):
        super().__init__()
        self.engine = engine
        self.model_name = model_name
        self.image_path = image_path
        self.class_idx = class_idx
    
    def run(self):
        try:
            # 预处理图像
            model = self.engine.models[self.model_name]
            from src.utils import inference_transform
            image = Image.open(self.image_path).convert('RGB')
            image_tensor = inference_transform(image)
            image_tensor = image_tensor.unsqueeze(0)
            
            # 生成热力图
            cam = generate_gradcam(model, image_tensor, self.class_idx)
            heatmap_image = overlay_heatmap(image, cam)
            self.finished.emit(heatmap_image)
        except Exception as e:
            self.error.emit(str(e))

class BrainTumorSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("脑肿瘤识别系统")
        self.setGeometry(100, 100, 1200, 800)
        
        # 初始化样式
        self.styles = Styles()
        
        # 初始化推理引擎
        self.engine = InferenceEngine()
        self.current_image_path = None
        
        # 加载模型
        self.load_models()
        
        # 创建主布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setStyleSheet(self.styles.get_central_widget_style())
        
        # 设置字体
        self.setFont(QFont("Microsoft YaHei", 10))
        
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(20)
        
        # 创建顶部布局（图片加载和模型选择）
        self.top_layout = QHBoxLayout()
        self.top_layout.setSpacing(20)
        
        # 图片加载区
        self.image_group = QGroupBox("图片加载")
        self.image_group.setStyleSheet(self.styles.get_group_box_style())
        self.image_layout = QVBoxLayout(self.image_group)
        self.image_layout.setContentsMargins(15, 15, 15, 15)
        self.image_layout.setSpacing(15)
        
        self.image_label = QLabel("请选择MRI图像")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(300)
        self.image_label.setStyleSheet(self.styles.get_image_label_style())
        self.image_layout.addWidget(self.image_label)
        
        self.load_button = QPushButton("加载图像")
        self.load_button.clicked.connect(self.load_image)
        self.load_button.setStyleSheet(self.styles.get_load_button_style())
        self.load_button.setMinimumHeight(40)
        self.image_layout.addWidget(self.load_button)
        
        self.top_layout.addWidget(self.image_group, 1)  # 调整比例，使图片加载区域不再太宽
        
        # 模型选择和结果区
        self.control_group = QGroupBox("模型控制")
        self.control_group.setStyleSheet(self.styles.get_group_box_style())
        self.control_layout = QVBoxLayout(self.control_group)
        self.control_layout.setContentsMargins(15, 15, 15, 15)
        self.control_layout.setSpacing(15)
        
        # 模型选择
        self.model_label = QLabel("选择模型:")
        self.model_label.setStyleSheet(f"font-weight: bold; color: {self.styles.text_color.name()};")
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.engine.get_available_models())
        self.model_combo.setStyleSheet(self.styles.get_model_combo_style())
        self.model_combo.setMinimumHeight(40)
        self.control_layout.addWidget(self.model_label)
        self.control_layout.addWidget(self.model_combo)
        
        # 按钮布局 - 水平排列三个按钮
        self.buttons_layout = QHBoxLayout()
        self.buttons_layout.setSpacing(10)
        
        # 识别按钮
        self.predict_button = QPushButton("开始识别")
        self.predict_button.clicked.connect(self.start_inference)
        self.predict_button.setStyleSheet(self.styles.get_predict_button_style())
        self.predict_button.setMinimumHeight(40)
        self.buttons_layout.addWidget(self.predict_button)
        
        # 导出按钮
        self.export_button = QPushButton("导出结果")
        self.export_button.clicked.connect(self.export_result)
        self.export_button.setStyleSheet(self.styles.get_export_button_style())
        self.export_button.setMinimumHeight(40)
        self.buttons_layout.addWidget(self.export_button)
        
        # 模型对比按钮
        self.compare_button = QPushButton("模型对比")
        self.compare_button.clicked.connect(self.compare_models)
        self.compare_button.setStyleSheet(self.styles.get_compare_button_style())
        self.compare_button.setMinimumHeight(40)
        self.buttons_layout.addWidget(self.compare_button)
        
        # 将按钮布局添加到控制布局
        self.control_layout.addLayout(self.buttons_layout)
        
        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet(self.styles.get_progress_bar_style())
        self.control_layout.addWidget(self.progress_bar)
        
        # 结果显示
        self.result_group = QGroupBox("识别结果")
        self.result_group.setStyleSheet(self.styles.get_group_box_style())
        self.result_layout = QVBoxLayout(self.result_group)
        self.result_layout.setContentsMargins(10, 10, 10, 10)
        
        self.result_text = QTextBrowser()
        self.result_text.setStyleSheet(self.styles.get_result_text_style())
        self.result_layout.addWidget(self.result_text)
        
        self.control_layout.addWidget(self.result_group)
        self.top_layout.addWidget(self.control_group, 1)  # 保持与图片加载区域相同的比例
        
        self.main_layout.addLayout(self.top_layout)
        
        # 数据分析区
        self.analysis_tab = QTabWidget()
        self.analysis_tab.setStyleSheet(self.styles.get_analysis_tab_style())
        
        # 置信度条形图
        self.confidence_tab = QWidget()
        self.confidence_layout = QVBoxLayout(self.confidence_tab)
        self.confidence_layout.setContentsMargins(15, 15, 15, 15)
        self.confidence_figure = plt.figure(figsize=(10, 6))
        self.confidence_canvas = FigureCanvas(self.confidence_figure)
        self.confidence_canvas.setStyleSheet(self.styles.get_confidence_canvas_style())
        self.confidence_layout.addWidget(self.confidence_canvas)
        self.analysis_tab.addTab(self.confidence_tab, "置信度分布")
        
        # GradCAM热力图
        self.gradcam_tab = QWidget()
        self.gradcam_layout = QVBoxLayout(self.gradcam_tab)
        self.gradcam_layout.setContentsMargins(15, 15, 15, 15)
        self.gradcam_label = QLabel("GradCAM热力图")
        self.gradcam_label.setAlignment(Qt.AlignCenter)
        self.gradcam_label.setMinimumHeight(300)
        self.gradcam_label.setStyleSheet(self.styles.get_gradcam_label_style())
        self.gradcam_layout.addWidget(self.gradcam_label)
        self.analysis_tab.addTab(self.gradcam_tab, "GradCAM热力图")
        
        self.main_layout.addWidget(self.analysis_tab)
    
    def load_models(self):
        """加载模型"""
        models_dir = config['paths']['model_save_path']
        if os.path.exists(models_dir):
            for model_file in os.listdir(models_dir):
                if model_file.endswith('.pth'):
                    # 正确提取模型名称，处理efficientnet_b0这样的情况
                    if 'efficientnet' in model_file:
                        # 对于efficientnet模型，提取完整的模型名称
                        model_name = '_'.join(model_file.split('_')[:2])
                    else:
                        # 对于其他模型，使用第一个部分
                        model_name = model_file.split('_')[0]
                    model_path = os.path.join(models_dir, model_file)
                    try:
                        self.engine.load_model(model_path, model_name)
                    except Exception as e:
                        print(f"加载模型 {model_name} 失败: {e}")
        
        # 如果没有模型，添加默认选项
        if not self.engine.get_available_models():
            self.model_combo.addItem("无模型可用")
            self.predict_button.setEnabled(False)
    
    def load_image(self):
        """加载图像"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择MRI图像", "", "Image files (*.png *.jpg *.jpeg)"
        )
        if file_path:
            self.current_image_path = file_path
            # 显示图像
            pixmap = QPixmap(file_path)
            # 调整图像大小以适应标签
            scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            # 清空结果
            self.result_text.clear()
            # 清空图表
            self.clear_plots()
    
    def start_inference(self):
        """开始推理"""
        if not self.current_image_path:
            self.result_text.setText("请先加载图像")
            return
        
        model_name = self.model_combo.currentText()
        if model_name == "无模型可用":
            self.result_text.setText("请先训练模型")
            return
        
        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # 不确定进度
        
        # 启动推理线程
        self.inference_thread = InferenceThread(self.engine, model_name, self.current_image_path)
        self.inference_thread.finished.connect(self.on_inference_finished)
        self.inference_thread.error.connect(self.on_inference_error)
        self.inference_thread.start()
    
    def on_inference_finished(self, result, original_image):
        """推理完成"""
        self.progress_bar.setVisible(False)
        
        # 显示结果
        result_text = f"模型: {self.model_combo.currentText()}\n"
        result_text += f"预测类别: {result['class_name']}\n"
        result_text += f"置信度: {result['confidence']:.4f}\n"
        result_text += "\n各类别概率:\n"
        for i, prob in enumerate(result['probabilities']):
            result_text += f"{CLASSES[i].strip()}: {prob:.4f}\n"
        self.result_text.setText(result_text)
        
        # 绘制置信度条形图
        self.plot_confidence(result['probabilities'])
        
        # 生成GradCAM热力图
        self.start_gradcam(result['predicted_class'])
    
    def on_inference_error(self, error_message):
        """推理错误"""
        self.progress_bar.setVisible(False)
        self.result_text.setText(f"错误: {error_message}")
    
    def start_gradcam(self, class_idx):
        """启动GradCAM热力图生成"""
        model_name = self.model_combo.currentText()
        self.gradcam_thread = GradCAMThread(
            self.engine, model_name, self.current_image_path, class_idx
        )
        self.gradcam_thread.finished.connect(self.on_gradcam_finished)
        self.gradcam_thread.error.connect(self.on_gradcam_error)
        self.gradcam_thread.start()
    
    def on_gradcam_finished(self, heatmap_image):
        """GradCAM热力图生成完成"""
        # 转换PIL图像为QPixmap
        img_array = np.array(heatmap_image)
        height, width, channel = img_array.shape
        bytes_per_line = 3 * width
        q_image = QImage(img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        
        # 调整大小并显示
        scaled_pixmap = pixmap.scaled(self.gradcam_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.gradcam_label.setPixmap(scaled_pixmap)
    
    def on_gradcam_error(self, error_message):
        """GradCAM热力图生成错误"""
        if "mean() received an invalid combination of arguments" in error_message:
            self.gradcam_label.setText("热力图生成失败: 无法计算梯度，请检查模型结构")
        elif "无法获取梯度或激活值" in error_message:
            self.gradcam_label.setText(error_message)
        else:
            self.gradcam_label.setText(f"热力图生成失败: {error_message}")
    
    def plot_confidence(self, probabilities):
        """绘制置信度条形图"""
        # 清空画布
        self.confidence_figure.clear()
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 设置中文字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        
        # 绘制条形图
        plt.figure(self.confidence_figure.number)
        bars = plt.bar(CLASSES, probabilities, color=['blue', 'green', 'red', 'purple'])
        plt.xlabel('类别')
        plt.ylabel('置信度')
        plt.title('预测置信度分布')
        plt.ylim(0, 1)
        
        # 在条形上方添加数值
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', ha='center', va='bottom')
        
        # 更新画布
        self.confidence_canvas.draw()
    
    def clear_plots(self):
        """清空图表"""
        # 清空置信度条形图
        self.confidence_figure.clear()
        self.confidence_canvas.draw()
        
        # 清空GradCAM热力图
        self.gradcam_label.setText("GradCAM热力图")
    
    def export_result(self):
        """导出结果"""
        if not self.current_image_path:
            self.result_text.setText("请先加载图像并进行识别")
            return
        
        # 选择保存路径
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出结果", "result.txt", "Text files (*.txt);;CSV files (*.csv)"
        )
        
        if file_path:
            # 获取当前结果
            result_text = self.result_text.toPlainText()
            
            # 保存结果
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(result_text)
            
            self.result_text.append(f"\n结果已导出到: {file_path}")
    
    def compare_models(self):
        """模型对比"""
        if not self.current_image_path:
            self.result_text.setText("请先加载图像")
            return
        
        # 获取所有可用模型
        models = self.engine.get_available_models()
        if len(models) < 2:
            self.result_text.setText("至少需要两个模型才能进行对比")
            return
        
        # 显示进度条
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(models))
        
        # 对每个模型进行预测
        results = []
        for i, model_name in enumerate(models):
            try:
                result, _ = self.engine.predict(model_name, self.current_image_path)
                results.append({
                    'model_name': model_name,
                    'class_name': result['class_name'],
                    'confidence': result['confidence'],
                    'probabilities': result['probabilities']
                })
            except Exception as e:
                results.append({
                    'model_name': model_name,
                    'error': str(e)
                })
            
            # 更新进度条
            self.progress_bar.setValue(i + 1)
        
        # 隐藏进度条
        self.progress_bar.setVisible(False)
        
        # 生成对比结果
        comparison_text = "模型对比结果:\n"
        comparison_text += "-" * 80 + "\n"
        
        for result in results:
            if 'error' in result:
                comparison_text += f"{result['model_name']}: 错误 - {result['error']}\n"
            else:
                comparison_text += f"{result['model_name']}:\n"
                comparison_text += f"  预测类别: {result['class_name']}\n"
                comparison_text += f"  置信度: {result['confidence']:.4f}\n"
                comparison_text += "  各类别概率:\n"
                for j, prob in enumerate(result['probabilities']):
                    comparison_text += f"    {CLASSES[j].strip()}: {prob:.4f}\n"
            comparison_text += "-" * 80 + "\n"
        
        # 显示对比结果
        self.result_text.setText(comparison_text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BrainTumorSystem()
    window.show()
    sys.exit(app.exec_())
