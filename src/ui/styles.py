from PyQt5.QtGui import QColor

class Styles:
    def __init__(self):
        # 主题色
        self.primary_color = QColor(44, 62, 80)
        self.secondary_color = QColor(52, 152, 219)
        self.accent_color = QColor(231, 76, 60)
        self.background_color = QColor(245, 246, 250)
        self.text_color = QColor(44, 62, 80)
        
    def get_central_widget_style(self):
        return f"background-color: {self.background_color.name()};"
    
    def get_group_box_style(self):
        return f"QGroupBox {{ border: 2px solid {self.secondary_color.name()}; border-radius: 8px; padding: 10px; font-weight: bold; color: {self.text_color.name()}; }}"
    
    def get_image_label_style(self):
        return f"border: 2px dashed {self.secondary_color.name()}; border-radius: 8px; padding: 10px; background-color: white;"
    
    def get_load_button_style(self):
        return f"QPushButton {{ background-color: {self.secondary_color.name()}; color: white; border-radius: 6px; padding: 10px; font-weight: bold; }} QPushButton:hover {{ background-color: {self.primary_color.name()}; }}"
    
    def get_model_combo_style(self):
        return f"QComboBox {{ border: 1px solid {self.secondary_color.name()}; border-radius: 6px; padding: 8px; min-width: 200px; }} QComboBox::drop-down {{ border-left: 1px solid {self.secondary_color.name()}; }} QComboBox QAbstractItemView {{ min-width: 200px; min-height: 30px; font-size: 12px; }}"
    
    def get_predict_button_style(self):
        return f"QPushButton {{ background-color: {self.secondary_color.name()}; color: white; border-radius: 6px; padding: 10px; font-weight: bold; }} QPushButton:hover {{ background-color: {self.primary_color.name()}; }}"
    
    def get_export_button_style(self):
        return f"QPushButton {{ background-color: #27ae60; color: white; border-radius: 6px; padding: 10px; font-weight: bold; }} QPushButton:hover {{ background-color: #229954; }}"
    
    def get_compare_button_style(self):
        return f"QPushButton {{ background-color: #f39c12; color: white; border-radius: 6px; padding: 10px; font-weight: bold; }} QPushButton:hover {{ background-color: #e67e22; }}"
    
    def get_progress_bar_style(self):
        return f"QProgressBar {{ border: 1px solid {self.secondary_color.name()}; border-radius: 6px; background-color: white; }} QProgressBar::chunk {{ background-color: {self.secondary_color.name()}; }}"
    
    def get_result_text_style(self):
        return "QTextBrowser { border: 1px solid #e0e0e0; border-radius: 6px; padding: 10px; background-color: white; }"
    
    def get_analysis_tab_style(self):
        return f"QTabWidget::pane {{ border: 2px solid {self.secondary_color.name()}; border-radius: 8px; background-color: white; }} QTabBar::tab {{ background-color: {self.background_color.name()}; color: {self.text_color.name()}; padding: 10px; border-top-left-radius: 6px; border-top-right-radius: 6px; }} QTabBar::tab:selected {{ background-color: white; border: 2px solid {self.secondary_color.name()}; border-bottom: none; }}"
    
    def get_confidence_canvas_style(self):
        return "border: 1px solid #e0e0e0; border-radius: 6px;"
    
    def get_gradcam_label_style(self):
        return "border: 1px solid #e0e0e0; border-radius: 6px; background-color: white;"
