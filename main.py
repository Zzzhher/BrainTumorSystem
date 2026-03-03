#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
脑肿瘤识别系统入口点
"""

import sys
import os

# 添加 src 目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.ui.main_window import BrainTumorSystem
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BrainTumorSystem()
    window.show()
    sys.exit(app.exec_())
