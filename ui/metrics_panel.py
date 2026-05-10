"""
Metrics display panel with confusion matrix
"""
import numpy as np
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QTableWidget, 
                               QTableWidgetItem, QHeaderView, QGroupBox)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
import pyqtgraph as pg


class MetricsPanel(QWidget):
    """Display classification metrics"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the UI"""
        layout = QVBoxLayout(self)
        
        # Overall metrics group
        overall_group = QGroupBox("Overall Metrics")
        overall_layout = QVBoxLayout()
        
        self.oa_label = QLabel("OA: --")
        self.aa_label = QLabel("AA: --")
        self.kappa_label = QLabel("Kappa: --")
        
        for label in [self.oa_label, self.aa_label, self.kappa_label]:
            label.setStyleSheet("""
                QLabel {
                    font-size: 14px;
                    font-weight: bold;
                    padding: 4px;
                }
            """)
            overall_layout.addWidget(label)
        
        overall_group.setLayout(overall_layout)
        layout.addWidget(overall_group)
        
        # Per-class accuracy table
        class_group = QGroupBox("Per-Class Accuracy")
        class_layout = QVBoxLayout()
        
        self.class_table = QTableWidget()
        self.class_table.setColumnCount(3)
        self.class_table.setHorizontalHeaderLabels(["Class", "Accuracy (%)", "Samples"])
        self.class_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.class_table.setAlternatingRowColors(True)
        
        class_layout.addWidget(self.class_table)
        class_group.setLayout(class_layout)
        layout.addWidget(class_group)
        
        # Confusion matrix
        cm_group = QGroupBox("Confusion Matrix")
        cm_layout = QVBoxLayout()
        
        self.cm_view = pg.GraphicsLayoutWidget()
        self.cm_plot = self.cm_view.addPlot()
        self.cm_plot.hideAxis('left')
        self.cm_plot.hideAxis('bottom')
        
        self.cm_image = pg.ImageItem()
        self.cm_plot.addItem(self.cm_image)
        
        cm_layout.addWidget(self.cm_view)
        cm_group.setLayout(cm_layout)
        layout.addWidget(cm_group)
        
        layout.addStretch()
    
    def set_metrics(self, metrics):
        """
        Display metrics
        
        Args:
            metrics: dict from MetricsCalculator.compute_all
        """
        # Overall metrics
        self.oa_label.setText(f"Overall Accuracy (OA): {metrics['oa']*100:.2f}%")
        self.aa_label.setText(f"Average Accuracy (AA): {metrics['aa']*100:.2f}%")
        self.kappa_label.setText(f"Kappa Coefficient: {metrics['kappa']:.4f}")
        
        # Per-class table
        per_class = metrics['per_class']
        self.class_table.setRowCount(len(per_class))
        
        for i, (cls, stats) in enumerate(sorted(per_class.items())):
            # Class number
            class_item = QTableWidgetItem(str(cls))
            class_item.setTextAlignment(Qt.AlignCenter)
            self.class_table.setItem(i, 0, class_item)
            
            # Accuracy
            acc = stats['accuracy'] * 100
            acc_item = QTableWidgetItem(f"{acc:.2f}")
            acc_item.setTextAlignment(Qt.AlignCenter)
            
            # Color code by accuracy
            if acc >= 90:
                acc_item.setBackground(QColor(0, 255, 0, 50))
            elif acc >= 70:
                acc_item.setBackground(QColor(255, 255, 0, 50))
            else:
                acc_item.setBackground(QColor(255, 0, 0, 50))
            
            self.class_table.setItem(i, 1, acc_item)
            
            # Samples
            samples_item = QTableWidgetItem(str(stats['samples']))
            samples_item.setTextAlignment(Qt.AlignCenter)
            self.class_table.setItem(i, 2, samples_item)
        
        # Confusion matrix
        cm = metrics['confusion_matrix']
        
        # Normalize for better visualization
        cm_normalized = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
        
        # Display
        colormap = pg.colormap.get('viridis', source='matplotlib')
        lut = colormap.getLookupTable(0.0, 1.0, 256)
        self.cm_image.setLookupTable(lut)
        self.cm_image.setImage(cm_normalized)
        self.cm_image.setLevels([0, 1])
