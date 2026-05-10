"""
Control panel for dataset loading and pipeline parameters
"""
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, 
                               QGroupBox, QSpinBox, QDoubleSpinBox, QComboBox,
                               QFileDialog, QTextEdit, QFormLayout, QHBoxLayout,
                               QProgressBar)
from PySide6.QtCore import Signal, Qt


class ControlsPanel(QWidget):
    """Control panel for the application"""
    
    # Signals
    load_mat_clicked = Signal(str)  # file_path
    load_separate_clicked = Signal(str, str)  # data_path, gt_path
    run_pipeline_clicked = Signal(dict)  # parameters
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the UI"""
        layout = QVBoxLayout(self)
        
        # Dataset loading group
        dataset_group = QGroupBox("Dataset Loading")
        dataset_layout = QVBoxLayout()
        
        # Single .mat file option
        self.load_mat_btn = QPushButton("Load .mat Dataset")
        self.load_mat_btn.clicked.connect(self._on_load_mat)
        dataset_layout.addWidget(self.load_mat_btn)
        
        # Separate files option
        sep_layout = QHBoxLayout()
        self.load_data_btn = QPushButton("Load Data")
        self.load_gt_btn = QPushButton("Load GT")
        self.load_data_btn.clicked.connect(self._on_load_data)
        self.load_gt_btn.clicked.connect(self._on_load_gt)
        sep_layout.addWidget(self.load_data_btn)
        sep_layout.addWidget(self.load_gt_btn)
        dataset_layout.addLayout(sep_layout)
        
        # Dataset info
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(100)
        self.info_text.setPlaceholderText("Dataset information will appear here...")
        dataset_layout.addWidget(self.info_text)
        
        dataset_group.setLayout(dataset_layout)
        layout.addWidget(dataset_group)
        
        # Pipeline parameters group
        params_group = QGroupBox("Pipeline Parameters")
        params_layout = QFormLayout()
        
        # LDA components
        self.n_lda_spin = QSpinBox()
        self.n_lda_spin.setRange(1, 20)
        self.n_lda_spin.setValue(8)
        params_layout.addRow("LDA Components:", self.n_lda_spin)
        
        # K neighbors
        self.k_spin = QSpinBox()
        self.k_spin.setRange(1, 20)
        self.k_spin.setValue(5)
        params_layout.addRow("K Neighbors:", self.k_spin)
        
        # Distance metric
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["euclidean", "manhattan", "chebyshev"])
        params_layout.addRow("Distance Metric:", self.metric_combo)
        
        # Test ratio
        self.test_ratio_spin = QDoubleSpinBox()
        self.test_ratio_spin.setRange(0.1, 0.9)
        self.test_ratio_spin.setSingleStep(0.05)
        self.test_ratio_spin.setValue(0.3)
        self.test_ratio_spin.setDecimals(2)
        params_layout.addRow("Test Ratio:", self.test_ratio_spin)
        
        # Random seed
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 9999)
        self.seed_spin.setValue(42)
        params_layout.addRow("Random Seed:", self.seed_spin)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Run button
        self.run_btn = QPushButton("Run Pipeline")
        self.run_btn.setEnabled(False)
        self.run_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)
        self.run_btn.clicked.connect(self._on_run_pipeline)
        layout.addWidget(self.run_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.status_label)
        
        layout.addStretch()
        
        # Internal state
        self.data_path = None
        self.gt_path = None
        self.mat_path = None
    
    def _on_load_mat(self):
        """Handle .mat file loading"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select .mat Dataset",
            "",
            "MATLAB Files (*.mat)"
        )
        
        if file_path:
            self.mat_path = file_path
            self.data_path = None
            self.gt_path = None
            self.load_mat_clicked.emit(file_path)
    
    def _on_load_data(self):
        """Handle data file loading"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Data File",
            "",
            "MATLAB Files (*.mat)"
        )
        
        if file_path:
            self.data_path = file_path
            self._check_separate_files()
    
    def _on_load_gt(self):
        """Handle ground truth file loading"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Ground Truth File",
            "",
            "MATLAB Files (*.mat)"
        )
        
        if file_path:
            self.gt_path = file_path
            self._check_separate_files()
    
    def _check_separate_files(self):
        """Check if both files are loaded and emit signal"""
        if self.data_path and self.gt_path:
            self.mat_path = None
            self.load_separate_clicked.emit(self.data_path, self.gt_path)
    
    def _on_run_pipeline(self):
        """Handle run pipeline button"""
        params = {
            'n_lda': self.n_lda_spin.value(),
            'k_neighbors': self.k_spin.value(),
            'metric': self.metric_combo.currentText(),
            'test_size': self.test_ratio_spin.value(),
            'random_state': self.seed_spin.value()
        }
        
        self.run_pipeline_clicked.emit(params)
    
    def set_dataset_info(self, info):
        """
        Display dataset information
        
        Args:
            info: dict with dataset info
        """
        text = f"""
Dataset Loaded Successfully!

Shape: {info['shape']}
Classes: {info['classes']}
Labeled Pixels: {info['labeled_pixels']}
Total Pixels: {info['total_pixels']}
"""
        self.info_text.setPlainText(text.strip())
        self.run_btn.setEnabled(True)
        self.status_label.setText("✓ Dataset ready")
        self.status_label.setStyleSheet("color: green; font-weight: bold;")
    
    def set_progress(self, value, maximum, message=""):
        """Update progress bar"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)
        
        if message:
            self.status_label.setText(message)
    
    def set_status(self, message, is_error=False):
        """Set status message"""
        self.status_label.setText(message)
        if is_error:
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
    
    def enable_controls(self, enabled):
        """Enable/disable controls during processing"""
        self.load_mat_btn.setEnabled(enabled)
        self.load_data_btn.setEnabled(enabled)
        self.load_gt_btn.setEnabled(enabled)
        self.run_btn.setEnabled(enabled)
        self.n_lda_spin.setEnabled(enabled)
        self.k_spin.setEnabled(enabled)
        self.metric_combo.setEnabled(enabled)
        self.test_ratio_spin.setEnabled(enabled)
        self.seed_spin.setEnabled(enabled)
