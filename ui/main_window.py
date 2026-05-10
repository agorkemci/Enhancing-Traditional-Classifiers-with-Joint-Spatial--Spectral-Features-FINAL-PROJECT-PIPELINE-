"""
Main application window
"""
from PySide6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
                               QSplitter, QMessageBox, QTabWidget)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction

from .controls_panel import ControlsPanel
from .map_viewer import MapViewer
from .metrics_panel import MetricsPanel

from pipeline import DatasetLoader, ClassificationPipeline


class PipelineWorker(QThread):
    """Worker thread for running the pipeline"""
    
    progress = Signal(int, str)  # step, message
    finished = Signal(dict)  # results
    error = Signal(str)  # error message
    
    def __init__(self, pipeline, data, gt):
        super().__init__()
        self.pipeline = pipeline
        self.data = data
        self.gt = gt
    
    def run(self):
        """Run the pipeline in a separate thread"""
        try:
            results = self.pipeline.run(
                self.data,
                self.gt,
                progress_callback=self.progress.emit
            )
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Hyperspectral Classification - LDA + EMP + CS-KNN")
        self.setGeometry(100, 100, 1400, 800)
        
        # Data
        self.loader = DatasetLoader()
        self.pipeline = None
        self.worker = None
        
        self._setup_ui()
        self._setup_menu()
    
    def _setup_ui(self):
        """Setup the main UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel (controls)
        self.controls = ControlsPanel()
        self.controls.setMaximumWidth(350)
        self.controls.load_mat_clicked.connect(self._on_load_mat)
        self.controls.load_separate_clicked.connect(self._on_load_separate)
        self.controls.run_pipeline_clicked.connect(self._on_run_pipeline)
        
        # Right panel (visualization and metrics)
        right_splitter = QSplitter(Qt.Vertical)
        
        # Map viewer (top)
        self.map_viewer = MapViewer()
        right_splitter.addWidget(self.map_viewer)
        
        # Metrics panel (bottom)
        self.metrics_panel = MetricsPanel()
        right_splitter.addWidget(self.metrics_panel)
        
        # Set initial sizes (70% map, 30% metrics)
        right_splitter.setSizes([700, 300])
        
        # Add to main layout
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(self.controls)
        main_splitter.addWidget(right_splitter)
        main_splitter.setSizes([350, 1050])
        
        main_layout.addWidget(main_splitter)
    
    def _setup_menu(self):
        """Setup menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        load_action = QAction("Load Dataset...", self)
        load_action.triggered.connect(self.controls._on_load_mat)
        file_menu.addAction(load_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _on_load_mat(self, file_path):
        """Handle .mat file loading"""
        try:
            data_key, gt_key = self.loader.load_mat(file_path)
            info = self.loader.get_info()
            
            self.controls.set_dataset_info(info)
            self.controls.set_status(f"✓ Loaded keys: {data_key}, {gt_key}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load dataset:\n{str(e)}")
            self.controls.set_status(f"✗ Error: {str(e)}", is_error=True)
    
    def _on_load_separate(self, data_path, gt_path):
        """Handle separate file loading"""
        try:
            data_key, gt_key = self.loader.load_separate(data_path, gt_path)
            info = self.loader.get_info()
            
            self.controls.set_dataset_info(info)
            self.controls.set_status(f"✓ Loaded keys: {data_key}, {gt_key}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load dataset:\n{str(e)}")
            self.controls.set_status(f"✗ Error: {str(e)}", is_error=True)
    
    def _on_run_pipeline(self, params):
        """Handle pipeline execution"""
        if self.loader.data is None:
            QMessageBox.warning(self, "Warning", "Please load a dataset first!")
            return
        
        # Disable controls
        self.controls.enable_controls(False)
        self.controls.set_status("Running pipeline...")
        
        # Create pipeline
        self.pipeline = ClassificationPipeline(params)
        
        # Create and start worker thread
        self.worker = PipelineWorker(
            self.pipeline,
            self.loader.data,
            self.loader.gt
        )
        
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_pipeline_finished)
        self.worker.error.connect(self._on_pipeline_error)
        
        self.worker.start()
    
    def _on_progress(self, step, message):
        """Handle progress updates"""
        self.controls.set_progress(step, 7, message)
    
    def _on_pipeline_finished(self, results):
        """Handle pipeline completion"""
        # Update map viewer
        self.map_viewer.set_data(
            results['classification_map'],
            results['ground_truth'],
            results['confidence_map'],
            results['error_map']
        )
        
        # Update metrics panel
        self.metrics_panel.set_metrics(results['metrics'])
        
        # Re-enable controls
        self.controls.enable_controls(True)
        self.controls.progress_bar.setVisible(False)
        self.controls.set_status("✓ Pipeline completed successfully!")
        
        # Show summary
        metrics = results['metrics']
        QMessageBox.information(
            self,
            "Pipeline Complete",
            f"Classification completed!\n\n"
            f"Overall Accuracy: {metrics['oa']*100:.2f}%\n"
            f"Average Accuracy: {metrics['aa']*100:.2f}%\n"
            f"Kappa: {metrics['kappa']:.4f}"
        )
    
    def _on_pipeline_error(self, error_msg):
        """Handle pipeline error"""
        self.controls.enable_controls(True)
        self.controls.progress_bar.setVisible(False)
        self.controls.set_status(f"✗ Error: {error_msg}", is_error=True)
        
        QMessageBox.critical(
            self,
            "Pipeline Error",
            f"An error occurred during pipeline execution:\n\n{error_msg}"
        )
    
    def _show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About",
            "Hyperspectral Classification Application\n\n"
            "Pipeline: LDA + EMP + CS-KNN\n\n"
            "Features:\n"
            "• Interactive classification map viewer\n"
            "• Hover to see pixel information\n"
            "• Multiple visualization modes\n"
            "• Comprehensive metrics\n"
            "• Confusion matrix visualization\n\n"
            "Developed with PySide6 + PyQtGraph"
        )
