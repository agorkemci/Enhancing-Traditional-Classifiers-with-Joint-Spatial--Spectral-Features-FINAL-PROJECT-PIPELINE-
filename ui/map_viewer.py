"""
Interactive classification map viewer with hover tooltips
"""
import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox
from PySide6.QtCore import Qt, Signal
import pyqtgraph as pg


class MapViewer(QWidget):
    """Interactive classification map viewer"""
    
    # Signal emitted when hovering over a pixel
    pixel_hovered = Signal(int, int, dict)  # x, y, info
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.classification_map = None
        self.ground_truth = None
        self.confidence_map = None
        self.error_map = None
        
        self.current_mode = "Prediction"
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the UI"""
        layout = QVBoxLayout(self)
        
        # Top controls
        controls_layout = QHBoxLayout()
        
        controls_layout.addWidget(QLabel("Display Mode:"))
        
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Prediction",
            "Ground Truth",
            "Error Map",
            "Confidence Map"
        ])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        controls_layout.addWidget(self.mode_combo)
        
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # Image view
        self.graphics_view = pg.GraphicsLayoutWidget()
        self.plot_item = self.graphics_view.addPlot()
        self.plot_item.hideAxis('left')
        self.plot_item.hideAxis('bottom')
        
        self.image_item = pg.ImageItem()
        self.plot_item.addItem(self.image_item)
        
        # Setup mouse tracking
        self.image_item.setOpts(axisOrder='row-major')
        self.graphics_view.scene().sigMouseMoved.connect(self._on_mouse_moved)
        
        layout.addWidget(self.graphics_view)
        
        # Info label for hover
        self.info_label = QLabel("Hover over the map to see pixel information")
        self.info_label.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                color: #ffffff;
                padding: 8px;
                border-radius: 4px;
                font-family: 'Consolas', monospace;
            }
        """)
        layout.addWidget(self.info_label)
    
    def set_data(self, classification_map, ground_truth, confidence_map, error_map):
        """
        Set the maps to display
        
        Args:
            classification_map: Predicted labels (M, N)
            ground_truth: True labels (M, N)
            confidence_map: Confidence scores (M, N)
            error_map: Error map (M, N) - 0=unlabeled, 1=correct, 2=incorrect
        """
        self.classification_map = classification_map
        self.ground_truth = ground_truth
        self.confidence_map = confidence_map
        self.error_map = error_map
        
        self._update_display()
    
    def _on_mode_changed(self, mode):
        """Handle mode change"""
        self.current_mode = mode
        self._update_display()
    
    def _update_display(self):
        """Update the displayed image"""
        if self.classification_map is None:
            return
        
        if self.current_mode == "Prediction":
            display_map = self.classification_map
            colormap = pg.colormap.get('tab20', source='matplotlib')
            
        elif self.current_mode == "Ground Truth":
            display_map = self.ground_truth
            colormap = pg.colormap.get('tab20', source='matplotlib')
            
        elif self.current_mode == "Error Map":
            display_map = self.error_map
            # Custom colormap: black (0), green (1), red (2)
            colors = np.array([
                [0, 0, 0, 255],       # Black - unlabeled
                [0, 255, 0, 255],     # Green - correct
                [255, 0, 0, 255]      # Red - incorrect
            ], dtype=np.ubyte)
            colormap = pg.ColorMap(pos=np.linspace(0, 1, 3), color=colors)
            
        elif self.current_mode == "Confidence Map":
            display_map = self.confidence_map
            colormap = pg.colormap.get('viridis', source='matplotlib')
        
        # Apply colormap
        if self.current_mode == "Error Map":
            # Direct mapping for error map
            lut = colormap.getLookupTable(0, 2, 3)
            self.image_item.setLookupTable(lut)
            self.image_item.setLevels([0, 2])
        else:
            lut = colormap.getLookupTable(0.0, 1.0, 256)
            self.image_item.setLookupTable(lut)
            vmin, vmax = display_map.min(), display_map.max()
            self.image_item.setLevels([vmin, vmax])
        
        self.image_item.setImage(display_map)
    
    def _on_mouse_moved(self, pos):
        """Handle mouse movement over the image"""
        if self.classification_map is None:
            return
        
        # Get mouse position in scene coordinates
        mouse_point = self.plot_item.vb.mapSceneToView(pos)
        
        x = int(mouse_point.x())
        y = int(mouse_point.y())
        
        # Check bounds
        M, N = self.classification_map.shape
        if 0 <= y < M and 0 <= x < N:
            # Get pixel information
            pred_class = int(self.classification_map[y, x])
            true_class = int(self.ground_truth[y, x])
            confidence = float(self.confidence_map[y, x])
            
            if true_class == 0:
                # Unlabeled pixel
                info_text = f"Position: ({x}, {y}) - Unlabeled"
            else:
                is_correct = (pred_class == true_class)
                status = "✓ Correct" if is_correct else "✗ Wrong"
                
                info_text = (
                    f"Position: ({x}, {y}) | "
                    f"GT: Class {true_class} | "
                    f"Prediction: Class {pred_class} | "
                    f"{status} | "
                    f"Confidence: {confidence:.3f}"
                )
            
            self.info_label.setText(info_text)
            
            # Emit signal
            info = {
                'x': x,
                'y': y,
                'true_class': true_class,
                'pred_class': pred_class,
                'confidence': confidence,
                'is_correct': is_correct if true_class > 0 else None
            }
            self.pixel_hovered.emit(x, y, info)
