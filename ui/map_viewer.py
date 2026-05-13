"""
Interactive classification map viewer
  - Harita tam alanı kullanır (stretch + setAspectLocked kaldırıldı)
  - Mod geçişinde smooth fade animasyonu (QGraphicsOpacityEffect + QPropertyAnimation)
  - Şık toolbar: ikon butonlar + mod seçici + zoom reset + export
  - Gelişmiş hover bilgi şeridi (renkli class badge)
"""

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QSizePolicy, QGraphicsOpacityEffect, QFrame,
    QFileDialog
)
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve, QTimer, QSize
from PySide6.QtGui import QColor
import pyqtgraph as pg

# ── 20-colour palette for class labels ───────────────────────────────────────
_TAB20 = [
    (31,  119, 180), (174, 199, 232), (255, 127,  14), (255, 187, 120),
    ( 44, 160,  44), (152, 223, 138), (214,  39,  40), (255, 152, 150),
    (148, 103, 189), (197, 176, 213), (140,  86,  75), (196, 156, 148),
    (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
    (188, 189,  34), (219, 219, 141), ( 23, 190, 207), (158, 218, 229),
]

def _class_color(cls_id: int) -> tuple:
    """Return (r,g,b) for a class label."""
    if cls_id == 0:
        return (20, 20, 20)
    return _TAB20[(cls_id - 1) % len(_TAB20)]


def _build_tab20_lut(n: int = 256) -> np.ndarray:
    lut = np.zeros((n, 4), dtype=np.uint8)
    for i in range(n):
        r, g, b = _TAB20[i % len(_TAB20)]
        lut[i] = [r, g, b, 255]
    lut[0] = [15, 15, 15, 255]   # class 0 → near-black
    return lut


# ── Toolbar button helper ─────────────────────────────────────────────────────

def _tool_btn(text: str, tooltip: str, checkable: bool = False) -> QPushButton:
    btn = QPushButton(text)
    btn.setToolTip(tooltip)
    btn.setCheckable(checkable)
    btn.setFixedHeight(28)
    btn.setCursor(Qt.PointingHandCursor)
    btn.setStyleSheet("""
        QPushButton {
            background: #3a3a3a;
            color: #ddd;
            border: 1px solid #555;
            border-radius: 5px;
            padding: 0 10px;
            font-size: 12px;
        }
        QPushButton:hover  { background: #4a4a4a; color: #fff; border-color: #777; }
        QPushButton:pressed { background: #555; }
        QPushButton:checked { background: #2a7fd4; color: #fff; border-color: #3a9ff5; }
    """)
    return btn


# ── Animated image swap helper ────────────────────────────────────────────────

class _FadeImageItem(pg.ImageItem):
    """ImageItem that fades in when setImage is called."""

    def __init__(self, gfx_widget):
        super().__init__()
        self._effect = QGraphicsOpacityEffect()
        self._effect.setOpacity(1.0)
        # We animate via QTimer + step (QPropertyAnimation needs QObject proxy)
        self._opacity = 1.0
        self._anim_steps = 0
        self._timer = QTimer()
        self._timer.setInterval(16)   # ~60fps
        self._timer.timeout.connect(self._step)

    def animated_set_image(self, data):
        """Fade out → swap → fade in."""
        self._pending = data
        self._fading_out = True
        self._opacity = 1.0
        self._anim_steps = 0
        self._timer.start()

    def _step(self):
        if self._fading_out:
            self._opacity = max(0.0, self._opacity - 0.12)
            self.setOpacity(self._opacity)
            if self._opacity <= 0.0:
                self.setImage(self._pending)
                self._fading_out = False
        else:
            self._opacity = min(1.0, self._opacity + 0.10)
            self.setOpacity(self._opacity)
            if self._opacity >= 1.0:
                self._timer.stop()


# ── Hover info bar ────────────────────────────────────────────────────────────

class _InfoBar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(36)
        self.setStyleSheet("""
            QFrame {
                background: #1e1e1e;
                border-top: 1px solid #333;
            }
        """)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 0, 12, 0)
        lay.setSpacing(16)

        self._pos   = self._seg("—")
        self._cls_badge = QLabel()
        self._cls_badge.setFixedSize(22, 22)
        self._cls_badge.setAlignment(Qt.AlignCenter)
        self._cls_badge.setStyleSheet(
            "border-radius:11px; font-size:10px; font-weight:bold; color:#fff;"
            "background:#555;"
        )
        self._gt    = self._seg("—")
        self._pred  = self._seg("—")
        self._status = QLabel("—")
        self._status.setFixedWidth(80)
        self._status.setAlignment(Qt.AlignCenter)
        self._status.setStyleSheet(
            "font-size:11px; font-weight:bold; border-radius:3px; padding:1px 6px;"
        )
        self._conf  = self._seg("—")

        for lbl, val in [("Position", self._pos), ("GT", self._gt),
                         ("Predicted", self._pred), ("Confidence", self._conf)]:
            lay.addWidget(QLabel(f"<span style='color:#666;font-size:10px'>{lbl}</span>"))
            if val is self._pos:
                lay.addWidget(self._pos)
                lay.addWidget(self._cls_badge)
            else:
                lay.addWidget(val)
            if val is not self._conf:
                sep = QFrame(); sep.setFrameShape(QFrame.VLine)
                sep.setStyleSheet("color:#333;"); lay.addWidget(sep)

        lay.addWidget(self._status)
        lay.addStretch()

    @staticmethod
    def _seg(txt):
        lbl = QLabel(txt)
        lbl.setStyleSheet("color:#ccc; font-size:11px; font-family:monospace;")
        return lbl

    def update_pixel(self, x, y, true_cls, pred_cls, confidence, is_correct):
        self._pos.setText(f"({x}, {y})")
        r, g, b = _class_color(pred_cls)
        self._cls_badge.setStyleSheet(
            f"border-radius:11px;font-size:9px;font-weight:bold;color:#fff;"
            f"background:rgb({r},{g},{b});"
        )
        self._cls_badge.setText(str(pred_cls))
        self._gt.setText(f"Class {true_cls}")
        self._pred.setText(f"Class {pred_cls}")
        self._conf.setText(f"{confidence:.3f}")
        if is_correct is None:
            self._status.setText("Unlabeled")
            self._status.setStyleSheet(
                "font-size:11px;font-weight:bold;color:#888;"
                "background:#2a2a2a;border-radius:3px;padding:1px 6px;"
            )
        elif is_correct:
            self._status.setText("✓  Correct")
            self._status.setStyleSheet(
                "font-size:11px;font-weight:bold;color:#2ecc71;"
                "background:#1a3a28;border-radius:3px;padding:1px 6px;"
            )
        else:
            self._status.setText("✗  Wrong")
            self._status.setStyleSheet(
                "font-size:11px;font-weight:bold;color:#e74c3c;"
                "background:#3a1a1a;border-radius:3px;padding:1px 6px;"
            )

    def set_idle(self):
        self._pos.setText("—")
        self._cls_badge.setText("")
        self._cls_badge.setStyleSheet(
            "border-radius:11px;font-size:9px;background:#333;"
        )
        self._gt.setText("—")
        self._pred.setText("—")
        self._conf.setText("—")
        self._status.setText("Hover map")
        self._status.setStyleSheet(
            "font-size:11px;color:#555;background:transparent;"
        )


# ── Main MapViewer ────────────────────────────────────────────────────────────

class MapViewer(QWidget):

    pixel_hovered = Signal(int, int, dict)

    # Mode definitions: (label, needs_tab20)
    _MODES = [
        ("Prediction",   True),
        ("Ground Truth", True),
        ("Error Map",    False),
        ("Confidence",   False),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.classification_map = None
        self.ground_truth       = None
        self.confidence_map     = None
        self.error_map          = None
        self.current_mode       = "Prediction"

        self._lut_tab20 = _build_tab20_lut()
        self._setup_ui()

    # ── Build UI ─────────────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Toolbar ──────────────────────────────────────────────────
        toolbar = QWidget()
        toolbar.setFixedHeight(44)
        toolbar.setStyleSheet("background:#252525; border-bottom:1px solid #333;")
        tl = QHBoxLayout(toolbar)
        tl.setContentsMargins(10, 0, 10, 0)
        tl.setSpacing(6)

        # Mode buttons
        self._mode_btns = {}
        for label, _ in self._MODES:
            btn = _tool_btn(label, f"Show {label}", checkable=True)
            btn.clicked.connect(lambda checked, m=label: self._set_mode(m))
            self._mode_btns[label] = btn
            tl.addWidget(btn)

        self._mode_btns["Prediction"].setChecked(True)

        tl.addStretch()

        # Reset zoom
        self._zoom_btn = _tool_btn("⊡  Fit", "Reset zoom to fit")
        self._zoom_btn.clicked.connect(self._reset_zoom)
        tl.addWidget(self._zoom_btn)

        # Export
        self._export_btn = _tool_btn("↓  Export", "Export current view as PNG")
        self._export_btn.clicked.connect(self._export_image)
        tl.addWidget(self._export_btn)

        root.addWidget(toolbar)

        # ── pyqtgraph view ───────────────────────────────────────────
        # Dark background
        pg.setConfigOption('background', '#111111')
        pg.setConfigOption('foreground', '#888888')

        self.gfx = pg.GraphicsLayoutWidget()
        self.gfx.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.plot = self.gfx.addPlot()
        self.plot.hideAxis('left')
        self.plot.hideAxis('bottom')
        self.plot.setAspectLocked(False)          # ← allow free stretch
        self.plot.vb.setDefaultPadding(0.01)

        self.img_item = _FadeImageItem(self.gfx)
        self.img_item.setOpts(axisOrder='row-major')
        self.plot.addItem(self.img_item)

        self.gfx.scene().sigMouseMoved.connect(self._on_mouse_moved)

        root.addWidget(self.gfx, stretch=1)

        # ── Info bar ─────────────────────────────────────────────────
        self._info = _InfoBar()
        self._info.set_idle()
        root.addWidget(self._info)

    # ── Data ─────────────────────────────────────────────────────────────────

    def set_data(self, classification_map, ground_truth, confidence_map, error_map):
        self.classification_map = classification_map
        self.ground_truth       = ground_truth
        self.confidence_map     = confidence_map
        self.error_map          = error_map
        self._update_display(animate=False)
        QTimer.singleShot(50, self._reset_zoom)

    # ── Mode switching ────────────────────────────────────────────────────────

    def _set_mode(self, mode: str):
        self.current_mode = mode
        for lbl, btn in self._mode_btns.items():
            btn.setChecked(lbl == mode)
        self._update_display(animate=True)

    def _update_display(self, animate: bool = False):
        if self.classification_map is None:
            return

        mode = self.current_mode

        if mode == "Prediction":
            data = self.classification_map.astype(np.float32)
            lut  = self._lut_tab20
            vmin, vmax = 0, max(data.max(), 1)

        elif mode == "Ground Truth":
            data = self.ground_truth.astype(np.float32)
            lut  = self._lut_tab20
            vmin, vmax = 0, max(data.max(), 1)

        elif mode == "Error Map":
            data = self.error_map.astype(np.float32)
            colors = np.array([
                [15,  15,  15,  255],   # 0 unlabeled → dark
                [46,  204, 113, 255],   # 1 correct   → green
                [231, 76,  60,  255],   # 2 wrong     → red
            ], dtype=np.uint8)
            lut  = colors
            vmin, vmax = 0, 2

        else:  # Confidence
            data = self.confidence_map.astype(np.float32)
            cmap = pg.colormap.get('viridis', source='matplotlib')
            lut  = cmap.getLookupTable(0.0, 1.0, 256)
            vmin, vmax = 0.0, 1.0

        self.img_item.setLookupTable(lut)
        self.img_item.setLevels([vmin, vmax])

        if animate:
            self.img_item.animated_set_image(data)
        else:
            self.img_item.setImage(data)

    # ── Zoom ─────────────────────────────────────────────────────────────────

    def _reset_zoom(self):
        if self.classification_map is None:
            return
        M, N = self.classification_map.shape
        self.plot.vb.setRange(xRange=(0, N), yRange=(0, M), padding=0.01)

    # ── Export ───────────────────────────────────────────────────────────────

    def _export_image(self):
        if self.classification_map is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Map", "classification_map.png",
            "PNG Images (*.png)"
        )
        if path:
            exporter = pg.exporters.ImageExporter(self.plot)
            exporter.export(path)

    # ── Mouse hover ──────────────────────────────────────────────────────────

    def _on_mouse_moved(self, pos):
        if self.classification_map is None:
            return

        pt = self.plot.vb.mapSceneToView(pos)
        x, y = int(pt.x()), int(pt.y())
        M, N = self.classification_map.shape

        if not (0 <= y < M and 0 <= x < N):
            return

        pred_cls  = int(self.classification_map[y, x])
        true_cls  = int(self.ground_truth[y, x])
        confidence = float(self.confidence_map[y, x])
        is_correct = (pred_cls == true_cls) if true_cls > 0 else None

        self._info.update_pixel(x, y, true_cls, pred_cls, confidence, is_correct)

        self.pixel_hovered.emit(x, y, {
            'x': x, 'y': y,
            'true_class': true_cls,
            'pred_class': pred_cls,
            'confidence': confidence,
            'is_correct': is_correct,
        })
