"""
Metrics display panel
  - Overall metric cards (OA / AA / Kappa)
  - Per-class accuracy: badge + animated progress bar
  - Confusion matrix: full-width, animated cell reveal, hover highlight,
    tooltips via status bar label, row/col headers
"""

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QFrame, QGroupBox, QSizePolicy, QApplication
)
from PySide6.QtCore import Qt, QTimer, QRect, QPoint, QPropertyAnimation, QEasingCurve, Property
from PySide6.QtGui import (
    QColor, QPainter, QFont, QPen, QBrush,
    QLinearGradient, QFontMetrics
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _acc_rgb(acc: float):
    if acc >= 90:   return (46, 204, 113)
    elif acc >= 70: return (241, 196, 15)
    else:           return (231, 76, 60)

def _acc_hex(acc):
    r,g,b = _acc_rgb(acc); return f"#{r:02x}{g:02x}{b:02x}"


# ─────────────────────────────────────────────────────────────────────────────
# Animated accuracy bar
# ─────────────────────────────────────────────────────────────────────────────

class _AccBar(QWidget):
    def __init__(self, accuracy, r, g, b, parent=None):
        super().__init__(parent)
        self._target = accuracy / 100.0
        self._current = 0.0
        self._r, self._g, self._b = r, g, b
        self.setFixedHeight(20)
        # Animate fill
        self._timer = QTimer(self)
        self._timer.setInterval(16)
        self._timer.timeout.connect(self._tick)
        QTimer.singleShot(50, self._timer.start)

    def _tick(self):
        self._current = min(self._current + 0.04, self._target)
        self.update()
        if self._current >= self._target:
            self._timer.stop()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.setBrush(QBrush(QColor(50, 50, 50)))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(0, (h-8)//2, w, 8, 4, 4)
        fw = max(8, int(w * self._current))
        grad = QLinearGradient(0, 0, fw, 0)
        grad.setColorAt(0, QColor(self._r, self._g, self._b, 150))
        grad.setColorAt(1, QColor(self._r, self._g, self._b, 255))
        p.setBrush(QBrush(grad))
        p.drawRoundedRect(0, (h-8)//2, fw, 8, 4, 4)
        p.end()


# ─────────────────────────────────────────────────────────────────────────────
# Per-class row
# ─────────────────────────────────────────────────────────────────────────────

class _ClassCard(QFrame):
    def __init__(self, cls_id, accuracy, samples, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.NoFrame)
        self.setFixedHeight(38)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        r, g, b = _acc_rgb(accuracy)
        chex = f"#{r:02x}{g:02x}{b:02x}"
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 2, 6, 2)
        lay.setSpacing(8)
        badge = QLabel(str(cls_id))
        badge.setFixedSize(26, 26)
        badge.setAlignment(Qt.AlignCenter)
        badge.setStyleSheet(f"QLabel{{background:{chex};color:#fff;font-weight:bold;"
                            f"font-size:10px;border-radius:13px;}}")
        lay.addWidget(badge)
        bar = _AccBar(accuracy, r, g, b)
        bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lay.addWidget(bar)
        pct = QLabel(f"{accuracy:.1f}%")
        pct.setFixedWidth(50)
        pct.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        pct.setStyleSheet(f"color:{chex};font-weight:bold;font-size:12px;")
        lay.addWidget(pct)
        smp = QLabel(f"{samples:,}")
        smp.setFixedWidth(46)
        smp.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        smp.setStyleSheet("color:#888;font-size:10px;")
        lay.addWidget(smp)


# ─────────────────────────────────────────────────────────────────────────────
# Confusion Matrix — full-width, animated, hoverable
# ─────────────────────────────────────────────────────────────────────────────

class _ConfusionMatrixWidget(QWidget):
    """
    Full-width confusion matrix with:
      • Animated cell reveal (fade in row by row)
      • Hover highlight (cross-hair row+col dimming)
      • Cell count text + row-normalised colour intensity
      • Axis labels and titles
    """

    _LABEL_W  = 28      # pixels reserved for row/col number labels
    _COL_TITLE_H = 18   # space for "Predicted" title at bottom
    _ROW_TITLE_W = 14   # space for rotated "True" title on left
    _PAD = 4

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cm:      np.ndarray | None = None
        self._classes: list[int] = []
        self._norm:    np.ndarray | None = None

        self._reveal   = 0.0      # 0→1, fraction of rows revealed
        self._hover_i  = -1       # hovered row
        self._hover_j  = -1       # hovered col

        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(20)
        self._anim_timer.timeout.connect(self._anim_tick)

        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(300)

    # ── data ─────────────────────────────────────────────────────────────────

    def set_confusion_matrix(self, cm: np.ndarray, classes: list[int]):
        self._cm      = cm
        self._classes = classes
        row_max = cm.max(axis=1, keepdims=True).astype(float)
        row_max[row_max == 0] = 1
        self._norm = cm.astype(float) / row_max
        self._reveal = 0.0
        self._hover_i = self._hover_j = -1
        self._anim_timer.start()
        self.update()

    # ── animation ────────────────────────────────────────────────────────────

    def _anim_tick(self):
        self._reveal = min(self._reveal + 0.03, 1.0)
        self.update()
        if self._reveal >= 1.0:
            self._anim_timer.stop()

    # ── geometry helpers ─────────────────────────────────────────────────────

    def _geometry(self):
        n  = len(self._classes)
        lw = self._LABEL_W
        rw = self._ROW_TITLE_W
        ch = self._COL_TITLE_H

        # Available grid area
        avail_w = self.width()  - lw - rw - self._PAD * 2
        avail_h = self.height() - lw - ch  - self._PAD * 2

        cell = max(12, min(avail_w, avail_h) // max(n, 1))

        grid_w = cell * n
        grid_h = cell * n

        ox = rw + lw + self._PAD + max(0, (avail_w - grid_w) // 2)
        oy = self._PAD + max(0, (avail_h - grid_h) // 2)

        return cell, ox, oy, n

    # ── mouse ─────────────────────────────────────────────────────────────────

    def mouseMoveEvent(self, ev):
        if self._cm is None:
            return
        cell, ox, oy, n = self._geometry()
        x, y = ev.position().x(), ev.position().y()
        j = int((x - ox) / cell)
        i = int((y - oy) / cell)
        if 0 <= i < n and 0 <= j < n:
            new_i, new_j = i, j
        else:
            new_i, new_j = -1, -1
        if (new_i, new_j) != (self._hover_i, self._hover_j):
            self._hover_i, self._hover_j = new_i, new_j
            self.update()

    def leaveEvent(self, _):
        self._hover_i = self._hover_j = -1
        self.update()

    # ── paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, _):
        if self._cm is None:
            p = QPainter(self)
            p.setPen(QColor(100, 100, 100))
            p.drawText(self.rect(), Qt.AlignCenter, "Run pipeline to see confusion matrix")
            p.end()
            return

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        cm   = self._cm
        norm = self._norm
        n    = len(self._classes)
        cell, ox, oy, _ = self._geometry()
        lw = self._LABEL_W
        rw = self._ROW_TITLE_W

        rows_visible = int(self._reveal * n) + 1   # partially revealed

        # ── cells ──────────────────────────────────────────────────────────
        for i in range(min(rows_visible, n)):
            row_alpha = 255
            if i == rows_visible - 1 and self._reveal < 1.0:
                frac = (self._reveal * n) - int(self._reveal * n)
                row_alpha = max(30, int(frac * 255))

            for j in range(n):
                v   = cm[i, j]
                nv  = norm[i, j]
                x   = ox + j * cell
                y   = oy + i * cell

                # base colour
                if i == j:
                    base = QColor(int(18 + nv*32), int(120 + nv*110), int(18 + nv*32))
                else:
                    vv   = int(28 + nv * 52)
                    base = QColor(vv, vv, int(vv * 1.5))

                # hover dimming: dim everything except hovered row & col
                if self._hover_i >= 0:
                    in_cross = (i == self._hover_i or j == self._hover_j)
                    alpha = row_alpha if in_cross else max(40, row_alpha // 3)
                else:
                    alpha = row_alpha

                base.setAlpha(alpha)
                p.fillRect(x, y, cell - 1, cell - 1, base)

                # hover highlight border on hovered cell
                if i == self._hover_i and j == self._hover_j:
                    p.setPen(QPen(QColor(255, 255, 255, 200), 1.5))
                    p.setBrush(Qt.NoBrush)
                    p.drawRect(x, y, cell - 1, cell - 1)

                # cell text — always draw when value > 0
                if v > 0 and row_alpha > 60:
                    font = QFont()
                    font.setPixelSize(max(6, min(11, cell // 2 - 1)))
                    font.setBold(i == j)
                    p.setFont(font)
                    txt_col = QColor(255, 255, 255, alpha) if nv > 0.28 else QColor(160, 160, 160, alpha)
                    p.setPen(txt_col)
                    txt = str(v) if v < 10000 else f"{v//1000}k"
                    p.drawText(x, y, cell-1, cell-1, Qt.AlignCenter, txt)

        # ── grid lines ─────────────────────────────────────────────────────
        p.setPen(QPen(QColor(65, 65, 65), 0.5))
        for k in range(n + 1):
            p.drawLine(ox, oy + k*cell, ox + n*cell, oy + k*cell)
            p.drawLine(ox + k*cell, oy, ox + k*cell, oy + n*cell)

        # ── row labels (True class IDs) ─────────────────────────────────
        lbl_font = QFont()
        lbl_font.setPixelSize(max(7, min(10, lw - 6)))
        p.setFont(lbl_font)
        p.setPen(QColor(180, 180, 180))
        for i, cls in enumerate(self._classes):
            alpha = 255
            if i >= rows_visible:
                alpha = 0
            elif i == rows_visible - 1 and self._reveal < 1.0:
                frac  = (self._reveal * n) - int(self._reveal * n)
                alpha = max(30, int(frac * 255))
            c = QColor(180, 180, 180, alpha)
            p.setPen(c)
            y = oy + i * cell
            p.drawText(ox - lw + 2, y, lw - 4, cell - 1,
                       Qt.AlignRight | Qt.AlignVCenter, str(cls))

        # ── column labels (Predicted class IDs) ─────────────────────────
        p.setPen(QColor(180, 180, 180))
        for j, cls in enumerate(self._classes):
            x = ox + j * cell
            p.drawText(x, oy + n*cell + 2, cell-1, lw,
                       Qt.AlignCenter | Qt.AlignTop, str(cls))

        # ── hover info overlay ──────────────────────────────────────────
        if self._hover_i >= 0 and self._hover_j >= 0:
            hi, hj = self._hover_i, self._hover_j
            true_cls = self._classes[hi]
            pred_cls = self._classes[hj]
            val      = cm[hi, hj]
            correct  = (hi == hj)
            tag      = "✓ Diagonal" if correct else "✗ Misclassified"
            tip      = f"  True: {true_cls}  →  Pred: {pred_cls}  |  {val}  |  {tag}  "

            tip_font = QFont()
            tip_font.setPixelSize(10)
            p.setFont(tip_font)
            fm      = QFontMetrics(tip_font)
            tw      = fm.horizontalAdvance(tip) + 12
            th      = fm.height() + 8
            tx      = min(ox + hj * cell, self.width() - tw - 4)
            ty_base = oy + hi * cell - th - 4
            ty      = ty_base if ty_base > 2 else oy + (hi+1)*cell + 4

            p.setBrush(QBrush(QColor(30, 30, 30, 220)))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(tx, ty, tw, th, 4, 4)
            c = QColor(46, 204, 113) if correct else QColor(231, 76, 60)
            p.setPen(c)
            p.drawText(tx, ty, tw, th, Qt.AlignCenter, tip)

        # ── axis titles ────────────────────────────────────────────────
        title_font = QFont()
        title_font.setPixelSize(10)
        title_font.setBold(True)
        p.setFont(title_font)
        p.setPen(QColor(130, 130, 130))

        # "Predicted" bottom
        p.drawText(ox, oy + n*cell + lw - 2, n*cell, 14,
                   Qt.AlignCenter, "Predicted")

        # "True" left (rotated)
        p.save()
        p.translate(rw - 2, oy + n*cell // 2)
        p.rotate(-90)
        p.drawText(-30, -6, 60, 14, Qt.AlignCenter, "True")
        p.restore()

        p.end()


# ─────────────────────────────────────────────────────────────────────────────
# Main MetricsPanel
# ─────────────────────────────────────────────────────────────────────────────

class MetricsPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _group_style():
        return """
            QGroupBox {
                font-weight:bold; font-size:11px;
                border:1px solid #444; border-radius:6px;
                margin-top:8px; padding-top:4px;
            }
            QGroupBox::title { subcontrol-origin:margin; left:10px; }
        """

    @staticmethod
    def _make_card(title, value):
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        frame.setStyleSheet("QFrame{background:#2a2a2a;border:1px solid #3a3a3a;border-radius:8px;}")
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        lay = QVBoxLayout(frame)
        lay.setSpacing(1)
        lay.setContentsMargins(10, 8, 10, 8)
        t = QLabel(title)
        t.setAlignment(Qt.AlignCenter)
        t.setStyleSheet("color:#888;font-size:10px;font-weight:bold;background:transparent;border:none;")
        lay.addWidget(t)
        v = QLabel(value)
        v.setObjectName("val")
        v.setAlignment(Qt.AlignCenter)
        v.setStyleSheet("color:#fff;font-size:18px;font-weight:bold;background:transparent;border:none;")
        lay.addWidget(v)
        return frame

    @staticmethod
    def _set_card(card, text, color="#ffffff"):
        lbl = card.findChild(QLabel, "val")
        if lbl:
            lbl.setText(text)
            lbl.setStyleSheet(
                f"color:{color};font-size:18px;font-weight:bold;background:transparent;border:none;"
            )

    # ── build UI ─────────────────────────────────────────────────────────────

    def _setup_ui(self):
        lay = QVBoxLayout(self)
        lay.setSpacing(8)
        lay.setContentsMargins(6, 6, 6, 6)

        # Overall
        og = QGroupBox("Overall Metrics")
        og.setStyleSheet(self._group_style())
        ol = QHBoxLayout()
        ol.setSpacing(10)
        ol.setContentsMargins(8, 8, 8, 8)
        self._oa  = self._make_card("OA",    "--")
        self._aa  = self._make_card("AA",    "--")
        self._kap = self._make_card("Kappa", "--")
        for c in (self._oa, self._aa, self._kap):
            ol.addWidget(c)
        og.setLayout(ol)
        lay.addWidget(og)

        # Per-class
        cg = QGroupBox("Per-Class Accuracy")
        cg.setStyleSheet(self._group_style())
        cl = QVBoxLayout()
        cl.setContentsMargins(4, 4, 4, 4)
        # header
        hdr = QWidget()
        hl  = QHBoxLayout(hdr)
        hl.setContentsMargins(6, 0, 6, 0)
        hl.setSpacing(8)
        for txt, w in [("#", 26), ("Accuracy", 0), ("%", 50), ("n", 46)]:
            lb = QLabel(txt)
            lb.setStyleSheet("color:#555;font-size:10px;font-weight:bold;")
            lb.setAlignment(Qt.AlignCenter)
            if w: lb.setFixedWidth(w)
            else: lb.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            hl.addWidget(lb)
        cl.addWidget(hdr)
        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(QFrame.NoFrame)
        sa.setStyleSheet("QScrollArea{background:transparent;}")
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sa.setMinimumHeight(130)
        sa.setMaximumHeight(220)
        self._cards_w = QWidget()
        self._cards_l = QVBoxLayout(self._cards_w)
        self._cards_l.setSpacing(2)
        self._cards_l.setContentsMargins(0, 0, 0, 0)
        self._cards_l.addStretch()
        sa.setWidget(self._cards_w)
        cl.addWidget(sa)
        cg.setLayout(cl)
        lay.addWidget(cg)

        # Confusion matrix — stretch=1 so it fills remaining space
        mg = QGroupBox("Confusion Matrix  (rows = True · cols = Predicted)")
        mg.setStyleSheet(self._group_style())
        ml = QVBoxLayout()
        ml.setContentsMargins(4, 4, 4, 4)
        self._cm_w = _ConfusionMatrixWidget()
        ml.addWidget(self._cm_w)
        mg.setLayout(ml)
        lay.addWidget(mg, stretch=1)   # ← fills all remaining vertical space

    # ── public ───────────────────────────────────────────────────────────────

    def set_metrics(self, metrics: dict):
        oa, aa, kappa = metrics['oa'], metrics['aa'], metrics['kappa']

        def pct_c(v): return "#2ecc71" if v>=90 else ("#f1c40f" if v>=70 else "#e74c3c")
        def kap_c(k): return "#2ecc71" if k>=.9  else ("#f1c40f" if k>=.7  else "#e74c3c")

        self._set_card(self._oa,  f"{oa*100:.2f}%",  pct_c(oa*100))
        self._set_card(self._aa,  f"{aa*100:.2f}%",  pct_c(aa*100))
        self._set_card(self._kap, f"{kappa:.4f}",     kap_c(kappa))

        # Rebuild per-class cards
        while self._cards_l.count() > 1:
            item = self._cards_l.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        per_class = metrics['per_class']
        for cls in sorted(per_class.keys()):
            acc  = per_class[cls]['accuracy'] * 100
            samp = per_class[cls]['samples']
            self._cards_l.insertWidget(self._cards_l.count()-1,
                                       _ClassCard(cls, acc, samp))

        self._cm_w.set_confusion_matrix(
            metrics['confusion_matrix'],
            sorted(per_class.keys())
        )
