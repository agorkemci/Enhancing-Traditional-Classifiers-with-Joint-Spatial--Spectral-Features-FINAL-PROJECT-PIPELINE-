"""
Performance metrics: OA, AA, Kappa, per-class accuracy, confusion matrix.

Formulas are identical to those used in the notebook.
"""

import numpy as np
from sklearn.metrics import confusion_matrix


class MetricsCalculator:
    """Compute and store classification performance metrics."""

    @staticmethod
    def compute_all(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict:
        """
        Compute OA, AA, Kappa and per-class stats.

        Parameters
        ----------
        y_true : (n_test,)   ground-truth labels (labeled pixels only)
        y_pred : (n_test,)   predicted labels

        Returns
        -------
        dict with keys:
            oa              float
            aa              float
            kappa           float
            per_class       dict[class_id -> {'accuracy': float, 'samples': int}]
            confusion_matrix np.ndarray (n_classes × n_classes)
        """
        classes = np.unique(np.concatenate([y_true, y_pred]))
        cm = confusion_matrix(y_true, y_pred, labels=classes)

        # Overall Accuracy
        oa = np.trace(cm) / np.sum(cm)

        # Per-class accuracy
        row_sums = cm.sum(axis=1)
        class_acc = np.where(row_sums > 0, np.diag(cm) / row_sums, np.nan)

        # Average Accuracy (ignore classes with no test samples)
        aa = float(np.nanmean(class_acc))

        # Kappa
        total = np.sum(cm)
        po = np.trace(cm) / total
        pe = np.sum(cm.sum(axis=0) * cm.sum(axis=1)) / (total ** 2)
        kappa = float((po - pe) / (1 - pe)) if (1 - pe) != 0 else 0.0

        # Per-class dict
        per_class = {}
        for idx, cls in enumerate(classes):
            per_class[int(cls)] = {
                'accuracy': float(class_acc[idx]) if not np.isnan(class_acc[idx]) else 0.0,
                'samples': int(row_sums[idx]),
            }

        return {
            'oa': float(oa),
            'aa': float(aa),
            'kappa': float(kappa),
            'per_class': per_class,
            'confusion_matrix': cm,
        }
