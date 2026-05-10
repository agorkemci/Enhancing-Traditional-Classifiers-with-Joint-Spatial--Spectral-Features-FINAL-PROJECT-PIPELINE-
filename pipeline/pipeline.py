"""
ClassificationPipeline
======================
Orchestrates the full LDA → EMP → CS-KNN workflow and returns
all artefacts needed by the UI (maps, metrics, etc.).

Progress steps (emitted to the UI via progress_callback):
  1  Standardising
  2  LDA
  3  Back-projecting
  4  EMP
  5  Train/test split
  6  Training CS-KNN
  7  Predicting & building maps
"""

import numpy as np
from sklearn.model_selection import train_test_split

from .preprocessing import Preprocessor
from .csknn import CSKNN
from .metrics import MetricsCalculator


class ClassificationPipeline:
    """
    End-to-end LDA + EMP + CS-KNN pipeline.

    Parameters
    ----------
    params : dict
        n_lda         int    (default 8)
        k_neighbors   int    (default 5)
        metric        str    (default 'euclidean')
        test_size     float  (default 0.3)
        random_state  int    (default 42)
    """

    def __init__(self, params: dict):
        self.n_lda = params.get('n_lda', 8)
        self.k_neighbors = params.get('k_neighbors', 5)
        self.metric = params.get('metric', 'euclidean')
        self.test_size = params.get('test_size', 0.3)
        self.random_state = params.get('random_state', 42)

    # ------------------------------------------------------------------

    def run(
        self,
        data: np.ndarray,
        gt: np.ndarray,
        progress_callback=None,
    ) -> dict:
        """
        Execute the full pipeline.

        Parameters
        ----------
        data : (M, N, C) float64
        gt   : (M, N) int — 0 = unlabeled
        progress_callback : callable(step: int, msg: str) | None

        Returns
        -------
        dict with keys:
            classification_map  (M, N) int
            ground_truth        (M, N) int
            confidence_map      (M, N) float  [0, 1]
            error_map           (M, N) int    0=unlabeled 1=correct 2=wrong
            metrics             dict (from MetricsCalculator.compute_all)
        """

        def _cb(step, msg):
            if progress_callback:
                progress_callback(step, msg)

        M, N = gt.shape
        n_pixels = M * N
        mask = gt.reshape(-1) > 0

        # ---- Steps 1-4: Preprocessing (LDA + EMP) ------------------
        preprocessor = Preprocessor(n_lda=self.n_lda)
        X_emp_labeled, y_labeled, X_emp_full = preprocessor.fit_transform(
            data, gt, progress_callback=progress_callback
        )

        # ---- Step 5: Train/test split ------------------------------
        _cb(5, "Splitting train / test sets…")
        X_train, X_test, y_train, y_test = train_test_split(
            X_emp_labeled,
            y_labeled,
            test_size=self.test_size,
            stratify=y_labeled,
            random_state=self.random_state,
        )

        # ---- Step 6: Fit CS-KNN ------------------------------------
        _cb(6, f"Training CS-KNN (k={self.k_neighbors}, metric={self.metric})…")
        clf = CSKNN(n_neighbors=self.k_neighbors, metric=self.metric)
        clf.fit(X_train, y_train)

        # ---- Step 7: Predictions -----------------------------------
        _cb(7, "Generating classification map…")

        # Test-set metrics
        y_pred_test, _ = clf.predict_proba_max(X_test)
        metrics = MetricsCalculator.compute_all(y_test, y_pred_test)

        # Full-image prediction (labeled pixels only)
        X_all_labeled = X_emp_full[mask]
        pred_all, conf_all = clf.predict_proba_max(X_all_labeled)

        # Build output maps
        full_pred = np.zeros(n_pixels, dtype=np.int32)
        full_conf = np.zeros(n_pixels, dtype=np.float32)
        full_pred[mask] = pred_all
        full_conf[mask] = conf_all

        classification_map = full_pred.reshape(M, N)
        confidence_map = full_conf.reshape(M, N)

        # Error map: 0=unlabeled, 1=correct, 2=incorrect
        error_flat = np.zeros(n_pixels, dtype=np.int32)
        gt_flat = gt.reshape(-1)
        correct = pred_all == gt_flat[mask]
        error_flat[mask] = np.where(correct, 1, 2)
        error_map = error_flat.reshape(M, N)

        return {
            'classification_map': classification_map,
            'ground_truth': gt,
            'confidence_map': confidence_map,
            'error_map': error_map,
            'metrics': metrics,
        }
