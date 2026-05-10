"""
Preprocessing pipeline: Standardisation → LDA → EMP

Steps (exactly as in the notebook):
  1. Flatten & StandardScaler
  2. LDA on labeled pixels (n_components configurable)
  3. Back-project LDA features to image space
  4. Extended Morphological Profile (grey opening + closing at 4 scales)
"""

import numpy as np
from scipy.ndimage import grey_opening, grey_closing
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


# Default morphological scales used in the notebook
_DEFAULT_SCALES = [3, 5, 7, 9]


class Preprocessor:
    """
    Applies LDA + EMP to a hyperspectral data cube.

    Parameters
    ----------
    n_lda : int
        Number of LDA discriminant components.
    scales : list[int], optional
        Structuring-element sizes for morphological profiles.
    """

    def __init__(self, n_lda: int = 8, scales: list | None = None):
        self.n_lda = n_lda
        self.scales = scales if scales is not None else _DEFAULT_SCALES

        self._scaler = StandardScaler()
        self._lda = LinearDiscriminantAnalysis(n_components=self.n_lda)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        data: np.ndarray,
        gt: np.ndarray,
        progress_callback=None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Full preprocessing: scale → LDA → EMP.

        Parameters
        ----------
        data : (M, N, C) float64
        gt   : (M, N) int  — 0 means unlabeled
        progress_callback : callable(step: int, msg: str) | None

        Returns
        -------
        X_emp_labeled : (n_labeled, n_emp_features)
            EMP features for labeled pixels only.
        y_labeled     : (n_labeled,)
            Corresponding class labels.
        X_emp_full    : (M*N, n_emp_features)
            EMP features for ALL pixels (needed to produce the full map).
        """

        def _cb(step, msg):
            if progress_callback:
                progress_callback(step, msg)

        M, N, C = data.shape
        n_pixels = M * N

        # ---- Step 1: Flatten + Standardise -------------------------
        _cb(1, "Standardising features…")
        X_flat = data.reshape(n_pixels, C)
        X_flat = self._scaler.fit_transform(X_flat)

        y = gt.reshape(-1)
        mask = y > 0
        X_labeled = X_flat[mask]
        y_labeled = y[mask]

        # ---- Step 2: LDA -------------------------------------------
        _cb(2, f"Applying LDA ({self.n_lda} components)…")
        # Cap n_components to legal maximum = n_classes - 1
        n_classes = len(np.unique(y_labeled))
        n_lda_actual = min(self.n_lda, n_classes - 1)
        if n_lda_actual != self.n_lda:
            self._lda.n_components = n_lda_actual

        X_lda = self._lda.fit_transform(X_labeled, y_labeled)

        # ---- Step 3: Back-project to image space -------------------
        _cb(3, "Back-projecting LDA to image…")
        lda_img = np.zeros((n_pixels, n_lda_actual), dtype=np.float64)
        lda_img[mask] = X_lda
        lda_img = lda_img.reshape(M, N, n_lda_actual)

        # ---- Step 4: EMP -------------------------------------------
        _cb(4, "Computing Extended Morphological Profiles…")
        emp_features = []

        for i in range(n_lda_actual):
            band = lda_img[:, :, i]
            profiles = [band]                              # original band
            for s in self.scales:
                profiles.append(grey_opening(band, size=s))
                profiles.append(grey_closing(band, size=s))
            emp_features.append(np.stack(profiles, axis=-1))

        emp_stack = np.concatenate(emp_features, axis=-1)  # (M, N, n_features)
        X_emp_full = emp_stack.reshape(n_pixels, -1)
        X_emp_labeled = X_emp_full[mask]

        return X_emp_labeled, y_labeled, X_emp_full
