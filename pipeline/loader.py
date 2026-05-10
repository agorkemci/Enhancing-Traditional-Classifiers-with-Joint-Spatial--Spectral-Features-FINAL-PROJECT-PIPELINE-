"""
Dataset loading for hyperspectral .mat files.
Supports:
  - Single .mat file containing both data and ground truth
  - Two separate .mat files (data + GT)
"""

import numpy as np
import scipy.io as sio


# Known key names for common hyperspectral datasets
_DATA_KEY_HINTS = [
    'indian_pines_corrected', 'indian_pines',
    'paviaU', 'pavia',
    'salinas', 'salinas_corrected',
    'KSC', 'Botswana',
    'data', 'hsi', 'X',
]

_GT_KEY_HINTS = [
    'indian_pines_gt',
    'paviaU_gt', 'pavia_gt',
    'salinas_gt',
    'KSC_gt', 'Botswana_gt',
    'gt', 'groundtruth', 'y', 'labels',
]


def _pick_key(mat_dict: dict, hints: list, exclude_private: bool = True) -> str:
    """Pick the most likely key from a .mat dict using hint order."""
    keys = [k for k in mat_dict.keys() if not k.startswith('_')]

    # First try hints in order
    for hint in hints:
        if hint in keys:
            return hint

    # Fall back to the first non-private array key that is ≥2-D
    for k in keys:
        v = mat_dict[k]
        if isinstance(v, np.ndarray) and v.ndim >= 2:
            return k

    raise KeyError(
        f"Cannot auto-detect the target key. Available keys: {keys}"
    )


class DatasetLoader:
    """
    Load and hold a hyperspectral dataset.

    After calling load_mat() or load_separate(), access:
        .data  -> np.ndarray  shape (M, N, C)   float64
        .gt    -> np.ndarray  shape (M, N)       int
    """

    def __init__(self):
        self.data: np.ndarray | None = None
        self.gt: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_mat(self, file_path: str):
        """
        Load a single .mat file that contains BOTH the hyperspectral
        cube and the ground-truth label map.

        Returns
        -------
        (data_key, gt_key) : tuple[str, str]
        """
        mat = sio.loadmat(file_path)

        # The mat file should contain exactly two meaningful arrays
        candidate_keys = [k for k in mat.keys() if not k.startswith('_')]

        # Try to find GT key first (2-D array)
        gt_key = None
        data_key = None

        for k in candidate_keys:
            v = mat[k]
            if isinstance(v, np.ndarray):
                if v.ndim == 2:
                    gt_key = k
                elif v.ndim == 3:
                    data_key = k

        # Fallback to hints if ambiguous
        if data_key is None:
            data_key = _pick_key(mat, _DATA_KEY_HINTS)
        if gt_key is None:
            gt_key = _pick_key(mat, _GT_KEY_HINTS)

        self.data = mat[data_key].astype(np.float64)
        self.gt = mat[gt_key].astype(np.int32)

        self._validate()
        return data_key, gt_key

    def load_separate(self, data_path: str, gt_path: str):
        """
        Load from two separate .mat files.

        Returns
        -------
        (data_key, gt_key) : tuple[str, str]
        """
        data_mat = sio.loadmat(data_path)
        gt_mat = sio.loadmat(gt_path)

        data_key = _pick_key(data_mat, _DATA_KEY_HINTS)
        gt_key = _pick_key(gt_mat, _GT_KEY_HINTS)

        self.data = data_mat[data_key].astype(np.float64)
        self.gt = gt_mat[gt_key].astype(np.int32)

        self._validate()
        return data_key, gt_key

    def get_info(self) -> dict:
        """Return a summary dict suitable for the UI."""
        if self.data is None:
            raise RuntimeError("No dataset loaded yet.")

        M, N, C = self.data.shape
        mask = self.gt > 0
        classes = np.unique(self.gt[mask]).tolist()

        return {
            'shape': f"{M} × {N} × {C}  (rows × cols × bands)",
            'classes': len(classes),
            'labeled_pixels': int(mask.sum()),
            'total_pixels': M * N,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate(self):
        if self.data.ndim != 3:
            raise ValueError(
                f"Expected a 3-D hyperspectral cube, got shape {self.data.shape}"
            )
        if self.gt.shape != self.data.shape[:2]:
            raise ValueError(
                f"GT shape {self.gt.shape} does not match spatial dims "
                f"{self.data.shape[:2]}"
            )
