"""
Cost-Sensitive K-Nearest Neighbours (CS-KNN)

Identical to the notebook implementation:
  - Standard sklearn KNeighborsClassifier for indexing
  - Custom weighted voting: weight = class_weight / (distance + ε)
  - Class weights inversely proportional to class frequency
    (balances minority classes)
"""

import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier


_EPSILON = 1e-6   # distance regulariser (avoids division by zero)


class CSKNN:
    """
    Cost-Sensitive K-Nearest Neighbours classifier.

    Parameters
    ----------
    n_neighbors : int
        Number of nearest neighbours.
    metric : str
        Distance metric passed to sklearn (euclidean / manhattan / chebyshev).
    """

    def __init__(self, n_neighbors: int = 5, metric: str = 'euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric

        self._knn: KNeighborsClassifier | None = None
        self._y_train: np.ndarray | None = None
        self._class_weights: dict | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'CSKNN':
        """
        Fit the classifier.

        Parameters
        ----------
        X_train : (n_train, n_features)
        y_train : (n_train,)
        """
        self._y_train = y_train.copy()

        # Compute inverse-frequency class weights
        class_counts = Counter(y_train)
        total = len(y_train)
        n_classes = len(class_counts)
        self._class_weights = {
            cls: total / (n_classes * count)
            for cls, count in class_counts.items()
        }

        # Fit underlying KNN (used only for index/distance look-up)
        self._knn = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            algorithm='auto',
            n_jobs=-1,
        )
        self._knn.fit(X_train, y_train)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using cost-sensitive weighted voting.

        Parameters
        ----------
        X : (n_samples, n_features)

        Returns
        -------
        preds : (n_samples,)
        """
        self._check_fitted()
        distances, indices = self._knn.kneighbors(X)
        return self._weighted_vote(distances, indices)

    def predict_proba_max(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict labels AND return the maximum normalised vote score
        as a proxy for confidence.

        Returns
        -------
        preds      : (n_samples,)
        confidence : (n_samples,)  values in [0, 1]
        """
        self._check_fitted()
        distances, indices = self._knn.kneighbors(X)
        preds = []
        confidences = []

        for dists, idxs in zip(distances, indices):
            votes = self._compute_votes(dists, idxs)
            total_vote = sum(votes.values()) + _EPSILON
            best_cls = max(votes, key=votes.get)
            preds.append(best_cls)
            confidences.append(votes[best_cls] / total_vote)

        return np.array(preds), np.array(confidences)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _weighted_vote(self, distances, indices) -> np.ndarray:
        preds = []
        for dists, idxs in zip(distances, indices):
            votes = self._compute_votes(dists, idxs)
            preds.append(max(votes, key=votes.get))
        return np.array(preds)

    def _compute_votes(self, dists: np.ndarray, idxs: np.ndarray) -> dict:
        votes: dict = {}
        for cls, dist in zip(self._y_train[idxs], dists):
            w = self._class_weights[cls] / (dist + _EPSILON)
            votes[cls] = votes.get(cls, 0.0) + w
        return votes

    def _check_fitted(self):
        if self._knn is None:
            raise RuntimeError("CSKNN is not fitted yet. Call fit() first.")
