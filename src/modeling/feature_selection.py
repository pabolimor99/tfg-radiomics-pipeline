import numpy as np
from scipy.stats import ranksums
from sklearn.base import BaseEstimator, TransformerMixin


class WilcoxonFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, top_k: int | None = None):
        self.top_k = top_k

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        n_features = X.shape[1]
        if self.top_k is None or int(self.top_k) <= 0 or int(self.top_k) >= n_features:
            self.selected_indices_ = np.arange(n_features, dtype=int)
            return self

        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("WilcoxonFeatureSelector requiere un problema binario.")

        class_a, class_b = classes[0], classes[1]
        p_values = np.full(n_features, np.inf, dtype=float)

        for idx in range(n_features):
            values = X[:, idx]
            group_a = values[(y == class_a) & ~np.isnan(values)]
            group_b = values[(y == class_b) & ~np.isnan(values)]
            if len(group_a) == 0 or len(group_b) == 0:
                continue
            try:
                _, p_value = ranksums(group_a, group_b)
            except ValueError:
                continue
            if np.isfinite(p_value):
                p_values[idx] = float(p_value)

        ranking = np.argsort(p_values, kind="stable")
        selected = ranking[: min(int(self.top_k), n_features)]
        finite_selected = selected[np.isfinite(p_values[selected])]
        if len(finite_selected) == 0:
            finite_selected = np.arange(min(int(self.top_k), n_features), dtype=int)

        self.selected_indices_ = np.sort(finite_selected.astype(int))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.selected_indices_]

    def get_support(self):
        return self.selected_indices_
