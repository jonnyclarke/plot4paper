
import numpy as np
from scipy.stats import multivariate_normal
from typing import Set


class MultiVariateGaussianMixture:
    """"""
    def __init__(
            self,
            alpha: np.array,
            mu: np.ndarray,
            cov: np.ndarray,
            **kwargs
            ) -> None:

        self.alpha = alpha
        self.mu = mu
        self.cov = cov

        self._metadata = kwargs  # store any extra keyword arguments

    def set_metadata(self, **kwargs):
        self._metadata.update(kwargs)

    @property
    def metadata(self):
        return self._metadata

    def iterate_gaussians(self):
        return zip(self.alpha, self.mu, self.cov)

    def compute_onto_grid(
            self,
            pxyz: np.ndarray
            ) -> np.array:

        z = np.zeros(len(pxyz))
        for (a, m, c) in self.iterate_gaussians():
            z += a * multivariate_normal.pdf(
                pxyz,
                mean=m,
                cov=c
            )

        return z

    def extract_subcomponents(
            self,
            set_idx: Set[int]
            ):
        _a = []
        _m = []
        _c = []
        for i, (a, m, c) in enumerate(self.iterate_gaussians()):
            if i in set_idx:
                _a.append(a)
                _m.append(m)
                _c.append(c)

        return MultiVariateGaussianMixture(
            alpha=np.array(_a),
            mu=np.array(_m),
            cov=np.array(_c)
        )

    def total_weight(self) -> float:
        return np.sum(self.alpha)

    def print(self):
        print(f"Weights: {self.alpha}")
        print(f"Means:\n{self.mu}")
        print(f"Covariances:\n{self.cov}")
