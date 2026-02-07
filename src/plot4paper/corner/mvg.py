
import numpy as np
from scipy.stats import multivariate_normal


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

        self.kwargs = kwargs

    def descale(
            self,
            lst_mu: np.array,
            lst_std: np.array
            ):

        std_diag = np.diag(lst_std)

        return MultiVariateGaussianMixture(
            alpha=self.alpha,
            mu=(self.mu * lst_std) + lst_mu,
            cov=(
                std_diag @ self.cov @ std_diag
            ),
            **self.kwargs
        )

    def compute_onto_grid(
            self,
            pxyz: np.ndarray
            ) -> np.ndarray:

        z = np.zeros(len(pxyz))
        for (a, m, c) in zip(self.alpha, self.mu, self.cov):
            z += a * multivariate_normal.pdf(
                pxyz,
                mean=m,
                cov=c
            )

        return z

    def total_weight(self) -> float:
        return np.sum(self.alpha)

    def print(self):
        print(f"Weights: {self.alpha}")
        print(f"Means:\n{self.mu}")
        print(f"Covariances:\n{self.cov}")
