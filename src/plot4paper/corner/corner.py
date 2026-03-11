from typing import Tuple, Union, List

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import ScalarFormatter

from scipy.stats import gaussian_kde

import numpy as np

from plot4paper.corner.mvg import MultiVariateGaussianMixture


class CornerPlot:

    def __init__(
            self,
            shape: Union[int, Tuple[int, int]],
            axis_wspace: float = 0.05,
            axis_hspace: float = 0.05
            ) -> None:
        """

        Args:
            n_param_lower_left (int):
                The number of parameters to plot in the lower-left corner.
        """

        if isinstance(shape, int):
            self._shape = (shape, shape)

        elif (
                isinstance(shape, tuple)
                and len(shape) == 2
                and all(isinstance(x, int) for x in shape)
                ):
            self._shape = shape

        else:
            raise ValueError("shape must be int or (int, int)")

        # set up figure
        self._fig = plt.figure(1)
        self._gs = gridspec.GridSpec(
            nrows=self._shape[0],
            ncols=self._shape[1],
            wspace=axis_wspace,
            hspace=axis_hspace
        )
        self._axes = {}

        self._ll_grids = {}
        self._ll_n_theta = -1

        self._ur_grids = {}
        self._ur_n_theta = -1

    def get_figure_axes(self):
        return (self._fig, self._axes)

    def generate_top_right_panel(
            self,
            dleft: int = 1,
            ddown: int = 1
            ) -> plt.axis:
        """"""
        return plt.subplot(self._gs[:ddown, -dleft:])

    # +------------+
    # | LOWER LEFT |
    # +------------+

    def _ll_ii(self, i: int) -> int:
        """Function to convert the *horizontal* index in corner space to
        grid spec coordinates.
        """
        return i

    def _ll_jj(self, j: int) -> int:
        """Function to convert the *vertical* index in corner space to
        grid spec coordinates.
        """
        return self._shape[0] - 1 - j

    def _ll_jj_data(self, j: int) -> int:
        """Function to map the j index in corner space to the
        data index of the y-coordinate"""
        return self._ll_n_theta - 1 - j

    def _is_ll_histogram(self, i: int, j: int) -> bool:
        return (i + j + 1) == self._ll_n_theta

    @staticmethod
    def ll_key_formatter(i: int, j: int) -> str:
        return f"ll_{i}_{j}"

    def construct_lower_left_axes(
            self,
            n_theta: int
            ) -> None:

        self._ll_n_theta = n_theta

        for i in range(self._ll_n_theta):
            key_i0 = self.ll_key_formatter(i, 0)

            for j in range(self._ll_n_theta - i):
                key_0j = self.ll_key_formatter(0, j)

                key = self.ll_key_formatter(i, j)

                self._axes[key] = plt.subplot(
                    self._gs[self._ll_jj(j), self._ll_ii(i)]
                )

                # this ensures that the x-axes will always remain in syn
                if j > 0:
                    self._axes[key].sharex(self._axes[key_i0])

                if not self._is_ll_histogram(i, j):
                    self._axes[key].sharey(self._axes[key_0j])

                # set tick formatter to avoid irritating plotting defaults
                self._axes[key].xaxis.set_major_formatter(
                    ScalarFormatter(useOffset=False)
                )
                self._axes[key].yaxis.set_major_formatter(
                    ScalarFormatter(useOffset=False)
                )

                if j > 0:
                    # we are not on bottom row so don't want x-tick labels
                    self._axes[key].tick_params(axis="x", labelbottom=False)

                if i > 0:
                    # we are not on leftmost column so don't want y-tick labels
                    self._axes[key].tick_params(axis="y", labelleft=False)

                if self._is_ll_histogram(i, j):
                    self._axes[key].set_yticks([])
                    self._axes[key].tick_params(
                        which="major",
                        direction="in",
                        bottom=True,
                        top=True,
                        left=False,
                        right=False
                    )
                else:
                    self._axes[key].tick_params(
                        which="major",
                        direction="in",
                        bottom=True,
                        left=True,
                        top=True,
                        right=True
                    )

    def configure_lower_left_grids(
            self,
            lst_grid_boundaries: np.ndarray,
            n_bins: Union[int, np.array]
            ) -> None:
        """"""
        assert len(lst_grid_boundaries) == self._ll_n_theta

        if isinstance(n_bins, int):
            n_bins = [n_bins for _ in range(self._ll_n_theta)]

        self._ll_grids["bounds"] = lst_grid_boundaries
        self._ll_grids["bins"] = n_bins

    def configure_lower_left_grids_from_data(
            self,
            data: np.ndarray,
            n_bins: Union[int, np.array]
            ) -> None:
        """"""
        assert data.shape[1] == self._ll_n_theta

        if isinstance(n_bins, int):
            n_bins = [n_bins for _ in range(self._ll_n_theta)]

        min_max_bounds = np.vstack((
            np.min(data, axis=0),
            np.max(data, axis=0)
        )).T
        d = np.array([row[1]-row[0] for row in min_max_bounds])
        d *= 0.1
        min_max_bounds[:, 0] -= d
        min_max_bounds[:, 1] += d

        self.configure_lower_left_grids(
            lst_grid_boundaries=min_max_bounds,
            n_bins=n_bins
        )

    # +-----------------------------+
    # | PLOT HISTOGRAM / COLOUR-MAP |
    # +-----------------------------+

    def add_lower_left_distributions(
            self,
            data: np.ndarray,
            apply_smoothing: bool = False,
            hist_color: str = "b",
            map_colorscheme: plt.cm = plt.cm.jet,
            ) -> None:
        """"""

        assert self._ll_grids

        (N, ntheta) = data.shape
        assert ntheta == self._ll_n_theta

        for i in range(self._ll_n_theta):
            for j in range(self._ll_n_theta - i):

                key = self.ll_key_formatter(i, j)

                if self._is_ll_histogram(i, j):
                    self.plot_1d_histogram(
                        axis=self._axes[key],
                        data_slice=data[:, i],
                        x_boundaries=self._ll_grids["bounds"][i],
                        n_bins=self._ll_grids["bins"][i],
                        apply_smoothing=apply_smoothing,
                        hist_color=hist_color
                    )

                else:
                    self.plot_2d_histogram_basic(
                        axis=self._axes[key],
                        x_data_slice=data[:, i],
                        y_data_slice=data[:, self._ll_jj_data(j)],
                        x_boundaries=self._ll_grids["bounds"][i],
                        y_boundaries=self._ll_grids["bounds"][
                            self._ll_jj_data(j)
                        ],
                        x_n_bins=self._ll_grids["bins"][i],
                        y_n_bins=self._ll_grids["bins"][self._ll_jj_data(j)],
                        apply_smoothing=apply_smoothing,
                        map_colorscheme=map_colorscheme
                    )

    # +-----------------------------+
    # | PLOT MULTIVARIATE GAUSSIANS |
    # +-----------------------------+
    # TODO: extensive cleanup needed here

    def add_lower_left_multivariate_gaussian(
            self,
            mvg: MultiVariateGaussianMixture,
            weight: float = 1.0,
            hist_color: str = "red",
            contour_colours: str = "red"
            ) -> None:
        """TODO: clean up this function"""

        assert len(mvg.mu[0]) == self._ll_n_theta

        grids, dv = self._build_centroid_grid(
            boundaries=self._ll_grids["bounds"],
            n_bins=self._ll_grids["bins"],
            indexing="ij"
        )
        shape = grids[0].shape
        pxyz = np.column_stack([g.ravel() for g in grids])

        zz_nd = mvg.compute_onto_grid(pxyz=pxyz)
        zz_nd = np.reshape(zz_nd, shape)

        allaxes = {v for v in range(zz_nd.ndim)}

        for i in range(self._ll_n_theta):
            for j in range(self._ll_n_theta - i):

                key = self.ll_key_formatter(i, j)

                if self._is_ll_histogram(i, j):
                    x, dx = self._build_centroid_array(
                        self._ll_grids["bounds"][i][0],
                        self._ll_grids["bounds"][i][1],
                        self._ll_grids["bins"][i]
                    )

                    z = np.sum(zz_nd, axis=tuple(allaxes-{i}))

                    # normalise
                    z /= (np.sum(z) * dx)
                    z *= weight

                    self._axes[key].plot(x, z, color=hist_color)

                else:
                    ((xx, yy), dv) = self._build_centroid_grid(
                        boundaries=np.vstack((
                            self._ll_grids["bounds"][i],
                            self._ll_grids["bounds"][self._ll_jj_data(j)]
                        )),
                        n_bins=np.array([
                            self._ll_grids["bins"][i],
                            self._ll_grids["bins"][self._ll_jj_data(j)]
                        ]),
                        indexing="ij"
                    )
                    zz_2d = np.sum(
                        zz_nd,
                        axis=tuple(sorted(allaxes-{i, self._ll_jj_data(j)}))
                    )

                    # normalise
                    zz_2d /= (np.sum(zz_2d) * dv)
                    zz_2d *= weight

                    self.plot_2d_contours(
                        axis=self._axes[key],
                        xx=xx,
                        yy=yy,
                        zz=zz_2d,
                        colors=contour_colours
                    )

    # +-----------------+
    # | AXIS MANAGEMENT |
    # +-----------------+

    def add_lower_left_axis_labels(
            self,
            list_labels: List[str],
            ) -> None:

        assert len(list_labels) == self._ll_n_theta

        for i, label in enumerate(list_labels):

            key_i0 = self.ll_key_formatter(i, 0)
            self._axes[key_i0].set_xlabel(label)

            if i > 0:
                key_0j = self.ll_key_formatter(0, self._ll_jj_data(i))
                self._axes[key_0j].set_ylabel(label)

    def set_lower_left_axis_limits(
            self,
            list_limits: List[List[float]],
            ) -> None:

        assert len(list_limits) == self._ll_n_theta

        for i, limits in enumerate(list_limits):

            key_i0 = self.ll_key_formatter(i, 0)
            self._axes[key_i0].set_xlim(limits)

            if i > 0:
                key_0j = self.ll_key_formatter(0, i-1)
                self._axes[key_0j].set_ylim(limits)

    def set_lower_left_axis_limits_to_grid_bounds(self) -> None:
        list_bounds = self._ll_grids["bounds"]
        self.set_lower_left_axis_limits(list_limits=list_bounds)

    # +-------------------+
    # | PLOT 1D HISTOGRAM |
    # +-------------------+

    def plot_1d_histogram(
            self,
            axis,
            data_slice: np.array,
            x_boundaries: np.array,
            n_bins: int,
            apply_smoothing: bool = False,
            hist_color: str = "b",
            ) -> None:
        """"""

        if apply_smoothing:

            x_cent, dx = self._build_centroid_array(
                x_min=x_boundaries[0],
                x_max=x_boundaries[1],
                n_bins=n_bins
            )

            kde = gaussian_kde(data_slice)
            y = kde(x_cent)

            # normalise
            y /= (np.sum(y) * dx)

            axis.plot(x_cent, y, color=hist_color)

        else:
            y, _ = np.histogram(
                a=data_slice,
                bins=n_bins,
                range=x_boundaries,
                density=True
            )

            axis.plot(x_cent, y, color=hist_color)

        # we update the maximum of the axis to ensure all distributions
        # are in full view
        axis.set_ylim(0, max(np.nanmax(y) * 1.10, axis.get_ylim()[1]))

    # +-----------------------------------+
    # | N - DIMENSIONAL GRID CONSTRUCTION |
    # +-----------------------------------+

    @staticmethod
    def _build_centroid_array(x_min, x_max, n_bins) -> Tuple[np.array, float]:
        """"""
        x_edges = np.linspace(x_min, x_max, n_bins + 1)
        dx = x_edges[1] - x_edges[0]
        return (x_edges[:-1] + dx / 2.0, dx)

    def _build_centroid_grid(
            self,
            boundaries: np.ndarray,
            n_bins: np.array,
            indexing: str,
            ) -> tuple[tuple[np.ndarray, ...], float]:
        """"""
        centroids = []
        deltas = []

        for b, n in zip(boundaries, n_bins):
            # centroid locations
            c, _ = self._build_centroid_array(
                x_min=b[0],
                x_max=b[1],
                n_bins=n,
            )
            centroids.append(c)

            # bin width in this dimension
            deltas.append((b[1] - b[0]) / n)

        grid = np.meshgrid(*centroids, indexing=indexing)

        # n-D volume element
        dV = float(np.prod(deltas))

        return grid, dV

    # +-------------------+
    # | PLOT 2D HISTOGRAM |
    # +-------------------+

    @staticmethod
    def _construct_2d_histogram_arrays_non_smoothed(
            x_data_slice: np.array,
            y_data_slice: np.array,
            x_boundaries: np.array,
            y_boundaries: np.array,
            x_n_bins: int,
            y_n_bins: int,
            ) -> np.ndarray:

        H, _, _ = np.histogram2d(
            x=x_data_slice,
            y=y_data_slice,
            bins=[x_n_bins, y_n_bins],
            range=np.vstack((x_boundaries, y_boundaries)),
            density=True
        )

        shape = H.shape
        h = H.ravel()
        h[h == 0.] = np.nan  # this is to prevent colormap filling whole axis
        H = np.reshape(h, shape)

        # we rotate through 90 degrees as this will be plotted with imshow
        return np.rot90(H)

    def _construct_2d_histogram_arrays_smoothed(
            self,
            x_data_slice: np.array,
            y_data_slice: np.array,
            x_boundaries: np.array,
            y_boundaries: np.array,
            x_n_bins: int,
            y_n_bins: int,
            ) -> np.ndarray:
        """"""
        kde = gaussian_kde(np.stack((x_data_slice, y_data_slice)))

        ((xx, yy), dv) = self._build_centroid_grid(
            boundaries=np.vstack((x_boundaries, y_boundaries)),
            n_bins=np.array([x_n_bins, y_n_bins]),
            indexing="xy"
        )

        # we flip this up down as this will be plotted with imshow
        xx = np.flipud(xx)
        yy = np.flipud(yy)

        H = kde(np.stack((xx.ravel(), yy.ravel()))).reshape(xx.shape)

        # normalise to volume integral of 1
        H /= (np.sum(H) * dv)
        # this is to prevent colour spill due to negligible numbers
        H[H < np.nanmax(H)/1000.] = np.nan

        return H

    def _construct_2d_histogram_arrays(
            self,
            x_data_slice: np.array,
            y_data_slice: np.array,
            x_boundaries: np.array,
            y_boundaries: np.array,
            x_n_bins: int,
            y_n_bins: int,
            apply_smoothing: bool
            ) -> np.ndarray:
        """"""
        return (
            self._construct_2d_histogram_arrays_smoothed(
                x_data_slice=x_data_slice,
                y_data_slice=y_data_slice,
                x_boundaries=x_boundaries,
                y_boundaries=y_boundaries,
                x_n_bins=x_n_bins,
                y_n_bins=y_n_bins
            ) if apply_smoothing else
            self._construct_2d_histogram_arrays_non_smoothed(
                x_data_slice=x_data_slice,
                y_data_slice=y_data_slice,
                x_boundaries=x_boundaries,
                y_boundaries=y_boundaries,
                x_n_bins=x_n_bins,
                y_n_bins=y_n_bins
            )
        )

    def plot_2d_histogram_basic(
            self,
            axis,
            x_data_slice: np.array,
            y_data_slice: np.array,
            x_boundaries: np.array,
            y_boundaries: np.array,
            x_n_bins: int,
            y_n_bins: int,
            apply_smoothing: bool = False,
            map_colorscheme: plt.cm = plt.cm.jet
            ) -> None:
        """"""

        Z = self._construct_2d_histogram_arrays(
            x_data_slice=x_data_slice,
            y_data_slice=y_data_slice,
            x_boundaries=x_boundaries,
            y_boundaries=y_boundaries,
            x_n_bins=x_n_bins,
            y_n_bins=y_n_bins,
            apply_smoothing=apply_smoothing
        )

        axis.imshow(
            Z,
            extent=[
                x_boundaries[0],
                x_boundaries[1],
                y_boundaries[0],
                y_boundaries[1],
            ],
            cmap=map_colorscheme,
            aspect="auto"
        )

    # +-----------------------------+
    # | PLOT N-DIMENSIONAL GAUSSIAN |
    # +-----------------------------+

    def plot_2d_contours(
            self,
            axis,
            xx: np.ndarray,
            yy: np.ndarray,
            zz: np.ndarray,
            colors: str = "red"
            ) -> None:

        axis.contour(xx, yy, zz, levels=5, colors=colors)
