"""
Python class to automate the generation of beautiful plots at
academic publication standard.
"""

import logging
import numpy as np
from typing import Optional, Set
import matplotlib.pyplot as plt
from matplotlib import rcParams
from cycler import cycler

import os

import subprocess as comsub
import yaml

logger = logging.getLogger(__name__)

congif_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "config.yaml"
)
with open(congif_path, "r") as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)

SET_PERMITTED_KEYS: Set[str] = list(config.keys())

# Compute the Golden Ratio for nicely proportioned plots
GOLDEN_RATIO_FRACTION: float = 2.0 / (1.0 + np.sqrt(5.0))

# define the conversion from inch to pt length measurements
INCHES_TO_POINTS_CONVERSION: float = 72.27


class qualfig(object):
    """Class to automate basic formatting of plots to ensure consistency."""

    def __init__(
        self,
        key: str = "default",
        n_columns: int = 1,
        height_fraction: float = GOLDEN_RATIO_FRACTION,
        width_fraction: float = 1.0,
    ) -> None:
        """Initialise the qualfig (QUALITY-FIGURE) class.

        Args
            key (str):
                The publication template name.
                NOTE: default = "default"

            n_columns (int):
                Number of columns the plot should span.
                NOTE: default = 1

            height_fraction (float):
                The fractional height of the plot relative to the width.
                NOTE: default = 1.618 (The Golden Ratio)

            wf (Optional[float]):
                The fractional width of the plot relative to full page/column
                width.
                NOTE: default = 1.000
        """

        if key.upper() not in SET_PERMITTED_KEYS:
            raise KeyError(
                f"Selected key '{key}' is not understood. "
                f"Available keys are: {SET_PERMITTED_KEYS}"
            )

        plot_config = config[key.upper()]

        self.save_suffix = f"{plot_config['savesuf']}.pdf"

        if n_columns == 1:
            plot_width = plot_config["texcolumnwidth"]
        elif n_columns == 2:
            plot_width = plot_config["texlinewidth"]
        else:
            raise ValueError(
                f"Number of columns must be 1 or 2, not {n_columns}"
            )

        self.plot_width = plot_width * width_fraction

        max_permitted_height = 0.8 * plot_config["texheight"]

        self.plot_height = min(
            max_permitted_height,
            self.plot_width * height_fraction
        )

        if self.plot_height == max_permitted_height:
            logger.warning(
                "Plot will exceed max page height. Curtailed to fit page."
            )

        logger.debug(
            "Figure size (inches): "
            f"{self.plot_width / INCHES_TO_POINTS_CONVERSION} X "
            f"{self.plot_height / INCHES_TO_POINTS_CONVERSION}"
        )

        rcParams["figure.subplot.top"] = 0.95

        rcParams["figure.subplot.bottom"] = 0.2 if n_columns == 1 else 0.1
        rcParams["figure.subplot.left"] = 0.16 if n_columns == 1 else 0.08
        rcParams["figure.subplot.right"] = 0.84 if n_columns == 1 else 0.92

        rcParams["figure.figsize"] = (
            self.plot_width / INCHES_TO_POINTS_CONVERSION,
            self.plot_height / INCHES_TO_POINTS_CONVERSION,
        )

        # configure text and line widths
        font_size = 8
        linewidth = 0.5

        rcParams["font.size"] = font_size
        rcParams["axes.labelsize"] = font_size
        rcParams["axes.titlesize"] = font_size

        rcParams["xtick.major.width"] = linewidth - 0.1
        rcParams["xtick.minor.width"] = linewidth - 0.1
        rcParams["ytick.major.width"] = linewidth - 0.1
        rcParams["ytick.minor.width"] = linewidth - 0.1
        rcParams["lines.linewidth"] = linewidth
        rcParams["grid.linewidth"] = linewidth

        rcParams["xtick.minor.visible"] = True
        rcParams["ytick.minor.visible"] = True

        rcParams["xtick.bottom"] = True
        rcParams["xtick.top"] = True

        rcParams["ytick.left"] = True
        rcParams["ytick.right"] = True

        rcParams["xtick.direction"] = "in"
        rcParams["ytick.direction"] = "in"

        rcParams["errorbar.capsize"] = 4

        rcParams["legend.fontsize"] = font_size
        rcParams["legend.framealpha"] = 0.0
        rcParams["legend.edgecolor"] = "w"

        rcParams["xtick.labelsize"] = font_size - 1
        rcParams["ytick.labelsize"] = font_size - 1

        rcParams["font.family"] = "sans-serif"  # default
        rcParams["text.usetex"] = True
        plt.rc("text.latex", preamble=r"\usepackage{underscore}")

        rcParams["contour.negative_linestyle"] = "solid"

        # determine the new cycling order
        rcParams["axes.prop_cycle"] = cycler(color="brckmgy")

    def is_colorbar_axis(self, axis: plt.Axes) -> None:
        """Function to tell script an axis is a colourbar axis.
        Reconfigures the tick settings.

        Args:
            axis (plt.Axes):
                Axis that should be treated as a colourbar axis.
        """

        axis.minorticks_off()
        axis.tick_params(which="major", direction="out")

    def remove_axis_border(self, axis) -> None:
        """Function to hide all borders and ticks of axes used to generate
        custom labels over multirow/col plots.

        Args:
            axis (plt.Axes):
                Axis that should have all borders removed.
        """
        for edge in ["top", "bottom", "left", "right"]:
            axis.spines[edge].set_visible(False)

        axis.set_xticks([])
        axis.set_yticks([])

        axis.patch.set_alpha(0.0)

    def set_label_spaces(
        self,
        side: Optional[float] = None,
        bottom: Optional[float] = None,
        top: Optional[float] = None,
        left: Optional[float] = None,
        right: Optional[float] = None,
    ) -> None:
        """Function to customise the plot whitespace borders to allow more
        elaborate axes labels or titles.
        A value of 0 means no space will be reserved along that border of
        the figure.
        A value of 0.05 corresponds to 5% of the figure will be whitespace gap
        along that gap.

        Args:
            side (Optional[float]):
                Fractional width to reserve at the side for white-space.

            bottom (Optional[float]):
                Fractional width to reserve at the bottom for white-space.

            top (Optional[float]):
                Fractional width to reserve at the top for white-space.

            left (Optional[float]):
                Fractional width to reserve at the left for white-space.

            right (Optional[float]):
                Fractional width to reserve at the right for white-space.
        """

        if side:
            if left or right:
                logger.warning(
                    "You have set both 'side' and one of 'left'/'right' "
                    "Using 'side' configuration only"
                )

            rcParams["figure.subplot.left"] = side
            rcParams["figure.subplot.right"] = 1 - side

        else:

            if left:
                rcParams["figure.subplot.left"] = left

            if right:
                rcParams["figure.subplot.right"] = 1 - right

        if bottom:
            rcParams["figure.subplot.bottom"] = bottom

        if top:
            rcParams["figure.subplot.top"] = 1 - top

    def _crop_figure(self, fname: str, dots_per_inch: int) -> None:
        """
        Function to crop figures correctly to the correct size by removing
        whitespace.
        This process operates in PdF format but can then be converted to other
        file types.

        Args:
            fname (str):
                Name of the file to be saved.

            dots_per_inch (int):
                Resolution of the figure.
        """
        name = "dum.pdf"
        plt.savefig(name, dpi=dots_per_inch)
        command = "gs -sDEVICE=bbox -dNOPAUSE -dBATCH %s" % name
        r = comsub.getoutput(command).split()
        bbox = [float(a) for a in r[-4:]]
        logger.debug(bbox)
        pdf_crop_tuple = (name, fname, 0, bbox[1], self.plot_width, bbox[3])
        os.system(
            'pdfcrop %s %s --bbox " %.6f %.6f %.6f %.6f " ' % pdf_crop_tuple
        )
        command = "rm %s" % (name)
        os.system(command)

    def save(
        self, fname: str, extension: str = "pdf", dots_per_inch: int = 1000
    ) -> None:
        """Save file maintaining the quality figure formatting.

        Args:
            fname (str):
                File name to save figure under.

            extension (str):
                File type to save as.
                NOTE: default = pdf

            dots_per_inch (int):
                Figure resolution.
                More dots will allow better zoom features but will make file
                larger.
                NOTE: default = 1000
        """

        # strip the .pdf extension if it is there
        if fname.endswith(".pdf"):
            fname = fname[:-4]

        # now construct the save name we will use...
        savename = f"{fname}{self.save_suffix}"

        # initially crop the figure
        self._crop_figure(fname=savename, dots_per_inch=dots_per_inch)

        if extension == "eps":
            # now convert to eps
            command = "pdftops -eps %s" % (savename)
            os.system(command)
            command = "rm %s" % (savename)
            os.system(command)
