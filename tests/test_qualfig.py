import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch
import pytest

from plot4paper.qualfig import QualityFigure, GOLDEN_RATIO_FRACTION


def test_golden_ratio():
    assert GOLDEN_RATIO_FRACTION < 1.0


@patch("matplotlib.pyplot.Figure.savefig")
def test_QualityFigure_save(mock_savefig):
    x = np.linspace(0, 1, 100)
    y = 2 * x**3 + x**2 + 3 * x + 2

    qfig = QualityFigure(use_latex=False)
    fig, ax = plt.subplots()

    ax.plot(x, y)

    qfig.save(fig, "dummy.pdf")
    plt.close(fig)

    # Assert that fig.savefig was called correctly
    mock_savefig.assert_called_once()


def test_QualityFigure_unknown_key():

    with pytest.raises(KeyError):
        _ = QualityFigure(key="__UNKNOWN__", use_latex=False)


def test_QualityFigure_two_columns():

    _ = QualityFigure(n_columns=2, height_fraction=100, use_latex=False)


def test_QualityFigure_n_columns():

    with pytest.raises(ValueError):
        _ = QualityFigure(n_columns=3, use_latex=False)


def test_QualityFigure_cbar_axis():

    qfig = QualityFigure(use_latex=False)
    fig, ax = plt.subplots(nrows=1, ncols=2)

    qfig.is_colorbar_axis(ax[1])

    assert len(ax[0].xaxis.get_minorticklocs()) > 0
    assert len(ax[1].xaxis.get_minorticklocs()) == 0

    plt.close(fig)


def test_QualityFigure_invisible_axis() -> None:

    qfig = QualityFigure(use_latex=False)
    fig, ax = plt.subplots()

    # Call your function
    qfig.remove_axis_border(ax)

    # Assert the axis is invisible
    assert not ax.get_visible(), "Axis should be invisible"

    # Assert the patch alpha is 0
    assert ax.patch.get_alpha() == 0.0, "Axis patch alpha should be 0.0"

    # Cleanup figure
    plt.close(fig)


def test_QualityFigure_label_spaces() -> None:

    qfig = QualityFigure(use_latex=False)
    fig, ax = plt.subplots()

    # Call your function
    qfig.set_label_spaces(side=0.2)
    qfig.set_label_spaces(top=0.2)
    qfig.set_label_spaces(bottom=0.1)
    qfig.set_label_spaces(left=0.1, right=0.01)
    qfig.set_label_spaces(side=0.3, left=0.1, right=0.01)
    plt.close(fig)
