import pytest

from plot4paper.corner.corner import CornerPlot


@pytest.mark.parametrize(
    "shape",
    [
        2,
        (2, 2),
    ],
)
def test_class_initialisation(shape):
    _ = CornerPlot(shape=shape)


@pytest.mark.parametrize(
    "shape",
    [
        1.0,
        (1, 2, 3),
    ],
)
def test_class_initialisation_failure(shape):
    with pytest.raises(ValueError):
        _ = CornerPlot(shape=shape)


def test_get_fit_axis():
    c = CornerPlot(shape=3)
    _, _ = c.get_figure_axes()


def test_generate_top_right_panel():
    c = CornerPlot(shape=3)
    _ = c.generate_top_right_panel(dleft=2, ddown=2)
