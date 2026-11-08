from plot4paper import PAPER_METRICS
import pytest
from typing import Union


@pytest.mark.parametrize(
    "scheme, parameter, truth",
    [("MNRAS", "texcolumnwidth", 240.0), ("DEFAULT", "savesuf", "")],
)
def test_load_latex_config(scheme: str, parameter: str, truth: Union[float, int, str]):
    assert PAPER_METRICS.get(scheme).get(parameter) == truth


def test_latex_config():
    """Test that all latex schema configs have the correct keys"""
    required_keys = {"texcolumnwidth", "texlinewidth", "texheight", "savesuf"}
    for _, schema in PAPER_METRICS.items():
        assert set(schema.keys()) == required_keys
