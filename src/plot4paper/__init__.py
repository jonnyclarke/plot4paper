from importlib.resources import files
import yaml


def load_latex_config():
    config_path = files("plot4paper.data") / "latex.yaml"

    with open(config_path, "r") as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)