import os
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import yaml


def get_labels(config: dict):
    labels = []

    cond_labels = config.data.conditional_labels
    c_labels = config.data.classifier_labels

    if cond_labels:
        labels.extend(cond_labels)

    if c_labels:
        labels.extend(c_labels)

    return labels

def load_yaml_config(config_filename: str) -> dict:
    """Load yaml config.

    Args:
        config_filename: Filename to config.

    Returns:
        Loaded config.
    """
    with open(config_filename) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def make_exp_folder(config):
    """Create experiment folder.

    Args:
        config: Yaml config file.

    Returns:
        Experiment folder path.
    """
    experiment_folder = os.path.join(
        "./outputs", datetime.now().strftime("%Y-%m-%d/"), config["experiment_folder"]
    )
    Path(experiment_folder).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(experiment_folder, "config.yaml"), "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
    return experiment_folder
