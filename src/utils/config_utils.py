"""
config_utils.py
===============
Helpers for resolving and instantiating classes from OmegaConf configs.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict

from omegaconf import DictConfig, OmegaConf


def get_obj_from_str(dotpath: str) -> Any:
    """Resolve ``'module.submodule.ClassName'`` → the class object."""
    module_path, cls_name = dotpath.rsplit(".", 1)
    return getattr(import_module(module_path), cls_name)


def instantiate_from_config(config: DictConfig | Dict[str, Any]) -> Any:
    """Instantiate an object from an OmegaConf section.

    Expected format::

        target: some.module.ClassName
        params:
          key1: val1
          key2: val2

    Returns an instance of ``ClassName(**params)``.
    """
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    target = config["target"]
    params = config.get("params", {})
    return get_obj_from_str(target)(**params)
