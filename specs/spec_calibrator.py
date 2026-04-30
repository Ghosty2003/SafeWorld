"""
specs/spec_calibrator.py

根据环境配置 JSON 动态替换 spec 公式里的阈值。
"""

from __future__ import annotations
import copy
import json
from pathlib import Path


def load_env_config(config_path: str | Path) -> dict:
    with open(config_path) as f:
        return json.load(f)


def patch_formula_thresholds(
    formula: dict,
    threshold_overrides: dict[str, float],
) -> dict:
    """
    递归遍历公式树，把 atom 节点的阈值替换为配置文件里的值。

    threshold_overrides 格式:
        {
            "hazard_dist": 0.0,
            "velocity":    0.4,
            "goal_dist":  -0.2
        }
    """
    formula = copy.deepcopy(formula)
    node_type = formula.get("type")

    if node_type == "atom":
        dim = formula["dim"]
        if dim in threshold_overrides:
            formula["threshold"] = threshold_overrides[dim]
        return formula

    # 递归处理子节点
    if node_type in ("always", "eventually", "next"):
        formula["child"] = patch_formula_thresholds(
            formula["child"], threshold_overrides
        )

    elif node_type in ("and", "or"):
        formula["left"]  = patch_formula_thresholds(formula["left"],  threshold_overrides)
        formula["right"] = patch_formula_thresholds(formula["right"], threshold_overrides)

    elif node_type == "until":
        formula["left"]  = patch_formula_thresholds(formula["left"],  threshold_overrides)
        formula["right"] = patch_formula_thresholds(formula["right"], threshold_overrides)

    elif node_type == "not":
        formula["child"] = patch_formula_thresholds(formula["child"], threshold_overrides)

    return formula


def apply_env_config_to_spec(spec: dict, env_config: dict) -> dict:
    """
    用环境配置里的 ap_thresholds 覆盖 spec 公式中的阈值。

    env_config 里需要有:
        "ap_thresholds": {
            "hazard_dist": 0.0,
            "velocity":    0.4,
            ...
        }
    """
    spec = copy.deepcopy(spec)
    overrides = env_config.get("ap_thresholds", {})

    if not overrides:
        return spec  # 没有覆盖项，直接返回

    spec["formula"] = patch_formula_thresholds(spec["formula"], overrides)

    # 同时更新 description 里的阈值说明（可选）
    spec.setdefault("threshold_source", "env_config")

    return spec


def apply_env_config_to_specs(
    specs: list[dict],
    env_config: dict,
) -> list[dict]:
    """批量处理所有 spec"""
    return [apply_env_config_to_spec(s, env_config) for s in specs]