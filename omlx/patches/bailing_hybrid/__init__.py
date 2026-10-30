# SPDX-License-Identifier: Apache-2.0
"""inclusionAI Ling 3.0 (``model_type: bailing_hybrid``) support for mlx-lm.

The pinned mlx-lm 0.31.3 has no ``bailing_hybrid`` model, so loads fail with
``ValueError: Model type bailing_hybrid not supported``. mlx-lm >= 0.32.0
ships it, but upgrading requires ``mlx>=0.32.2`` while this environment pins
``mlx==0.32.0`` (custom-kernel nanobind domain), so we vendor the upstream
model file from mlx-lm 0.32.2 (Apple, MIT) instead.

One helper the file needs, ``rope_utils.apply_yarn_mscale``, is also missing
from 0.31.3 — injected here (pure-python port of upstream).

Upstream-first: once mlx-lm ships ``bailing_hybrid`` natively (mlx-lm upgrade
past the mlx pin), registration becomes a no-op.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import math
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MLX_LM_VERSION = "0.32.2"
_APPLIED = False

_VENDORED_MODULES = (
    ("mlx_lm.models.bailing_hybrid", "bailing_hybrid_model.py", "mlx_lm.models"),
)


def _register_module(qualname: str, filename: str, package: str) -> None:
    if qualname in sys.modules:
        return

    file_path = Path(__file__).parent / filename
    spec = importlib.util.spec_from_file_location(qualname, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {qualname} from {file_path}")

    module = importlib.util.module_from_spec(spec)
    module.__package__ = package
    sys.modules[qualname] = module
    spec.loader.exec_module(module)

    parent = importlib.import_module(package)
    setattr(parent, qualname.rsplit(".", 1)[1], module)
    logger.info("Registered %s from %s", qualname, filename)


def _ensure_yarn_mscale() -> None:
    """0.31.3 lacks apply_yarn_mscale; inject the upstream pure-python port."""
    rope_utils = importlib.import_module("mlx_lm.models.rope_utils")
    if hasattr(rope_utils, "apply_yarn_mscale"):
        return

    def apply_yarn_mscale(scale: float, scaling_config: Optional[dict]) -> float:
        """Fold the yarn mscale into an attention scale (upstream mlx-lm 0.32.2)."""
        if not scaling_config:
            return scale
        rope_type = scaling_config.get("type") or scaling_config.get(
            "rope_type", "default"
        )
        mscale_all_dim = scaling_config.get("mscale_all_dim", 0)
        factor = scaling_config.get("factor", 1)
        if rope_type == "default" or not mscale_all_dim or factor <= 1:
            return scale
        s = 0.1 * mscale_all_dim * math.log(factor) + 1.0
        return scale * s * s

    setattr(rope_utils, "apply_yarn_mscale", apply_yarn_mscale)
    logger.info("Injected rope_utils.apply_yarn_mscale (from mlx-lm %s)", MLX_LM_VERSION)


def apply_bailing_hybrid_patch() -> bool:
    """Register bailing_hybrid support when the pinned mlx-lm does not provide it."""
    global _APPLIED
    if _APPLIED:
        return False

    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        logger.debug("mlx_lm not importable - bailing_hybrid patch skipped")
        return False

    try:
        importlib.import_module("mlx_lm.models.bailing_hybrid")
        upstream = True
    except ImportError:
        upstream = False

    if not upstream:
        _ensure_yarn_mscale()
        for qualname, filename, package in _VENDORED_MODULES:
            _register_module(qualname, filename, package)
        _APPLIED = True
        logger.info(
            "bailing_hybrid mlx-lm patch applied (vendored from mlx-lm %s)",
            MLX_LM_VERSION,
        )
        return True

    _APPLIED = True
    logger.debug("mlx_lm.models.bailing_hybrid already available upstream")
    return False
