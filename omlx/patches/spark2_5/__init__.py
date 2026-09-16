# SPDX-License-Identifier: Apache-2.0
"""XHToken Spark-X2.5 (``model_type: spark2_5``) support for mlx-lm.

The released checkpoints (e.g. Spark-X2.5-4B) ship only PyTorch remote code
(``configuration_spark.py`` / ``modeling_spark.py``); stock mlx-lm has no
``spark2_5`` model, so every load fails with
``ValueError: Model type spark2_5 not supported``.

This patch vendors the MLX port from mlx-optiq 0.5.8 (MIT,
https://pypi.org/project/mlx-optiq/, ``optiq/mlx_lm_patches/spark2_5.py``)
and registers it as ``mlx_lm.models.spark2_5`` without modifying the pinned
mlx-lm package. mlx-optiq itself is *not* installed because it pins
``transformers<5.13``, which would downgrade this environment's 5.15.

The hook is upstream-first: once mlx-lm ships ``spark2_5`` natively the
registration becomes a no-op.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

OPTIQ_VERSION = "0.5.8"
OPTIQ_URL = "https://pypi.org/project/mlx-optiq/"

_APPLIED = False

_VENDORED_MODULES = (
    # (qualname, vendored filename, parent package)
    ("mlx_lm.models.spark2_5", "spark2_5_model.py", "mlx_lm.models"),
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


def apply_spark2_5_patch() -> bool:
    """Register Spark-X2.5 support when the pinned mlx-lm does not provide it."""
    global _APPLIED
    if _APPLIED:
        return False

    try:
        import mlx_lm  # noqa: F401
    except ImportError:
        logger.debug("mlx_lm not importable - spark2_5 patch skipped")
        return False

    try:
        importlib.import_module("mlx_lm.models.spark2_5")
        upstream = True
    except ImportError:
        upstream = False

    if not upstream:
        for qualname, filename, package in _VENDORED_MODULES:
            _register_module(qualname, filename, package)
        _APPLIED = True
        logger.info(
            "Spark-X2.5 mlx-lm patch applied (vendored from mlx-optiq %s)",
            OPTIQ_VERSION,
        )
        return True

    _APPLIED = True
    logger.debug("mlx_lm.models.spark2_5 already available upstream")
    return False


def is_applied() -> bool:
    return _APPLIED


__all__ = ["apply_spark2_5_patch", "is_applied", "OPTIQ_VERSION", "OPTIQ_URL"]
