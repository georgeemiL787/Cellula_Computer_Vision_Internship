from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "model_config.json"
WEIGHTS_PATH = BASE_DIR / "weights" / "best_model.pth"


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)
