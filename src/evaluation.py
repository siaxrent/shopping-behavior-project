import json
from typing import Any, Dict


def save_metrics(metrics: Dict[str, Any], path: str) -> None:
    """Save metrics dictionary to a JSON file with UTF-8 encoding."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


