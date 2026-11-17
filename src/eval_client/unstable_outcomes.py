"""Per-run unstable layout receipts, independent of Isaac Sim."""

from collections.abc import Iterable
import json
import os
from pathlib import Path
from typing import Literal

from utils.save_file import save_json


def record_unstable_layouts(
    save_dir: str | os.PathLike,
    native_indices: Iterable[int],
    *,
    phase: Literal["reset", "evaluation"],
) -> None:
    """Atomically merge unstable receipts keyed by SeedManager's ordinal IDs.

    These IDs are internal layout indices, not layout filename suffixes. The
    sidecar is cumulative within save_dir and never adds scores or videos.
    """
    path = Path(save_dir) / "_skips.json"
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"details": {}}
    for index in native_indices:
        payload["details"][str(int(index))] = {"reason": "unstable", "phase": phase}
    save_json(payload, path)
