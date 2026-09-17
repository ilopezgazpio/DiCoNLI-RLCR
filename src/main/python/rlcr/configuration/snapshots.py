"""Record resolved settings without discarding previous invocation settings."""
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import yaml


def save_resolved_config(output_dir, settings):
    """Save the latest configuration, archiving the previous one when it changes."""
    serialized = yaml.safe_dump(settings, sort_keys=False)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target = output / "resolved-config.yaml"
    if target.exists():
        if target.read_text() == serialized:
            return
        history = output / "config-history"
        history.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        target.rename(history / f"{stamp}-{uuid4().hex[:8]}.yaml")
    target.write_text(serialized)
