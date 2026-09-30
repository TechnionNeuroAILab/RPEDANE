"""Channel naming: signal family (DA/NE), hemisphere, NAc subregion."""
from __future__ import annotations

import re

_PAT = re.compile(r"^(?P<region>[A-Za-z]+)\((?P<hemi>[LR])\)-(?P<sensor>.+)$")


def parse_channel(name: str) -> dict | None:
    m = _PAT.match(str(name))
    if not m:
        return None
    sensor = m["sensor"]
    if sensor.endswith("DA"):
        signal = "DA"
    elif "LCAxonCa" in sensor:
        signal = "NE"
    else:
        return None
    return {"channel": str(name), "signal": signal, "region": m["region"], "hemi": m["hemi"]}


def classify_channel(name: str) -> str | None:
    info = parse_channel(name)
    return info["signal"] if info else None
