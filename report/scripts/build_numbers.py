#!/usr/bin/env python
"""Merge every per-script ledger in report/numbers/*.json into report/numbers.json
and check that no key is defined twice with different values."""
from __future__ import annotations

import glob
import json
from pathlib import Path

from common import NUMBERS_DIR, REPORT

merged = {}
dups = []
for f in sorted(glob.glob(str(NUMBERS_DIR / "*.json"))):
    d = json.load(open(f))
    for k, v in d.items():
        if k in merged and merged[k]["value"] != v["value"]:
            dups.append((k, merged[k]["script"], v["script"]))
        merged[k] = v
with open(REPORT / "numbers.json", "w") as fh:
    json.dump(merged, fh, indent=1, sort_keys=True)
print(f"numbers.json: {len(merged)} entries from {len(glob.glob(str(NUMBERS_DIR / '*.json')))} ledgers")
if dups:
    print("CONFLICTS:", dups)
    raise SystemExit(1)
