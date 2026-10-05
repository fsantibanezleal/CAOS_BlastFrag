#!/usr/bin/env python3
"""T10, the SDD gate (ADR-0075, ADR-0078): a design document exists, and every requirement names a gate that is real.

Mirrored from CAOS_OreFlow's scripts/check_sdd.py (the validated reference, 2026-09-28), with two additions:
the ADR-fit section ADR-0078 requires in every SDD, and named TypeScript test cases (`file.test.ts::name`) checked
the way pytest cases are.

Checks, in increasing strength:

1. The repository has ``docs/design/SDD.md`` with the nine sections the convention requires.
2. Every requirement of a live feature (a row ``| ID | statement | gate |`` of
   ``docs/design/features/<slug>/requirements.md``) is stated with SHALL and has a gate cell.
3. **The named gate exists.** The gate cell names at least one file in backticks; every named file is on disk,
   and a named test (``path.py::test_name`` or ``path.test.ts::name``) is defined in it.

Check 3 is the one that matters: a requirement can name ``tests/test_nothing.py::test_imaginary`` and pass check 2
while verifying nothing, which is the failure the rule exists for.

``Status: superseded`` at the top of a requirements.md skips it (listed); ``Status: planned`` applies checks 1 and 2
and requires a file gate that need not exist yet (counted apart, never reported as a real gate).

Standard library only; exit 1 on any finding. Usage: ``python scripts/check_sdd.py [repo_root]``.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
SDD = ROOT / "docs" / "design" / "SDD.md"
FEATURES = ROOT / "docs" / "design" / "features"

SECTIONS = (
    "Problem and non-goals", "Contracts", "Lanes", "Method ladder", "Cases", "Oracles", "Deploy driver",
    "Risks and kill criteria", "ADR fit",
)
ROW = re.compile(r"^\|\s*(?P<id>[A-Z]{2,}-\d+[a-z]?)\s*\|(?P<statement>.*?)\|(?P<gate>.*)\|\s*$")
TICKED = re.compile(r"`([^`]+)`")
FILE_LIKE = re.compile(r"^[\w./<>*-]+\.(py|ts|tsx|mjs|js|json|md|yml|yaml)(::[^`]+)?$")


def sdd_findings() -> list[str]:
    if not SDD.is_file():
        return [f"{SDD.relative_to(ROOT).as_posix()} is missing"]
    headings = [ln.lstrip("#").strip() for ln in SDD.read_text(encoding="utf-8").splitlines() if ln.startswith("## ")]
    return [f"SDD.md has no section for {name!r}" for name in SECTIONS if not any(name.lower() in h.lower() for h in headings)]


def test_defined(path: Path, name: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".py":
        return re.search(rf"^\s*def {re.escape(name.split('[')[0])}\(", text, re.M) is not None
    # TypeScript: it('name', ...) or test('name', ...), the name as written (a template literal prefix also counts)
    return re.search(rf"""\b(?:it|test|describe)\(\s*[`'"]{re.escape(name)}""", text) is not None


def gate_findings(rel: str, rid: str, gate: str) -> list[str]:
    out: list[str] = []
    named = [t for t in TICKED.findall(gate) if FILE_LIKE.match(t)]
    if not named:
        return [f"{rel} {rid}: the gate names no file ({gate.strip()[:80]!r})"]
    for target in named:
        path_part, _, test = target.partition("::")
        path = ROOT / path_part
        if not path.is_file():
            out.append(f"{rel} {rid}: names {path_part!r}, which does not exist")
            continue
        if test and path.suffix in {".py", ".ts", ".tsx"} and not test_defined(path, test):
            out.append(f"{rel} {rid}: {path_part} defines no test {test!r}")
    return out


def main() -> int:
    findings = sdd_findings()
    live = planned = 0
    skipped: list[str] = []
    pending: list[str] = []
    for req in sorted(FEATURES.glob("*/requirements.md")):
        rel = req.relative_to(ROOT).as_posix()
        text = req.read_text(encoding="utf-8")
        head = "\n".join(text.splitlines()[:6])
        if re.search(r"^Status:\s*superseded", head, re.M | re.I):
            skipped.append(req.parent.name)
            continue
        is_planned = bool(re.search(r"^Status:\s*planned", head, re.M | re.I))
        rows = [m for m in (ROW.match(line) for line in text.splitlines()) if m]
        if not rows:
            findings.append(f"{rel}: no requirement rows")
        if is_planned:
            pending.append(req.parent.name)
        for r in rows:
            if "SHALL" not in r["statement"]:
                findings.append(f"{rel} {r['id']}: the statement has no SHALL")
            if is_planned:
                planned += 1
                if not any(FILE_LIKE.match(t) for t in TICKED.findall(r["gate"])):
                    findings.append(f"{rel} {r['id']}: the planned gate names no file ({r['gate'].strip()[:80]!r})")
                continue
            live += 1
            if not r["gate"].strip():
                findings.append(f"{rel} {r['id']}: no gate")
                continue
            findings.extend(gate_findings(rel, r["id"], r["gate"]))
    if findings:
        print("SDD CHECK FAILED:")
        for f in findings:
            print(f"  - {f}")
        return 1
    note = f"; superseded, not checked: {', '.join(skipped)}" if skipped else ""
    note += f"; {planned} planned requirements, gates not yet built: {', '.join(pending)}" if pending else ""
    print(f"check_sdd: OK, SDD.md complete, {live} live requirements with real gates{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
