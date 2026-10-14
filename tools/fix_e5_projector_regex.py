#!/usr/bin/env python3
"""Repair a malformed projector regex in a *local, ignored* E5 kernel generator.

This tool neither runs the generator nor submits a Kaggle job.

Usage:
  python tools/fix_e5_projector_regex.py work/create_e5_kernel.py
  python tools/fix_e5_projector_regex.py --validate-notebook work/kaggle_e5/lab_day10_e5.ipynb

For safety, the source edit only changes a unique, complete assignment line.
If a generator embeds the assignment in some other encoding or expression, it
fails instead of modifying unrelated code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ASSIGNMENT = r"_PROJECTOR_RE = re.compile(r'(?:^|\.)visual\.(?:merger|deepstack_merger_list)(?:\.|$)')"
LINE = re.compile(
    r"^(?P<indent>[ \t]*)_PROJECTOR_RE[ \t]*=[ \t]*re\.compile\([^\r\n]*\)[ \t]*$",
    re.MULTILINE,
)


def patch_generator(path: Path) -> bool:
    if not path.is_file():
        raise FileNotFoundError(f"Local E5 generator not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        source = f.read()
    matches = list(LINE.finditer(source))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one complete _PROJECTOR_RE assignment in {path}; "
            f"found {len(matches)}. No changes made. Inspect generator source manually."
        )
    found = matches[0]
    if "visual" not in found.group() or "deepstack_merger_list" not in found.group():
        raise ValueError("The matched assignment is not the expected vision projector regex")
    replacement = found.group("indent") + ASSIGNMENT
    if found.group() == replacement:
        print("Already patched:", path)
        return False
    updated = source[: found.start()] + replacement + source[found.end() :]
    # Check that rewriting the source generator itself introduces no syntax error.
    compile(updated, str(path), "exec")
    backup = path.with_name(path.name + ".bak")
    if backup.exists():
        raise FileExistsError(f"Backup exists: {backup}; refusing to overwrite it")
    with backup.open("w", encoding="utf-8", newline="") as f:
        f.write(source)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(updated)
    print("Patched:", path)
    print("Original backup:", backup)
    return True


def validate_notebook(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Generated notebook not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("nbformat") != 4 or not isinstance(data.get("cells"), list):
        raise ValueError("Not a notebook JSON document with nbformat=4")

    # IPython's transformer understands notebook magics such as !pip and %cd.
    try:
        from IPython.core.inputtransformer2 import TransformerManager
    except ImportError as exc:
        raise RuntimeError(
            "Install IPython locally first: python -m pip install ipython"
        ) from exc

    transformer = TransformerManager()
    num_code = 0
    for index, cell in enumerate(data["cells"], start=1):
        if cell.get("cell_type") != "code":
            continue
        raw = cell.get("source", "")
        source = "".join(raw) if isinstance(raw, list) else raw
        if not source.strip():
            continue
        num_code += 1
        # Parsing/compiling does NOT execute GPU code, dependencies or Kaggle CLI.
        transformed = transformer.transform_cell(source)
        try:
            compile(transformed, f"{path}::cell_{index}", "exec")
        except SyntaxError as exc:
            raise SyntaxError(
                f"Notebook cell {index} failed compilation: {exc.msg} "
                f"at line {exc.lineno}; source: {exc.text!r}"
            ) from exc
    print(f"PASS: compiled {num_code} code cells in {path} (no execution)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("generator", nargs="?", type=Path)
    group.add_argument("--validate-notebook", metavar="PATH", type=Path)
    args = parser.parse_args()
    try:
        if args.validate_notebook:
            validate_notebook(args.validate_notebook)
        else:
            patch_generator(args.generator)
    except (OSError, ValueError, SyntaxError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
