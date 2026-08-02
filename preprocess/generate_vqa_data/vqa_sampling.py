#!/usr/bin/env python3
"""
Generate per-path VQA pool assignments with balance and diversity.

Output JSON structure:
{
  "assignments": {
    "<wsi>::<path>": {
      "wsi_name": "...",
      "path_id": "...",
      "types": {
        "A": {"pool": "fixed", "template": "Q1"},
        "B": {"pool": "free", "template": null},
        ...
      }
    }
  },
  "stats": {
    "A": {"fixed": 10, "free": 12, "Q1": 6, "Q2": 4},
    ...
  }
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
import ast
from random import Random
import sys

_THIS_DIR = Path(__file__).parent
if str(_THIS_DIR) not in sys.path:
    sys.path.append(str(_THIS_DIR))




def _to_state_key(wsi_name: str, path_id: str) -> str:
    return f"{wsi_name}::{path_id}"


def load_state(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def init_state(state: Dict, types: List[str], seed: int) -> Dict:
    state.setdefault("seed", seed)
    state.setdefault("per_type_counts", {})
    state.setdefault("template_counts", {})
    state.setdefault("path_assignments", {})
    for t in types:
        state["per_type_counts"].setdefault(t, {"fixed": 0, "free": 0})
        state["template_counts"].setdefault(t, {"Q1": 0, "Q2": 0})
    return state


def init_rng(state: Dict, seed: int) -> Random:
    rng = Random(seed)
    if "rng_state" in state:
        try:
            rng.setstate(ast.literal_eval(state["rng_state"]))
        except Exception:
            pass
    return rng


def record_rng_state(state: Dict, rng: Random) -> None:
    state["rng_state"] = repr(rng.getstate())


def choose_fixed_count(rng: Random) -> int:
    # For 5 types, target 50/50 by choosing 2 or 3 fixed per path.
    return rng.choice([2, 3])


def choose_pool_for_path(
    state: Dict, types: List[str], wsi_name: str, path_id: str, rng: Random
) -> Dict[str, str]:
    key = _to_state_key(wsi_name, path_id)
    if key in state["path_assignments"]:
        return state["path_assignments"][key]

    k_fixed = choose_fixed_count(rng)
    per_type = state["per_type_counts"]
    # Score types by need for fixed (higher means more need for fixed).
    scored: List[Tuple[int, str]] = []
    for t in types:
        need = per_type[t]["free"] - per_type[t]["fixed"]
        scored.append((need, t))
    rng.shuffle(scored)
    scored.sort(reverse=True)

    fixed_types = {t for _, t in scored[:k_fixed]}
    assignments = {t: ("fixed" if t in fixed_types else "free") for t in types}
    state["path_assignments"][key] = assignments
    return assignments


def choose_fixed_template(state: Dict, q_type: str, rng: Random) -> str:
    counts = state["template_counts"][q_type]
    if counts["Q1"] < counts["Q2"]:
        choice = "Q1"
    elif counts["Q2"] < counts["Q1"]:
        choice = "Q2"
    else:
        choice = rng.choice(["Q1", "Q2"])
    counts[choice] += 1
    return choice


def bump_pool_count(state: Dict, q_type: str, pool: str) -> None:
    state["per_type_counts"][q_type][pool] += 1


def update_state_from_cache_record(
    state: Dict,
    q_type: str,
    wsi_name: str,
    path_id: str,
    pool: str,
    fixed_template: str | None,
) -> None:
    key = _to_state_key(wsi_name, path_id)
    state["path_assignments"].setdefault(key, {})[q_type] = pool
    if pool in {"fixed", "free"}:
        state["per_type_counts"][q_type][pool] += 1
    if fixed_template in {"Q1", "Q2"}:
        state["template_counts"][q_type][fixed_template] += 1


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample VQA pools per path.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--types", type=str, default="A,B,C,D,E")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--state-json",
        type=Path,
        default=None,
        help="State file for resume. Default: <output-json>.state.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.state_json is None:
        args.state_json = args.output_json.with_suffix(".state.json")

    types = [t.strip().upper() for t in args.types.split(",") if t.strip()]
    state = init_state(load_state(args.state_json), types, args.seed)
    rng = init_rng(state, args.seed)

    assignments: Dict[str, Any] = {}
    input_records = read_jsonl(args.input_jsonl)
    for entry in input_records:
        wsi_name = entry.get("wsi_name")
        for path_item in entry.get("paths", []):
            path_id = path_item.get("path_id")
            pool_by_type = choose_pool_for_path(state, types, wsi_name, path_id, rng)
            types_out: Dict[str, Any] = {}
            for q_type in types:
                pool = pool_by_type[q_type]
                template = choose_fixed_template(state, q_type, rng) if pool == "fixed" else None
                bump_pool_count(state, q_type, pool)
                types_out[q_type] = {"pool": pool, "template": template}
            key = f"{wsi_name}::{path_id}"
            assignments[key] = {
                "wsi_name": wsi_name,
                "path_id": path_id,
                "types": types_out,
            }
            record_rng_state(state, rng)

    stats: Dict[str, Any] = {}
    for t in types:
        counts = state["per_type_counts"][t]
        templates = state["template_counts"][t]
        stats[t] = {
            "fixed": counts["fixed"],
            "free": counts["free"],
            "Q1": templates["Q1"],
            "Q2": templates["Q2"],
        }

    output = {"assignments": assignments, "stats": stats}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    save_state(args.state_json, state)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
