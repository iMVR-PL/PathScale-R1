"""Finalize MCQ data into train/val/test JSON from merged VQA and WSI split CSV.

Input:
  1) --vqa-jsonl: processed_data/merged_and_balanced/vqa_grouped.merged.balanced.jsonl
  2) --split-csv: processed_data/split/wsi_split.csv
Output:
  - all_vqa.json: JSON with keys train/val/test, each a list of MCQ items.
  - train.json / val.json / test.json: each contains the list for that split.
"""

import argparse
import csv
import json
import random
from pathlib import Path


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_split_csv(path: Path):
    mapping = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wsi_name = row.get("wsi_name")
            if not wsi_name:
                raise ValueError(f"Missing wsi_name in split CSV {path}")
            path_ids = json.loads(row.get("path_id", "[]"))
            split = row.get("split")
            if split not in {"train", "val", "test"}:
                raise ValueError(f"Invalid split '{split}' for wsi {wsi_name}")
            mapping[wsi_name] = {"split": split, "path_ids": set(path_ids)}
    return mapping


def get_image_path(mag_entry):
    if isinstance(mag_entry, dict):
        return mag_entry.get("image_path")
    return None


def prefix_path(path_value: str, data_root: Path | None) -> str:
    if not path_value or data_root is None:
        return path_value
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return path_value
    return str(data_root / path_obj)


def finalize_records(vqa_jsonl: Path, split_csv: Path, seed: int, data_root: Path | None):
    split_map = load_split_csv(split_csv)
    output = {"train": [], "val": [], "test": []}

    for record in load_jsonl(vqa_jsonl):
        wsi_name = record.get("wsi_name")
        if not wsi_name or wsi_name not in split_map:
            continue
        split_info = split_map[wsi_name]

        for path_item in record.get("paths", []):
            path_id = path_item.get("path_id")
            if not path_id or path_id not in split_info["path_ids"]:
                continue

            image_path = {
                "low_mag": prefix_path(get_image_path(path_item.get("low_mag")), data_root),
                "mid_mag": prefix_path(get_image_path(path_item.get("mid_mag")), data_root),
                "high_mag": prefix_path(get_image_path(path_item.get("high_mag")), data_root),
            }

            vqa = path_item.get("vqa", {})
            for question_type, items in vqa.items():
                if not isinstance(items, list):
                    items = [items]
                for qa in items:
                    split = split_info["split"]
                    output[split].append(
                        {
                            "question": qa.get("question"),
                            "options": qa.get("options"),
                            "answer": qa.get("answer"),
                            "rationale": qa.get("rationale"),
                            "image_path": image_path,
                            "extra_info": {
                                "wsi_name": wsi_name,
                                "path_id": path_id,
                                "question_type": question_type,
                            },
                        }
                    )

    rng = random.Random(seed)
    for split_name, items in output.items():
        rng.shuffle(items)
        for idx, item in enumerate(items, start=1):
            item["No"] = idx

    return output


def default_output_path() -> Path:
    return Path("processed_data/finalized") / "all_vqa.json"


def main():
    parser = argparse.ArgumentParser(
        description="Finalize MCQ data into train/val/test JSON."
    )
    parser.add_argument(
        "--vqa-jsonl",
        type=Path,
        default=Path("processed_data/merged_and_balanced/vqa_grouped.merged.balanced.jsonl"),
        help="Merged and balanced VQA jsonl.",
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=Path("processed_data/split/wsi_split.csv"),
        help="WSI split CSV produced by 03b_wsi_split.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to processed_data/finalized/all_vqa.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to shuffle questions within each split.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional root to prefix relative image paths.",
    )
    args = parser.parse_args()

    output_path = args.output or default_output_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    finalized = finalize_records(args.vqa_jsonl, args.split_csv, args.seed, args.data_root)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(finalized, f, ensure_ascii=False, indent=2)
    for split_name in ("train", "val", "test"):
        split_path = output_path.parent / f"{split_name}.json"
        with split_path.open("w", encoding="utf-8") as f:
            json.dump(finalized.get(split_name, []), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
