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


def default_output_path(input_path: Path) -> Path:
    return Path("processed_data/split") / f"{input_path.stem}.split.csv"


def split_wsi(records, ratios, seed):
    rng = random.Random(seed)
    items = []
    for record in records:
        wsi_name = record.get("wsi_name")
        if not wsi_name:
            raise ValueError("Missing wsi_name in record")
        path_ids = record.get("paths", [])
        path_count = len(path_ids) if path_ids else 0
        items.append(
            {
                "wsi_name": wsi_name,
                "path_count": path_count,
            }
        )

    total_paths = sum(item["path_count"] for item in items)
    if total_paths == 0:
        # Fall back to WSI-count-based split if no paths exist at all.
        wsi_names = [item["wsi_name"] for item in items]
        rng.shuffle(wsi_names)
        total = len(wsi_names)
        train_count = int(total * ratios[0])
        val_count = int(total * ratios[1])
        splits = {}
        for name in wsi_names[:train_count]:
            splits[name] = "train"
        for name in wsi_names[train_count : train_count + val_count]:
            splits[name] = "val"
        for name in wsi_names[train_count + val_count :]:
            splits[name] = "test"
        return splits

    # Shuffle before sorting so equal path counts are randomized deterministically.
    rng.shuffle(items)
    items.sort(key=lambda x: x["path_count"], reverse=True)

    current_paths = {"train": 0, "val": 0, "test": 0}
    current_wsis = {"train": 0, "val": 0, "test": 0}
    splits = {}
    split_order = ("train", "val", "test")

    for item in items:
        path_count = item["path_count"]
        # Assign to split that best matches target ratios after adding this WSI.
        best_split = None
        best_score = None
        for split_name in split_order:
            projected = dict(current_paths)
            projected[split_name] += path_count
            total = sum(projected.values())
            if total == 0:
                ratio_score = 0.0
            else:
                ratio_score = sum(
                    abs(projected[name] / total - ratios[idx])
                    for idx, name in enumerate(split_order)
                )
            # Tie-break by fewer WSIs, then fewer paths in that split.
            score = (
                ratio_score,
                current_wsis[split_name],
                current_paths[split_name],
            )
            if best_score is None or score < best_score:
                best_score = score
                best_split = split_name

        splits[item["wsi_name"]] = best_split
        current_paths[best_split] += path_count
        current_wsis[best_split] += 1

    return splits


def write_csv(records, splits, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wsi_name", "path_id", "split"])
        for record in records:
            wsi_name = record.get("wsi_name")
            if not wsi_name:
                raise ValueError("Missing wsi_name in record")
            path_ids = record.get("paths", [])
            writer.writerow([wsi_name, json.dumps(path_ids, ensure_ascii=False), splits[wsi_name]])


def main():
    parser = argparse.ArgumentParser(
        description="Split WSI list into train/val/test and output CSV."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("processed_data/split/triplet_merged.paths.jsonl"),
        help="Input jsonl file produced by 03a_extract_wsi_paths.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to processed_data/split/<input_stem>.split.csv",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling.",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=(0.7, 0.1, 0.2),
        help="Split ratios for train/val/test. Must sum to 1.0",
    )
    args = parser.parse_args()

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {args.ratios}")

    records = list(load_jsonl(args.input))
    splits = split_wsi(records, args.ratios, args.seed)
    output_path = args.output or default_output_path(args.input)
    write_csv(records, splits, output_path)


if __name__ == "__main__":
    main()
