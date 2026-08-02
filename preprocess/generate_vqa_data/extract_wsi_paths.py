"""Extract wsi_name, wsi_path, and path_id list from a jsonl file.

Input: --input jsonl (e.g., processed_data/input/triplet_merged.jsonl)
Output: jsonl with {"wsi_name", "wsi_path", "paths": [path_id, ...]}
Default output: processed_data/split/<input_stem>.paths.jsonl
"""

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def extract_wsi_paths(input_path: Path):
    records = []
    for record in load_jsonl(input_path):
        wsi_name = record.get("wsi_name")
        wsi_path = record.get("wsi_path")
        if not wsi_name:
            raise ValueError(f"Missing wsi_name in {input_path}")
        if not wsi_path:
            raise ValueError(f"Missing wsi_path in {input_path} for wsi {wsi_name}")

        path_ids = []
        for path_item in record.get("paths", []):
            if isinstance(path_item, dict):
                path_id = path_item.get("path_id")
            else:
                path_id = path_item
            if not path_id:
                raise ValueError(f"Missing path_id in {input_path} for wsi {wsi_name}")
            path_ids.append(path_id)

        records.append(
            {
                "wsi_name": wsi_name,
                "wsi_path": wsi_path,
                "paths": path_ids,
            }
        )
    return records


def write_jsonl(records, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def default_output_path(input_path: Path) -> Path:
    return Path("processed_data/split") / f"{input_path.stem}.paths.jsonl"


def main():
    parser = argparse.ArgumentParser(
        description="Extract wsi_name, wsi_path, and path_id list from a jsonl."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("processed_data/input/triplet_merged.jsonl"),
        help="Input jsonl file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output jsonl path. Defaults to processed_data/split/<input_stem>.paths.jsonl",
    )
    args = parser.parse_args()

    output_path = args.output or default_output_path(args.input)
    records = extract_wsi_paths(args.input)
    write_jsonl(records, output_path)


if __name__ == "__main__":
    main()
