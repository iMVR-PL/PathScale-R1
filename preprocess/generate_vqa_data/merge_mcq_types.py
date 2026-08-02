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


def merge_records(input_dir: Path):
    records_by_wsi = {}

    jsonl_files = sorted(input_dir.glob("vqa_grouped.*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No vqa_grouped.*.jsonl files found in {input_dir}")

    for file_path in jsonl_files:
        for record in load_jsonl(file_path):
            wsi_name = record.get("wsi_name")
            if not wsi_name:
                raise ValueError(f"Missing wsi_name in {file_path}")

            if wsi_name not in records_by_wsi:
                base_record = {
                    "wsi_name": record.get("wsi_name"),
                    "wsi_path": record.get("wsi_path"),
                    "paths": [],
                }
                records_by_wsi[wsi_name] = {
                    "record": base_record,
                    "paths_by_id": {},
                }

            entry = records_by_wsi[wsi_name]
            for path_item in record.get("paths", []):
                path_id = path_item.get("path_id")
                if not path_id:
                    raise ValueError(f"Missing path_id in {file_path} for wsi {wsi_name}")

                if path_id not in entry["paths_by_id"]:
                    merged_path = {
                        "path_id": path_id,
                        "low_mag": path_item.get("low_mag"),
                        "mid_mag": path_item.get("mid_mag"),
                        "high_mag": path_item.get("high_mag"),
                        "vqa": {},
                    }
                    entry["paths_by_id"][path_id] = merged_path
                    entry["record"]["paths"].append(merged_path)

                merged_path = entry["paths_by_id"][path_id]
                vqa = path_item.get("vqa", {})
                for vqa_type, items in vqa.items():
                    if vqa_type not in merged_path["vqa"]:
                        merged_path["vqa"][vqa_type] = []
                    if isinstance(items, list):
                        merged_path["vqa"][vqa_type].extend(items)
                    else:
                        merged_path["vqa"][vqa_type].append(items)

    return [entry["record"] for entry in records_by_wsi.values()]


def write_jsonl(records, output_path: Path):
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Merge vqa_grouped.*.jsonl files by path_id into a single jsonl."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("processed_data/generated"),
        help="Directory containing vqa_grouped.*.jsonl files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("processed_data/generated/vqa_grouped.merged.jsonl"),
        help="Output merged jsonl path.",
    )
    args = parser.parse_args()

    records = merge_records(args.input_dir)
    write_jsonl(records, args.output)


if __name__ == "__main__":
    main()
