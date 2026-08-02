#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _escape_braces(text: Optional[str]) -> str:
    if not text:
        return ""
    return text.replace("{", "{{").replace("}", "}}")


def build_filled_prompt(prompt: str, low: Optional[str], mid: Optional[str], high: Optional[str]) -> str:
    return prompt.format(
        low_mag_caption=_escape_braces(low),
        mid_mag_caption=_escape_braces(mid),
        high_mag_caption=_escape_braces(high),
    )


def parse_feature_blocks(text: str) -> Dict[str, List[str]]:
    sections = {
        "low_mag_features": [],
        "mid_mag_features": [],
        "high_mag_features": [],
    }
    current: Optional[str] = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        key = stripped.rstrip(":").lower()
        if key in sections:
            current = key
            continue
        if stripped.startswith("-") or stripped.startswith("•"):
            if current:
                item = stripped.lstrip("-•").strip()
                if item:
                    sections[current].append(item)
    return sections


def response_text(response: Any) -> str:
    if hasattr(response, "output_text"):
        return response.output_text
    try:
        parts: List[str] = []
        for output in response.output:
            for content in output.content:
                if getattr(content, "type", None) == "output_text":
                    parts.append(content.text)
        if parts:
            return "\n".join(parts)
    except Exception:
        pass
    return str(response)


def build_entry_paths(entry: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for path_item in entry.get("paths", []):
        yield path_item


def extract_features(
    client: Any,
    model: str,
    prompt: str,
    path_item: Dict[str, Any],
    max_output_tokens: int,
    temperature: float,
    parse_features: bool,
) -> Tuple[Dict[str, Any], str]:
    low_caption = (path_item.get("low_mag") or {}).get("caption")
    mid_caption = (path_item.get("mid_mag") or {}).get("caption")
    high_caption = (path_item.get("high_mag") or {}).get("caption")

    input_text = build_filled_prompt(prompt, low_caption, mid_caption, high_caption)

    response = client.responses.create(
        model=model,
        input=input_text,
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )
    text = response_text(response)
    if parse_features:
        parsed = parse_feature_blocks(text)
    else:
        parsed = {
            "low_mag_features": [],
            "mid_mag_features": [],
            "high_mag_features": [],
        }

    return parsed, text


def open_jsonl_writer(output_path: Path, mode: str = "w"):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path.open(mode, encoding="utf-8")


def write_jsonl_line(handle, record: Dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False))
    handle.write("\n")
    handle.flush()


def parse_args() -> argparse.Namespace:
    default_prompt = Path(__file__).resolve().parents[1] / "prompts" / "visual_features_prompt.txt"
    parser = argparse.ArgumentParser(
        description="Extract visual features from captions using a GPT prompt."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        help="Grouped-by-WSI JSONL output written after processing completes.",
    )
    parser.add_argument(
        "--cache-jsonl",
        type=Path,
        default=None,
        help="Per-path streaming JSONL cache. Default: <output-jsonl>.paths.jsonl",
    )
    parser.add_argument("--prompt-file", type=Path, default=default_prompt)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip entries missing any of low/mid/high captions.",
    )
    parser.add_argument(
        "--include-raw-response",
        action="store_true",
        help="Include raw model response in output.",
    )
    parser.add_argument(
        "--defer-parse",
        action="store_true",
        help="Skip parsing and grouping; only cache raw responses.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing cache by skipping already processed path_id entries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    load_env_file(repo_root / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("Missing dependency: install the openai package.") from exc

    client = OpenAI()
    prompt = load_prompt(args.prompt_file)

    # Pre-scan to get total for progress.
    total = 0
    for entry in read_jsonl(args.input_jsonl):
        total += len(entry.get("paths", []))

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None

    progress = tqdm(total=total) if tqdm else None

    if args.cache_jsonl is None:
        args.cache_jsonl = args.output_jsonl.with_suffix(".paths.jsonl")

    seen: set[tuple[str, str]] = set()
    if args.resume and args.cache_jsonl.exists():
        for record in read_jsonl(args.cache_jsonl):
            wsi_name = record.get("wsi_name")
            path_id = record.get("path_id")
            if wsi_name and path_id:
                seen.add((wsi_name, path_id))

    count = 0
    with open_jsonl_writer(args.cache_jsonl, mode="a") as out_f:
        for entry in read_jsonl(args.input_jsonl):
            wsi_name = entry.get("wsi_name")
            wsi_path = entry.get("wsi_path")
            for path_item in build_entry_paths(entry):
                if args.resume:
                    key = (wsi_name, path_item.get("path_id"))
                    if key in seen:
                        if progress:
                            progress.update(1)
                        continue
                low_caption = (path_item.get("low_mag") or {}).get("caption")
                mid_caption = (path_item.get("mid_mag") or {}).get("caption")
                high_caption = (path_item.get("high_mag") or {}).get("caption")
                if args.skip_missing and (not low_caption or not mid_caption or not high_caption):
                    if progress:
                        progress.update(1)
                    continue

                parsed, raw_text = extract_features(
                    client=client,
                    model=args.model,
                    prompt=prompt,
                    path_item=path_item,
                    max_output_tokens=args.max_output_tokens,
                    temperature=args.temperature,
                    parse_features=False,
                )

                out_record = {
                    "wsi_name": wsi_name,
                    "path_id": path_item.get("path_id"),
                }
                out_record["raw_response"] = raw_text

                write_jsonl_line(out_f, out_record)

                count += 1
                if args.sleep_sec > 0:
                    time.sleep(args.sleep_sec)
                if progress:
                    progress.update(1)
                if args.limit and count >= args.limit:
                    break
            if args.limit and count >= args.limit:
                break

    if progress:
        progress.close()

    if not args.defer_parse:
        # Load feature cache: keyed by (wsi_name, path_id).
        feature_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for record in read_jsonl(args.cache_jsonl):
            key = (record.get("wsi_name"), record.get("path_id"))
            raw_text = record.get("raw_response") or ""
            parsed = parse_feature_blocks(raw_text)
            feature_cache[key] = {
                "low_mag_features": parsed.get("low_mag_features", []),
                "mid_mag_features": parsed.get("mid_mag_features", []),
                "high_mag_features": parsed.get("high_mag_features", []),
                "raw_response": raw_text,
            }

        # Build grouped-by-WSI output by joining features with input JSONL.
        with open_jsonl_writer(args.output_jsonl, mode="w") as grouped_f:
            for entry in read_jsonl(args.input_jsonl):
                wsi_name = entry.get("wsi_name")
                wsi_path = entry.get("wsi_path")
                out_entry = {
                    "wsi_name": wsi_name,
                    "wsi_path": wsi_path,
                    "paths": [],
                }
                for path_item in build_entry_paths(entry):
                    key = (wsi_name, path_item.get("path_id"))
                    feat = feature_cache.get(
                        key,
                        {
                            "low_mag_features": [],
                            "mid_mag_features": [],
                            "high_mag_features": [],
                            "raw_response": None,
                        },
                    )
                    low_mag = dict(path_item.get("low_mag") or {})
                    mid_mag = dict(path_item.get("mid_mag") or {})
                    high_mag = dict(path_item.get("high_mag") or {})
                    low_mag["feature"] = feat.get("low_mag_features", [])
                    mid_mag["feature"] = feat.get("mid_mag_features", [])
                    high_mag["feature"] = feat.get("high_mag_features", [])

                    out_path = {
                        "path_id": path_item.get("path_id"),
                        "path_description": path_item.get("path_description"),
                        "path_summary": path_item.get("path_summary"),
                        "low_mag": low_mag,
                        "mid_mag": mid_mag,
                        "high_mag": high_mag,
                    }
                    if args.include_raw_response and feat.get("raw_response") is not None:
                        out_path["raw_response"] = feat.get("raw_response")
                    out_entry["paths"].append(out_path)

                write_jsonl_line(grouped_f, out_entry)


if __name__ == "__main__":
    main()
