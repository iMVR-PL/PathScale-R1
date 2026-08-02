from __future__ import annotations

import argparse
import json
import os
import sys
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


def open_jsonl_writer(output_path: Path, mode: str = "w"):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path.open(mode, encoding="utf-8")


def write_jsonl_line(handle, record: Dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False))
    handle.write("\n")
    handle.flush()


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


def format_feature_list(features: Optional[List[str]]) -> str:
    if not features:
        return ""
    return "\n".join(f"- {item}" for item in features)


def response_text(response: Any) -> str:
    if hasattr(response, "output_text"):
        return response.output_text
    try:
        parts: List[str] = []
        for output in response.output:
            for content in output.content:
                content_type = getattr(content, "type", None)
                text_val = getattr(content, "text", None)
                if text_val:
                    parts.append(text_val)
                elif content_type == "output_text":
                    parts.append(content.text)
                elif content_type == "refusal":
                    parts.append(getattr(content, "refusal", "") or "")
        if parts:
            return "\n".join(parts)
    except Exception:
        pass
    try:
        return response.model_dump_json()
    except Exception:
        return str(response)


def create_response(client: Any, model: str, input_text: str, max_tokens: int, temperature: float) -> Any:
    try:
        return client.responses.create(
            model=model,
            input=input_text,
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
    except Exception as exc:
        message = str(exc)
        if "temperature" in message and "not supported" in message:
            return client.responses.create(
                model=model,
                input=input_text,
                max_output_tokens=max_tokens,
            )
        raise


def parse_vqa_response(text: str) -> List[Dict[str, Any]]:
    questions: List[Dict[str, Any]] = []
    current: Dict[str, Any] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if "|" in stripped and stripped[0].upper() == "Q":
            parts = stripped.split("|", 1)
            if parts[0].upper().startswith("Q"):
                if current:
                    questions.append(current)
                current = {"question": parts[1].strip(), "options": {}}
                continue
        if stripped.lower().startswith("q") and ":" in stripped:
            if current:
                questions.append(current)
            current = {"question": stripped.split(":", 1)[1].strip(), "options": {}}
            continue

        if stripped.upper().startswith("ANSWER|"):
            current["answer"] = stripped.split("|", 1)[1].strip()
            continue
        if stripped.upper().startswith("RATIONALE|"):
            current["rationale"] = stripped.split("|", 1)[1].strip()
            continue

        if "|" in stripped and stripped[:1] in {"A", "B", "C", "D"}:
            label, option_text = stripped.split("|", 1)
            if len(label) == 1:
                current.setdefault("options", {})[label] = option_text.strip()
                continue
        if stripped.startswith("A.") or stripped.startswith("B.") or stripped.startswith("C.") or stripped.startswith("D."):
            label = stripped[0]
            option_text = stripped[2:].strip()
            current.setdefault("options", {})[label] = option_text
            continue

        if stripped.lower().startswith("answer"):
            current["answer"] = stripped.split(":", 1)[-1].strip()
            continue
        if stripped.lower().startswith("rationale"):
            current["rationale"] = stripped.split(":", 1)[-1].strip()
            continue

        if "question" not in current:
            current["question"] = stripped
        elif "rationale" not in current:
            current["rationale"] = stripped

    if current:
        questions.append(current)
    if len(questions) > 1:
        return questions[:1]
    return questions


def parse_args() -> argparse.Namespace:
    default_prompt_dir = Path(__file__).resolve().parents[1] / "prompts" / "vqa"
    parser = argparse.ArgumentParser(description="Generate VQA from visual features.")
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--assignment-json", type=Path, required=True)
    parser.add_argument(
        "--cache-jsonl",
        type=Path,
        default=None,
        help="Per-path streaming JSONL cache. Default: <output-jsonl>.paths.jsonl",
    )
    parser.add_argument("--prompt-dir", type=Path, default=default_prompt_dir)
    parser.add_argument(
        "--types",
        type=str,
        default="A,B,C,D,E",
        help="Comma-separated type list (e.g., A,B).",
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max-output-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sleep-sec", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0, help="0 means no limit.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing cache by skipping already processed (wsi_name, path_id, type).",
    )
    parser.add_argument(
        "--defer-parse",
        action="store_true",
        help="Only cache raw responses; skip parsing/grouping.",
    )
    parser.add_argument(
        "--include-raw-response",
        action="store_true",
        help="Include raw model response in grouped output.",
    )
    parser.add_argument(
        "--split-by-type",
        action="store_true",
        help="Write separate cache and grouped outputs per VQA type.",
    )
    return parser.parse_args()


def load_assignments(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("assignments", {})


def get_assignment(assignments: Dict[str, Any], wsi_name: str, path_id: str, q_type: str) -> Dict[str, Any]:
    key = f"{wsi_name}::{path_id}"
    if key not in assignments:
        raise RuntimeError(f"Missing assignments for {key}")
    return assignments[key]["types"][q_type]


def load_fixed_templates(path: Path) -> Dict[str, Dict[str, str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_pool_instruction(pool: str, fixed_templates: Dict[str, str], fixed_template: Optional[str]) -> str:
    if pool == "fixed":
        if not fixed_template:
            raise RuntimeError("Missing fixed template key for fixed pool.")
        template_text = fixed_templates[fixed_template]
        return "\n".join(
            [
                "OVERRIDE FOR THIS CALL:",
                "Generate exactly ONE question.",
                f"Use FIXED TEMPLATE {fixed_template} verbatim:",
                f"\"{template_text}\"",
                "Do NOT generate any free questions.",
            ]
        )
    q1 = fixed_templates["Q1"]
    q2 = fixed_templates["Q2"]
    return "\n".join(
        [
            "OVERRIDE FOR THIS CALL:",
            "Generate exactly ONE free question.",
            "Do NOT use or paraphrase these fixed templates:",
            f"Q1 \"{q1}\"",
            f"Q2 \"{q2}\"",
        ]
    )


def build_prompt(
    base_text: str,
    type_constraints: str,
    pool_instruction: str,
) -> str:
    type_block = type_constraints.splitlines()[0]
    return base_text.format(
        TYPE_BLOCK=type_block,
        TYPE_CONSTRAINTS=type_constraints,
        POOL_INSTRUCTION=pool_instruction,
        low_mag_features="{low_mag_features}",
        mid_mag_features="{mid_mag_features}",
        high_mag_features="{high_mag_features}",
        path_description="{path_description}",
        path_summary="{path_summary}",
    )


def fill_prompt(
    prompt: str,
    low: List[str],
    mid: List[str],
    high: List[str],
    path_description: Optional[str],
    path_summary: Optional[str],
) -> str:
    return prompt.format(
        low_mag_features=_escape_braces(format_feature_list(low)),
        mid_mag_features=_escape_braces(format_feature_list(mid)),
        high_mag_features=_escape_braces(format_feature_list(high)),
        path_description=_escape_braces(path_description or ""),
        path_summary=_escape_braces(path_summary or ""),
    )


def build_output_path_item(path_item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "path_id": path_item.get("path_id"),
        "low_mag": {"image_path": (path_item.get("low_mag") or {}).get("image_path")},
        "mid_mag": {"image_path": (path_item.get("mid_mag") or {}).get("image_path")},
        "high_mag": {"image_path": (path_item.get("high_mag") or {}).get("image_path")},
    }


def build_output_for_type(
    input_jsonl: Path,
    cache_path: Path,
    output_path: Path,
    q_type: str,
    include_raw: bool,
) -> None:
    cache_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for record in read_jsonl(cache_path):
        key = (record.get("wsi_name"), record.get("path_id"))
        parsed = parse_vqa_response(record.get("raw_response") or "")
        cache_map[key] = {
            "questions": parsed,
            "raw_response": record.get("raw_response"),
        }

    with open_jsonl_writer(output_path, mode="w") as out_f:
        for entry in read_jsonl(input_jsonl):
            wsi_name = entry.get("wsi_name")
            out_entry = {
                "wsi_name": wsi_name,
                "wsi_path": entry.get("wsi_path"),
                "paths": [],
            }
            for path_item in entry.get("paths", []):
                path_id = path_item.get("path_id")
                data = cache_map.get((wsi_name, path_id), {"questions": [], "raw_response": None})
                payload: Any = data["questions"]
                if include_raw and data["raw_response"] is not None:
                    payload = {"questions": payload, "raw_response": data["raw_response"]}
                out_path = build_output_path_item(path_item)
                out_path["vqa"] = {f"type_{q_type}": payload}
                out_entry["paths"].append(out_path)
            write_jsonl_line(out_f, out_entry)


def build_output_all_types(
    input_jsonl: Path,
    cache_path: Path,
    output_path: Path,
    types: List[str],
    include_raw: bool,
) -> None:
    cache_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for record in read_jsonl(cache_path):
        key = (record.get("wsi_name"), record.get("path_id"), record.get("type"))
        parsed = parse_vqa_response(record.get("raw_response") or "")
        cache_map[key] = {
            "questions": parsed,
            "raw_response": record.get("raw_response"),
        }

    with open_jsonl_writer(output_path, mode="w") as out_f:
        for entry in read_jsonl(input_jsonl):
            wsi_name = entry.get("wsi_name")
            out_entry = {
                "wsi_name": wsi_name,
                "wsi_path": entry.get("wsi_path"),
                "paths": [],
            }
            for path_item in entry.get("paths", []):
                path_id = path_item.get("path_id")
                vqa_by_type = {}
                for q_type in types:
                    data = cache_map.get((wsi_name, path_id, q_type), {"questions": [], "raw_response": None})
                    payload: Any = data["questions"]
                    if include_raw and data["raw_response"] is not None:
                        payload = {"questions": payload, "raw_response": data["raw_response"]}
                    vqa_by_type[f"type_{q_type}"] = payload

                out_path = build_output_path_item(path_item)
                out_path["vqa"] = vqa_by_type
                out_entry["paths"].append(out_path)
            write_jsonl_line(out_f, out_entry)


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

    if args.cache_jsonl is None:
        args.cache_jsonl = args.output_jsonl.with_suffix(".paths.jsonl")

    types = [t.strip().upper() for t in args.types.split(",") if t.strip()]
    assignments = load_assignments(args.assignment_json)

    prompt_dir = args.prompt_dir
    base_features_free = (prompt_dir / "base_features_free.txt").read_text(encoding="utf-8")
    base_features_fixed = (prompt_dir / "base_features_fixed.txt").read_text(encoding="utf-8")
    base_desc_free = (prompt_dir / "base_description_free.txt").read_text(encoding="utf-8")
    base_desc_fixed = (prompt_dir / "base_description_fixed.txt").read_text(encoding="utf-8")
    fixed_templates = load_fixed_templates(prompt_dir / "fixed_templates.json")

    type_constraints = {
        t: (prompt_dir / f"type_{t.lower()}_constraints.txt").read_text(encoding="utf-8")
        for t in types
    }

    total = 0
    for entry in read_jsonl(args.input_jsonl):
        total += len(entry.get("paths", [])) * len(types)

    try:
        from tqdm import tqdm
    except Exception:
        tqdm = None
    progress = tqdm(total=total) if tqdm else None

    seen: set[tuple[str, str, str]] = set()
    if args.resume:
        if args.split_by_type:
            for q_type in types:
                cache_path = args.cache_jsonl.with_suffix(f".{q_type.lower()}.paths.jsonl")
                if not cache_path.exists():
                    continue
                for record in read_jsonl(cache_path):
                    wsi_name = record.get("wsi_name")
                    path_id = record.get("path_id")
                    if wsi_name and path_id:
                        seen.add((wsi_name, path_id, q_type))
        else:
            if args.cache_jsonl.exists():
                for record in read_jsonl(args.cache_jsonl):
                    wsi_name = record.get("wsi_name")
                    path_id = record.get("path_id")
                    q_type = record.get("type")
                    if wsi_name and path_id and q_type:
                        seen.add((wsi_name, path_id, q_type))

    client = OpenAI()

    if args.split_by_type:
        cache_handles = {
            q_type: open_jsonl_writer(
                args.cache_jsonl.with_suffix(f".{q_type.lower()}.paths.jsonl"), mode="a"
            )
            for q_type in types
        }
    else:
        cache_handles = {"__all__": open_jsonl_writer(args.cache_jsonl, mode="a")}

    count = 0
    try:
        for entry in read_jsonl(args.input_jsonl):
            wsi_name = entry.get("wsi_name")
            for path_item in entry.get("paths", []):
                path_id = path_item.get("path_id")
                low = (path_item.get("low_mag") or {}).get("feature") or []
                mid = (path_item.get("mid_mag") or {}).get("feature") or []
                high = (path_item.get("high_mag") or {}).get("feature") or []
                path_description = path_item.get("path_description")
                path_summary = path_item.get("path_summary")
                for q_type in types:
                    if args.resume and (wsi_name, path_id, q_type) in seen:
                        if progress:
                            progress.update(1)
                        continue

                    assign = get_assignment(assignments, wsi_name, path_id, q_type)
                    pool = assign["pool"]
                    fixed_template = assign.get("template")
                    pool_instruction = build_pool_instruction(pool, fixed_templates[q_type], fixed_template)

                    if q_type == "E":
                        base_text = base_desc_fixed if pool == "fixed" else base_desc_free
                    else:
                        base_text = base_features_fixed if pool == "fixed" else base_features_free

                    prompt = build_prompt(base_text, type_constraints[q_type], pool_instruction)
                    input_text = fill_prompt(prompt, low, mid, high, path_description, path_summary)

                    response = create_response(
                        client=client,
                        model=args.model,
                        input_text=input_text,
                        max_tokens=args.max_output_tokens,
                        temperature=args.temperature,
                    )
                    raw_text = response_text(response)

                    record = {
                        "wsi_name": wsi_name,
                        "path_id": path_id,
                        "type": q_type,
                        "raw_response": raw_text,
                        "pool": pool,
                        "fixed_template": fixed_template,
                    }
                    if args.split_by_type:
                        record.pop("type", None)
                        write_jsonl_line(cache_handles[q_type], record)
                    else:
                        write_jsonl_line(cache_handles["__all__"], record)

                    count += 1
                    if args.sleep_sec > 0:
                        time.sleep(args.sleep_sec)
                    if progress:
                        progress.update(1)
                    if args.limit and count >= args.limit:
                        break
                if args.limit and count >= args.limit:
                    break
            if args.limit and count >= args.limit:
                break
    finally:
        if progress:
            progress.close()
        for handle in cache_handles.values():
            handle.close()

    if args.defer_parse:
        return

    if args.split_by_type:
        for q_type in types:
            cache_path = args.cache_jsonl.with_suffix(f".{q_type.lower()}.paths.jsonl")
            output_path = args.output_jsonl.with_suffix(f".{q_type.lower()}.jsonl")
            build_output_for_type(args.input_jsonl, cache_path, output_path, q_type, args.include_raw_response)
    else:
        build_output_all_types(
            args.input_jsonl,
            args.cache_jsonl,
            args.output_jsonl,
            types,
            args.include_raw_response,
        )


if __name__ == "__main__":
    main()
