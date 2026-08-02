import argparse
import json
import random
from collections import Counter, defaultdict
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

LETTERS = ("A", "B", "C", "D")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def iter_mcq_items(
    records: Iterable[Dict[str, Any]],
) -> Iterable[Tuple[Dict[str, Any], str]]:
    for record in records:
        for path_item in record.get("paths", []):
            vqa = path_item.get("vqa", {})
            for vqa_type, items in vqa.items():
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            yield item, vqa_type
                elif isinstance(items, dict):
                    yield items, vqa_type


def count_answers(items: Iterable[Tuple[Dict[str, Any], str]]) -> Counter:
    counts = Counter()
    for item, _ in items:
        answer = item.get("answer")
        if answer in LETTERS:
            counts[answer] += 1
    for letter in LETTERS:
        counts.setdefault(letter, 0)
    return counts


def compute_target_counts(total: int, current: Counter) -> Dict[str, int]:
    base = total // len(LETTERS)
    remainder = total % len(LETTERS)
    target = {letter: base for letter in LETTERS}
    # Give remainder to currently least frequent letters to reduce changes.
    for letter, _ in sorted(current.items(), key=lambda x: (x[1], x[0]))[:remainder]:
        target[letter] += 1
    return target


def swap_answer(options: Dict[str, Any], current: str, target: str) -> None:
    options[current], options[target] = options[target], options[current]


def balance_answers(
    records: List[Dict[str, Any]],
    seed: int,
) -> Tuple[Counter, Counter, Dict[str, int], Dict[str, Counter], Dict[str, Counter]]:
    rng = random.Random(seed)

    items: List[Tuple[Dict[str, Any], str]] = list(iter_mcq_items(records))
    before = count_answers(items)

    total = sum(before.values())
    target = compute_target_counts(total, before)
    remaining = dict(target)

    # Per-type stats
    before_by_type: Dict[str, Counter] = defaultdict(Counter)
    after_by_type: Dict[str, Counter] = defaultdict(Counter)
    for item, vqa_type in items:
        ans = item.get("answer")
        if ans in LETTERS:
            before_by_type[vqa_type][ans] += 1

    rng.shuffle(items)
    for item, vqa_type in items:
        options = item.get("options")
        current = item.get("answer")
        if current not in LETTERS or not isinstance(options, dict):
            continue
        if any(letter not in options for letter in LETTERS):
            continue

        if remaining.get(current, 0) > 0:
            target_letter = current
        else:
            target_letter = max(LETTERS, key=lambda l: remaining.get(l, 0))

        if target_letter != current:
            swap_answer(options, current, target_letter)
            item["answer"] = target_letter

        remaining[target_letter] = remaining.get(target_letter, 0) - 1
        after_by_type[vqa_type][item["answer"]] += 1

    after = count_answers(items)
    return before, after, target, before_by_type, after_by_type


def counters_to_dict(counter: Counter) -> Dict[str, int]:
    return OrderedDict((letter, int(counter.get(letter, 0))) for letter in LETTERS)


def by_type_to_dict(by_type: Dict[str, Counter]) -> Dict[str, Dict[str, int]]:
    return OrderedDict(
        (vqa_type, counters_to_dict(counts))
        for vqa_type, counts in sorted(by_type.items(), key=lambda x: x[0])
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Balance MCQ answer distribution by swapping options."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--stats-json", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    records = read_jsonl(args.input_jsonl)
    before, after, target, before_by_type, after_by_type = balance_answers(
        records, args.seed
    )

    write_jsonl(records, args.output_jsonl)

    stats = OrderedDict(
        [
            ("total", int(sum(before.values()))),
            ("before", counters_to_dict(before)),
            ("after", counters_to_dict(after)),
            ("target", counters_to_dict(Counter(target))),
            ("before_by_type", by_type_to_dict(before_by_type)),
            ("after_by_type", by_type_to_dict(after_by_type)),
        ]
    )
    with args.stats_json.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
