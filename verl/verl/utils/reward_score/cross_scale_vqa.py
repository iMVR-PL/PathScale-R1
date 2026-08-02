import re
from typing import Iterable, Optional

ANSWER_TAG_RE = re.compile(r"<answer>\s*([A-Za-z])\s*</answer>")
THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

def extract_answer_letter(predict_str: str) -> Optional[str]:
    """Extracts the single answer letter inside <answer>...</answer>."""
    m = ANSWER_TAG_RE.search(predict_str)
    return m.group(1).strip().upper() if m else None

def has_valid_think_block(predict_str: str) -> bool:
    """Checks for at least one <think>...</think> block."""
    return THINK_TAG_RE.search(predict_str) is not None

def strictly_one_letter_answer(predict_str: str, valid_options: Iterable[str]) -> bool:
    """Ensures exactly one <answer>LETTER</answer> and no extra content after."""
    all_matches = list(ANSWER_TAG_RE.finditer(predict_str))
    if len(all_matches) != 1:
        return False
    letter = all_matches[0].group(1).strip().upper()
    if letter not in {v.upper() for v in valid_options}:
        return False
    # Nothing after </answer>
    tail = predict_str[all_matches[0].end():]
    return tail.strip() == ""


def format_reward(
    predict_str: str,
    valid_options: Iterable[str] = tuple(chr(i) for i in range(65, 91)),  # A–Z
    require_think: bool = True,
    strict_only_letter: bool = True,
) -> float:
    """Format reward: check structure and validity."""
    if require_think and not has_valid_think_block(predict_str):
        return 0.0
    
    # Check for exactly one valid <answer>...</answer> tag
    all_answers = re.findall(r"<answer>.*?</answer>", predict_str, re.DOTALL)
    if len(all_answers) != 1:
        return 0.0
    
    # Strict check: only one letter inside <answer> and nothing else after
    if strict_only_letter:
        return 1.0 if strictly_one_letter_answer(predict_str, valid_options) else 0.0
    letter = extract_answer_letter(predict_str)
    return 1.0 if (letter and letter in {v.upper() for v in valid_options}) else 0.0

def acc_reward(
    predict_str: str,
    ground_truth_letter: str,
    use_answer_tag: bool = True,
) -> float:
    """Accuracy reward: compares extracted letter to ground truth."""
    gt = ground_truth_letter.strip().upper()
    if use_answer_tag:
        pred = extract_answer_letter(predict_str)
    else:
        m = re.search(r"\b([A-Za-z])\b", predict_str)
        pred = m.group(1).upper() if m else None
    return 1.0 if (pred and pred == gt) else 0.0


def compute_score(
    predict_str: Optional[str] = None,
    ground_truth_letter: Optional[str] = None,
    *,
    data_source: Optional[str] = None,
    solution_str: Optional[str] = None,
    ground_truth: Optional[str] = None,
    extra_info: Optional[dict] = None,
    valid_options: Iterable[str] = tuple(chr(i) for i in range(65, 91)),  # A–Z
    require_think: bool = True,
    strict_only_letter: bool = True,
    use_answer_tag_for_acc: bool = True,
    format_weight: float = 0.2,
    **kwargs,
) -> float:
    """Final reward = (1 - format_weight)*accuracy + format_weight*format_adherence."""
    if solution_str is not None:
        predict_str = solution_str
    if ground_truth is not None:
        ground_truth_letter = ground_truth
    if predict_str is None or ground_truth_letter is None:
        raise ValueError("compute_score requires predict_str and ground_truth_letter (or solution_str/ground_truth).")
    f = format_reward(
        predict_str,
        valid_options=valid_options,
        require_think=require_think,
        strict_only_letter=strict_only_letter,
    )
    a = acc_reward(
        predict_str,
        ground_truth_letter=ground_truth_letter,
        use_answer_tag=use_answer_tag_for_acc,
    )
    return (1.0 - format_weight) * a + format_weight * f


if __name__ == "__main__":
    # Some test cases
    samples = [
        (
            # perfect prediction
            "<think>This tissue shows necrosis.</think><answer>B</answer>",
            "B",
        ),
        (
            # wrong answer but correct format
            "<think>These cells are hyperchromatic.</think><answer>C</answer>",
            "A",
        ),
        (
            # missing think tag
            "<answer>A</answer>",
            "A",
        ),
        (
            # extra text after answer
            "<think>Okay</think><answer>D</answer> The correct diagnosis is D.",
            "D",
        ),
        (
            # not only letter inside <answer> tag
            "<think>Okay</think><answer>The correct diagnosis is D.</answer> ",
            "D",
        ),
    ]

    for i, (pred, gt) in enumerate(samples):
        f = format_reward(pred)
        a = acc_reward(pred, gt)
        s = compute_score(pred, gt)
        print(f"\nCase {i+1}")
        print("Prediction:", pred)
        print("Ground truth:", gt)
        print(f"Format reward = {f:.2f} | Acc reward = {a:.2f} | Final score = {s:.2f}")
        
    """
    Case 1
    Prediction: <think>This tissue shows necrosis.</think><answer>B</answer>
    Ground truth: B
    Format reward = 1.00 | Acc reward = 1.00 | Final score = 1.00

    Case 2
    Prediction: <think>These cells are hyperchromatic.</think><answer>C</answer>
    Ground truth: A
    Format reward = 1.00 | Acc reward = 0.00 | Final score = 0.20

    Case 3
    Prediction: <answer>A</answer>
    Ground truth: A
    Format reward = 0.00 | Acc reward = 1.00 | Final score = 0.80

    Case 4
    Prediction: <think>Okay</think><answer>D</answer> The correct diagnosis is D.
    Ground truth: D
    Format reward = 0.00 | Acc reward = 1.00 | Final score = 0.80

    Case 5
    Prediction: <think>Okay</think><answer>The correct diagnosis is D.</answer> 
    Ground truth: D
    Format reward = 0.00 | Acc reward = 0.00 | Final score = 0.00
    """
