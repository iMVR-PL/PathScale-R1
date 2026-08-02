import re

ANSWER_TAG_RE = re.compile(r"<answer>\s*([A-Za-z])\s*</answer>")
FULL_FORMAT_RE = re.compile(
    r"^\s*<think>.*?</think>\s*<answer>\s*[A-Za-z]\s*</answer>\s*$",
    re.DOTALL,
)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

IMAGE_ANALYSIS_RE = re.compile(
    r"\*\*Image analysis:\*\*(.*?)(?:\*\*Option evaluation:\*\*|</think>|$)",
    re.DOTALL | re.IGNORECASE,
)
IMAGE_LABEL_RE = re.compile(
    r"(?:^|\n)\s*-?\s*\*\*Image\s+(\d+)\s*:\*\*",
    re.IGNORECASE,
)

WEIGHTS = {
    "acc": 0.75,
    "str": 0.20,
    "fmt": 0.05,
}

def extract_answer_letter(predict_str):
    """Extract the single answer letter inside <answer>...</answer>."""
    match = ANSWER_TAG_RE.search(predict_str)
    return match.group(1).strip().upper() if match else None

def accuracy_reward(predict_str, ground_truth):
    """R_accuracy = 1 if pred == gt else 0."""
    pred = extract_answer_letter(predict_str)
    gt = ground_truth.strip().upper()
    acc_reward = 1.0 if pred == gt else 0.0
    return acc_reward


def structure_reward(predict_str, num_images):
    """
    R_structure checks whether the Image analysis section matches num_images.
    """
    try:
        num_images = int(num_images)
    except (TypeError, ValueError):
        num_images = 1

    # Need <think>...</think>
    think_match = THINK_RE.search(predict_str)
    if not think_match:
        return 0.0

    think_text = think_match.group(1)

    # Need **Image analysis:** block
    image_match = IMAGE_ANALYSIS_RE.search(think_text)
    if not image_match:
        return 0.0

    image_block = image_match.group(1).strip()
    if not image_block:
        return 0.0

    # Extract labels like: - **Image 1: ... **Image 2:**
    image_ids = [int(x) for x in IMAGE_LABEL_RE.findall(image_block)]

    if num_images == 1:
        # Accept either direct paragraph or only Image 1.
        if len(image_ids) == 0 or image_ids == [1]:
            return 1.0
        return 0.0
    
    ## Cross-scale: require exactly Image 1 ... Image N.
    expected_ids = list(range(1, num_images + 1))
    # Must be exactly Image 1 ... Image N.
    if image_ids == expected_ids:
        return 1.0

    return 0.0

def format_reward(predict_str: str) -> float:
    """R_format = 1 if output matches '<think>...</think><answer>...</answer>' else 0."""
    format_reward = 1.0 if FULL_FORMAT_RE.match(predict_str) is not None else 0.0
    return format_reward


def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
    weights = WEIGHTS
    num_images = extra_info.get("num_images")
        
    r_acc = accuracy_reward(solution_str, ground_truth)
    r_str = structure_reward(solution_str, num_images)
    r_fmt = format_reward(solution_str)

    r_total = (
        weights["acc"] * r_acc
        + weights["str"] * r_str
        + weights["fmt"] * r_fmt
    )
    return {
        "score": r_total,
        "acc": r_acc,
        "structure_reward": r_str,
        "format_reward": r_fmt,
    }

if __name__ == "__main__":
    # Some test cases
    samples = [
        {
            "name": "single_direct_paragraph_correct_format_correct_answer",
            "solution_str": (
                "<think>The user wants me to identify the tissue type.\n\n"
                "**Image analysis:**\n"
                "The image shows small round blue cells with scant cytoplasm, consistent with lymphocytes.\n\n"
                "**Option evaluation:**\n"
                "- **Option A:** Incorrect.\n"
                "- **Option B:** Correct.\n"
                "</think>\n\n"
                "<answer>B</answer>"
            ),
            "ground_truth": "B",
            "extra_info": {"num_images": 1},
        },
        {
            "name": "single_with_only_image_1_allowed",
            "solution_str": (
                "<think>The user wants me to identify the tissue type.\n\n"
                "**Image analysis:**\n"
                "- **Image 1:** The image shows small round blue cells with scant cytoplasm.\n\n"
                "**Option evaluation:**\n"
                "- **Option A:** Incorrect.\n"
                "- **Option B:** Correct.\n"
                "</think>\n\n"
                "<answer>B</answer>"
            ),
            "ground_truth": "B",
            "extra_info": {"num_images": 1},
        },
        {
            "name": "single_hallucinates_image_2_should_fail_structure",
            "solution_str": (
                "<think>The user wants me to identify the tissue type.\n\n"
                "**Image analysis:**\n"
                "- **Image 1:** The image shows lymphoid cells.\n"
                "- **Image 2:** The image confirms lymphoid morphology.\n\n"
                "**Option evaluation:**\n"
                "- **Option A:** Incorrect.\n"
                "- **Option B:** Correct.\n"
                "</think>\n\n"
                "<answer>B</answer>"
            ),
            "ground_truth": "B",
            "extra_info": {"num_images": 1},
        },
        {
            "name": "single_missing_image_analysis_should_fail_structure",
            "solution_str": (
                "<think>The user wants me to identify the tissue type.\n\n"
                "**Option evaluation:**\n"
                "- **Option A:** Incorrect.\n"
                "- **Option B:** Correct.\n"
                "</think>\n\n"
                "<answer>B</answer>"
            ),
            "ground_truth": "B",
            "extra_info": {"num_images": 1},
        },
        {
            "name": "cross_scale_three_images_exact_should_pass",
            "solution_str": (
                "<think>The user wants me to compare three pathology images.\n\n"
                "**Image analysis:**\n"
                "- **Image 1:** Low-power view shows preserved architecture with abnormal tumor region.\n"
                "- **Image 2:** Mid-power view shows disorganized cellular proliferation.\n"
                "- **Image 3:** High-power view shows pleomorphic malignant cells.\n\n"
                "**Option evaluation:**\n"
                "- **Option A:** Incorrect.\n"
                "- **Option B:** Correct.\n"
                "- **Option C:** Incorrect.\n"
                "</think>\n\n"
                "<answer>B</answer>"
            ),
            "ground_truth": "B",
            "extra_info": {"num_images": 3},
        },
        {
            "name": "cross_scale_two_images_exact_should_pass",
            "solution_str": (
                "<think>The user wants me to compare two pathology images.\n\n"
                "**Image analysis:**\n"
                "- **Image 1:** Lower magnification shows glandular structures.\n"
                "- **Image 2:** Higher magnification shows nuclear atypia.\n\n"
                "**Option evaluation:**\n"
                "- **Option A:** Correct.\n"
                "- **Option B:** Incorrect.\n"
                "</think>\n\n"
                "<answer>A</answer>"
            ),
            "ground_truth": "A",
            "extra_info": {"num_images": 2},
        },
        {
            "name": "cross_scale_three_images_missing_one_should_fail_structure",
            "solution_str": (
                "<think>The user wants me to compare three pathology images.\n\n"
                "**Image analysis:**\n"
                "- **Image 1:** Low-power view shows abnormal architecture.\n"
                "- **Image 2:** Mid-power view shows tumor nests.\n\n"
                "**Option evaluation:**\n"
                "- **Option A:** Incorrect.\n"
                "- **Option B:** Correct.\n"
                "</think>\n\n"
                "<answer>B</answer>"
            ),
            "ground_truth": "B",
            "extra_info": {"num_images": 3},
        },
        {
            "name": "cross_scale_two_images_extra_image_3_should_fail_structure",
            "solution_str": (
                "<think>The user wants me to compare two pathology images.\n\n"
                "**Image analysis:**\n"
                "- **Image 1:** Lower magnification shows abnormal glands.\n"
                "- **Image 2:** Higher magnification shows atypical nuclei.\n"
                "- **Image 3:** Another high-power image confirms malignancy.\n\n"
                "**Option evaluation:**\n"
                "- **Option A:** Correct.\n"
                "- **Option B:** Incorrect.\n"
                "</think>\n\n"
                "<answer>A</answer>"
            ),
            "ground_truth": "A",
            "extra_info": {"num_images": 2},
        },
        {
            "name": "cross_scale_three_images_wrong_order_should_fail_structure",
            "solution_str": (
                "<think>The user wants me to compare three pathology images.\n\n"
                "**Image analysis:**\n"
                "- **Image 2:** Mid-power view shows disorganized cells.\n"
                "- **Image 1:** Low-power view shows abnormal architecture.\n"
                "- **Image 3:** High-power view shows pleomorphic nuclei.\n\n"
                "**Option evaluation:**\n"
                "- **Option A:** Incorrect.\n"
                "- **Option B:** Correct.\n"
                "</think>\n\n"
                "<answer>B</answer>"
            ),
            "ground_truth": "B",
            "extra_info": {"num_images": 3},
        },
        {
            "name": "correct_structure_but_wrong_answer",
            "solution_str": (
                "<think>The user wants me to identify the best option.\n\n"
                "**Image analysis:**\n"
                "The image shows features most consistent with lymphocytes.\n\n"
                "**Option evaluation:**\n"
                "- **Option A:** Incorrect.\n"
                "- **Option B:** Correct.\n"
                "</think>\n\n"
                "<answer>A</answer>"
            ),
            "ground_truth": "B",
            "extra_info": {"num_images": 1},
        },
        {
            "name": "missing_answer_tag_should_fail_format_and_accuracy",
            "solution_str": (
                "<think>The user wants me to identify the best option.\n\n"
                "**Image analysis:**\n"
                "The image shows features most consistent with lymphocytes.\n\n"
                "**Option evaluation:**\n"
                "- **Option A:** Incorrect.\n"
                "- **Option B:** Correct.\n"
                "</think>\n\n"
                "B"
            ),
            "ground_truth": "B",
            "extra_info": {"num_images": 1},
        },
        {
            "name": "no_think_tag_should_fail_format_and_structure",
            "solution_str": (
                "**Image analysis:**\n"
                "The image shows lymphocytes.\n\n"
                "**Option evaluation:**\n"
                "- **Option B:** Correct.\n"
                "<answer>B</answer>"
            ),
            "ground_truth": "B",
            "extra_info": {"num_images": 1},
        },
    ]

    for sample in samples:
        result = compute_score(
            data_source=None,
            solution_str=sample["solution_str"],
            ground_truth=sample["ground_truth"],
            extra_info=sample["extra_info"],
        )

        print("=" * 80)
        print(sample["name"])
        print(result)