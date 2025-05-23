import re

from latex2sympy2_extended import NormalizationConfig
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

REMOVED_EXPRESSIONS = [
    "square",
    "ways",
    "integers",
    "dollars",
    "mph",
    "inches",
    "hours",
    "km",
    "units",
    "\\ldots",
    "sue",
    "points",
    "feet",
    "minutes",
    "digits",
    "cents",
    "degrees",
    "cm",
    "gm",
    "pounds",
    "meters",
    "meals",
    "edges",
    "students",
    "childrentickets",
    "multiples",
    "\\text{s}",
    "\\text{.}",
    "\\text{\ns}",
    "\\text{}^2",
    "\\text{}^3",
    "\\text{\n}",
    "\\text{}",
    r"\mathrm{th}",
    r"^\circ",
    r"^{\circ}",
    r"\;",
    r",\!",
    "{,}",
    '"',
    "\\dots",
]

def normalize(text: str) -> str:
    text = text.split("=")[-1]

    # Apply substitutions and removals
    for before, after in SUBSTITUTIONS:
        text = text.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        text = text.replace(expr, "")

    # Extract and normalize LaTeX math
    text = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", text)
    text = re.sub(r"(\\text\{)(.*?)(\})", "\\2", text)
    text = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", text)
    text = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", text)
    text = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", text)

    # Normalize shorthand TeX:
    #  \fracab -> \frac{a}{b}
    #  \frac{abc}{bef} -> \frac{abc}{bef}
    #  \fracabc -> \frac{a}{b}c
    #  \sqrta -> \sqrt{a}
    #  \sqrtab -> sqrt{a}b
    text = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", text)
    text = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", text)
    text = text.replace("$", "")

    # Normalize numbers
    if text.replace(",", "").isdigit():
        text = text.replace(",", "")

    return text.strip()

def math_equiv(prediction: str, ground_truth: str, timeout: int = 60) -> bool:
    if prediction is None:
        prediction = ""

    prediction = normalize(prediction)
    ground_truth = normalize(ground_truth)

    prediction = f"\\boxed{{{prediction}}}"
    ground_truth = f"\\boxed{{{ground_truth}}}"

    parsed_ground_truth = parse(
        ground_truth, 
        extraction_config=[LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()],
        parsing_timeout=timeout
    )
    if len(parsed_ground_truth) == 0:
        parsed_ground_truth_with_env = f'${ground_truth}$'
        parsed_ground_truth = parse(
            parsed_ground_truth_with_env,
            extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()],
            parsing_timeout=timeout
        )

    if len(parsed_ground_truth) != 0:
        parsed_prediction = parse(
            prediction,
            extraction_config=[
                LatexExtractionConfig(
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                    normalization_config=NormalizationConfig(),
                )
            ],
            parsing_timeout=timeout
        )
        if verify(parsed_prediction, parsed_ground_truth, timeout_seconds=timeout):
            correct = True
        else:
            correct = False
    else:
        correct = True
    
    return {
        "correct": correct,
        "prediction": prediction,
        "ground_truth": ground_truth,
    }

def naive_math_equiv(prediction: str, ground_truth: str) -> bool:
    prediction = normalize(prediction)
    ground_truth = normalize(ground_truth)
    return {
        "correct": prediction == ground_truth,
        "prediction": prediction,
        "ground_truth": ground_truth,
    }

def scorer(prediction: str, ground_truth: str, timeout: int = 60, math_verify: bool = False) -> bool:
    """
    Determine whether the prediction is equivalent to the ground truth, 
    either by simple string comparison or mathematical verification.

    Args:
        prediction (str): The predicted mathematical expression.
        ground_truth (str): The correct mathematical expression.
        timeout (int, optional): The timeout duration in seconds for mathematical verification. Defaults to 60.
        math_verify (bool, optional): If True, perform mathematical verification; 
                                    if False, perform a simple string comparison. Defaults to False.

    Returns:
        bool: True if the prediction is equivalent to the ground truth, False otherwise.
    """
    if math_verify:
        # Perform mathematical verification using math_equiv function
        return math_equiv(prediction, ground_truth, timeout)
    else:
        # Perform a simple string comparison using naive_math_equiv function
        return naive_math_equiv(prediction, ground_truth)