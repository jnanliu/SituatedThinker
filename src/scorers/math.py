import re
import logging
from typing import TypeVar, Sequence, Literal, Callable, Any, Coroutine, Optional, Union, cast
from itertools import product
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import asyncio

from sympy import Basic, MatrixBase
from latex2sympy2_extended import NormalizationConfig
from math_verify import ExprExtractionConfig, LatexExtractionConfig
from math_verify.parser import get_extraction_regexes, extract_target_from_pred, ExtractionTarget
from math_verify.grader import sympy_expr_eq


T = TypeVar('T')
logger = logging.getLogger(__name__)

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

def timeout(seconds: float):
    """
    A decorator factory that adds a timeout mechanism to a function.

    Args:
        seconds (float): The maximum number of seconds the function is allowed to run before timing out.

    Returns:
        Callable: A decorator that can be applied to a function to add a timeout.
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Coroutine[Any, Any, T]]:
        """
        A decorator that wraps a given function with a timeout mechanism.

        Args:
            func (Callable[..., Any]): The function to be wrapped with a timeout.

        Returns:
            Callable[..., Coroutine[Any, Any, T]]: An asynchronous wrapper function with timeout capability.
        """
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            """
            An asynchronous wrapper function that executes the original function with a timeout.

            Args:
                *args (Any): Positional arguments passed to the original function.
                **kwargs (Any): Keyword arguments passed to the original function.

            Returns:
                T: The return value of the original function if it completes within the timeout.

            Raises:
                TimeoutError: If the original function does not complete within the specified timeout.
            """
            # Check if the original function is a coroutine function
            is_coroutine = asyncio.iscoroutinefunction(func)
            
            async def execute() -> T:
                """
                An asynchronous helper function to execute the original function.

                Returns:
                    T: The return value of the original function.
                """
                if is_coroutine:
                    # If the function is a coroutine, await it directly
                    return await func(*args, **kwargs)
                else:
                    # If the function is not a coroutine, run it in a separate thread
                    return await asyncio.to_thread(func, *args, **kwargs)
            try:
                # Wait for the function to complete within the specified timeout
                return await asyncio.wait_for(execute(), timeout=seconds)
            except asyncio.TimeoutError:
                # Raise a TimeoutError if the function does not complete in time
                raise TimeoutError(f"Function '{func.__name__}' timed out after {seconds} seconds.")
            except Exception as e:
                # Raise any other exceptions that occur during execution
                raise e
                
        return wrapper
    return decorator

# def timeout(seconds):
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             with ThreadPoolExecutor(max_workers=1) as executor:
#                 future = executor.submit(func, *args, **kwargs)
#                 try:
#                     return future.result(timeout=seconds)
#                 except FuturesTimeoutError:
#                     raise TimeoutError(f"Function '{func.__name__}' timed out after {seconds} seconds.")
#                 except Exception as e:
#                     raise e
#         return wrapper
#     return decorator

def parse(
    pred: str,
    extraction_config: Sequence[ExtractionTarget] = [
        LatexExtractionConfig(),
        ExprExtractionConfig(),
    ],
    fallback_mode: Literal["no_fallback", "first_match"] = "first_match",
    extraction_mode: Literal["first_match", "any_match"] = "any_match",
    parsing_timeout: int = 5,
):
    """Extracts and parses mathematical expressions from a prediction string.

    This function attempts to extract mathematical expressions from text using various strategies
    (LaTeX, plain expressions, etc.) and converts them to SymPy objects.

    Args:
        pred (str): The prediction string to parse.
        extraction_config (Sequence[ExtractionTarget], optional): Configuration for what types of expressions
            to extract and how to extract them. Defaults to [LatexExtractionConfig(), ExprExtractionConfig()].
        fallback_mode (Literal["no_fallback", "first_match"], optional): How to handle extraction failures. Defaults to "first_match".
            - "no_fallback": Return only successfully parsed expressions
            - "first_match": Include the first string match even if parsing failed
        extraction_mode (Literal["first_match", "any_match"], optional): Strategy for extracting matches. Defaults to "any_match".
            - "first_match": Stop after finding the first match
            - "any_match": Try to extract all possible matches, stops after first sucesful parsing attempt
        parsing_timeout (int, optional): Maximum time in seconds to spend parsing each expression. Defaults to 3.

    Returns:
        list: List of extracted predictions. Each prediction can be:
            - SymPy expression (for successfully parsed mathematical expressions)
            - String (for fallback matches when fallback_mode="first_match")
            Empty list if no matches are found.

    Examples:
        >>> parse("The answer is $\\frac{1}{2}$")
        [Rational(1, 2)]
        >>> parse("The answer is 1/2")
        [Rational(1, 2)]
        >>> parse("The answer is A", extraction_config=[StringExtractionConfig()])
        ['a']
    """
    try:
        target_res = get_extraction_regexes(extraction_config)
        return asyncio.run(
            timeout(parsing_timeout)(extract_target_from_pred)(
                pred,
                target_res,
                fallback_mode=fallback_mode,
                extraction_mode=extraction_mode,
            )
        )
    except Exception:
        logger.exception(f"Error parsing: {pred}")
        return []
    except TimeoutError:
        logger.error(f"Timeout during parsing: {pred}")
        return []

def compare_single_extraction(
    gold: Basic | MatrixBase | str, target: Basic | MatrixBase | str, float_rounding: int, numeric_precision: int, strict: int
) -> bool:
    # If both are sympy expressions, we can use sympy to compare them
    if isinstance(gold, (Basic, MatrixBase)) and isinstance(
        target, (Basic, MatrixBase)
    ):
        return sympy_expr_eq(
            gold, target, float_rounding, numeric_precision, strict
        )

    # We don't support str / sympy.Expr comparison. Imo there is no point in doing this, as chances
    # of this happening are very low.  The only why one of them is not converted to sympy expression
    # is usually because the parsing logic failed in this case we should improve the parsing logic
    # instead of somehow fixing adhoc.
    elif isinstance(gold, str) and isinstance(target, str):
        # We just do string comparison for everything else
        gold = gold.strip()
        target = target.strip()

        # Ensure it's both not empty and equal
        return len(gold) > 0 and len(target) > 0 and gold == target

    return False
    
def verify(
    gold: list[Basic | MatrixBase | str] | Basic | MatrixBase | str,
    target: list[Basic | MatrixBase | str] | Basic | MatrixBase | str,
    float_rounding: int = 6,
    numeric_precision: int = 15,
    strict: bool = True,
    timeout_seconds: int = 5,
) -> bool:
    """Verifies if the target expression matches the gold expression using multiple comparison strategies.

    This function implements a comprehensive comparison system for mathematical expressions,
    handling various types of mathematical objects (numbers, expressions, sets, matrices, etc.)
    with multiple fallback strategies.

    Note:
        - It's expected that both gold and pred has been parsed with math_verify.parse function.
        - Function is not symmetric, gold answer should be passed as gold and prediction as pred. The non-symmetric nature appears at assignment simplification and equation interval conversion.

    Args:
        gold: The reference/correct expression(s). Can be:
            - A single SymPy expression (Basic or MatrixBase)
            - A string
            - A list of any of the above
        target: The expression(s) to verify. Same types as gold.
        float_rounding: Number of decimal places to round floats to. Defaults to 6.
        numeric_precision: Number of decimal places to consider for numeric comparisons. Defaults to 15.
            - If you know the evaluated expressions will be small, you should increase this. See: https://docs.sympy.org/latest/modules/evalf.html
        strict: Whether to enforce strict comparison mode. Defaults to True.
            - In strict mode: Variables matter and sets are not comparable with tuples
            - In non-strict mode: Variables are matched by position and sets can be compared with tuples
        timeout_seconds: Maximum time in seconds to spend on any single comparison operation.
            Defaults to 5 seconds.

    Returns:
        bool: True if target matches gold according to any of the comparison strategies,
              False otherwise.

    Comparison Strategy:
        1. String to String comparison
        2. Numeric expressions: Comparison within specified precision
        3. Symbolic equality through simplification
        4. Special handling for:
            - Relational expressions (equations/inequalities)
            - Sets and intervals
            - Matrices and vectors
            - Complex numbers
        5. Robust error handling with timeout protection

    Example:
        >>> verify(sympy.Rational(1, 3), 0.333333)  # Numeric comparison
        True
        >>> verify(sympy.Symbol('x') + 1, sympy.Symbol('y') + 1, strict=False)  # Variable matching
        True
        >>> verify(sympy.FiniteSet(1, 2), sympy.Tuple(1, 2), strict=False)  # Set-tuple comparison
        True
    """
    def compare_single_extraction_wrapper(g, t):
        try:
            return asyncio.run(
                timeout(timeout_seconds)(compare_single_extraction)(
                    g, t, float_rounding, numeric_precision, strict
                )
            )
        except Exception:
            #! Do not attempt to print out the g and t during handling of exception
            # Because a) it can throw an exception itself and b) it can cause it to be stuck forever during str conversion
            logger.exception("Error during comparison")
            return False
        except TimeoutError:
            logger.error("Timeout during comparison")
            return False

    if not isinstance(gold, list):
        gold = [gold]
    if not isinstance(target, list):
        target = [target]

    return any(
        compare_single_extraction_wrapper(g, t) for g, t in product(gold, target)
    )

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