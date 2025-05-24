import re
from typing import List, Dict
from itertools import islice, zip_longest
from collections import defaultdict

import scorers
from misc.constant import DataType
from interfaces import retrieval_and_code
import scorers.question_answering


format_pattern = re.compile(
    r"""
    ^                               # Start of the string
    # Ensure no duplicate <conclusion> tags
    (?!.*<conclusion>.*<conclusion>)
    # Ensure no duplicate </conclusion> tags
    (?!.*</conclusion>.*</conclusion>)
    # Ensure no duplicate \boxed commands
    (?!.*\\boxed.*\\boxed)
    # Capture the reasoning part non-greedily
    (?P<reasoning>.*?)
    <conclusion>                    # Match the opening <conclusion> tag
    \s*                             # Match zero or more whitespace characters
    # Capture the conclusion part
    (?P<conclusion>
        .*?                         # Match any characters non-greedily
        The\s+answer\s+is\s+\\boxed\{ # Match the literal string "The answer is \boxed{"
        # Capture the answer part
        (?P<answer>.+?)
        \}                          # Match the closing brace
        .*?                         # Match any characters non-greedily
    )
    \s*                             # Match zero or more whitespace characters
    </conclusion>                   # Match the closing </conclusion> tag
    $                               # End of the string
    """,
    re.DOTALL | re.VERBOSE,
)
boxed_pattern = re.compile(
    r"""
    ^                               # Start of the string
    # Ensure no duplicate \boxed commands
    (?!.*\\boxed.*\\boxed)
    .*?                             # Match any characters non-greedily
    \\boxed\{                       # Match the literal string "\boxed{"
    # Capture the boxed part
    (?P<boxed>.+?)
    \}                              # Match the closing brace
    .*?                             # Match any characters non-greedily
    $                               # End of the string
    """,
    re.DOTALL | re.VERBOSE,
)

def extract_reasoning(completion: str)  -> str | None:
    """
    Extract the reasoning part from the given completion string.

    This function uses a predefined regular expression pattern to search for the reasoning
    part before the <conclusion> tag in the input string.

    Args: 
        completion (str): The input string from which the reasoning is to be extracted.

    Returns:
        str | None: The extracted reasoning string if found, otherwise None.
    """
    # Strip leading and trailing whitespace from the input string
    # and attempt to match it against the predefined regular expression pattern
    match = format_pattern.match(completion.strip())
    # If a match is found, return the 'reasoning' named group from the match object
    # Otherwise, return None
    return match.group("reasoning") if match else None

def extract_conclusion(completion: str)  -> str | None:
    """
    Extract the conclusion part from the given completion string.

    This function uses a predefined regular expression pattern to search for the conclusion
    enclosed between <conclusion> and </conclusion> tags in the input string.

    Args: 
        completion (str): The input string from which the conclusion is to be extracted.

    Returns:
        str | None: The extracted conclusion string if found, otherwise None.
    """
    # Strip leading and trailing whitespace from the input string
    # Then attempt to match the entire string against the predefined regular expression pattern
    match = format_pattern.match(completion.strip())
    # If a match is found, extract and return the 'conclusion' named group from the match object
    # Otherwise, return None
    return match.group("conclusion") if match else None

def extract_answer(completion: str) -> str | None:
    """
    Extract the answer part from the given completion string.

    This function uses a predefined regular expression pattern to search for the answer
    enclosed within the <conclusion> tags and the \\boxed{} environment in the input string.

    Args:
        completion (str): The input string from which the answer is to be extracted.

    Returns:
        str | None: The extracted answer string if found, otherwise None.
    """
    # Strip leading and trailing whitespace from the input string
    # Then attempt to match the entire string against the predefined regular expression pattern
    match = format_pattern.match(completion.strip())
    # If a match is found, extract and return the 'answer' named group from the match object
    # Otherwise, return None
    return match.group("answer") if match else None

def extract_boxed(completion: str) -> str | None:
    """
    Extract the content enclosed within the \\boxed{} environment from the given completion string.

    This function uses a predefined regular expression pattern to search for all occurrences of
    content within the \\boxed{} environment in the input string. It then returns the first match
    if any matches are found.

    Args:
        completion (str): The input string from which the \\boxed{} content is to be extracted.

    Returns:
        str | None: The extracted content if found, otherwise None.
    """
    # Strip leading and trailing whitespace from the input string
    # to ensure accurate matching with the regular expression pattern
    # Then attempt to match the entire stripped string against the predefined boxed_pattern
    match = boxed_pattern.match(completion.strip())
    # Check if a match was found using the regular expression pattern
    # If a match is found, retrieve the 'boxed' named group from the match object
    # Otherwise, return None indicating no \\boxed{} content was found
    return match.group("boxed") if match else None

def check_format(completion: str) -> bool:
    """
    Check if the given completion string matches the predefined format pattern.

    This function uses the `re.match` method to attempt to match the entire 
    `completion` string against the `format_pattern`. The `re.DOTALL` flag, which is set 
    when compiling `format_pattern`, allows the dot (`.`) to match any character including newlines. 
    Note that `re.MULTILINE` flag is not used here; in the current `format_pattern`, `^` and `$` 
    match the start and end of the entire string.

    Args:
        completion (str): The input string to be checked against the format pattern.

    Returns:
        bool: True if the `completion` string matches the format pattern, False otherwise.
    """
    # Use re.match to check if the stripped completion string matches the format pattern
    # If a match is found, re.match returns a match object; otherwise, it returns None
    # Convert the result to a boolean value: True if a match is found, False otherwise
    return format_pattern.match(completion.strip()) is not None

def compute_format_score(completion: str) -> float:
    """
    Compute the format reward for a given completion string.

    This function checks if the provided completion string matches the predefined format pattern
    using the `check_format` function. If it matches, the format score is set to 1; otherwise, it is set to 0.

    Args:
        completion (str): The input string to be checked for format compliance.

    Returns:
        float: The format score, 1 if the completion string matches the format, 0 otherwise.
               Note: The docstring previously indicated a dictionary return type, 
               but the actual implementation returns a float.
    """
    # Check if the completion string matches the predefined format
    # If it does, return 1 as the format reward; otherwise, return 0
    return 1 if check_format(completion) else 0

def compute_f1_score(
    completion: str, 
    ground_truths: List[str], 
    is_validate: bool = False,
) -> float:        
    """
    Compute the F1 score reward for a given completion string against a list of ground truth answers.

    This function checks if the completion string matches the predefined format. 
    Depending on the format and validation status, it extracts the prediction from the completion 
    and calculates the F1 score using the `scorers.question_answering.scorer` function.

    Args:
        completion (str): The input string containing the predicted answer.
        ground_truths (List[str]): A list of ground truth answers.
        is_validate (bool, optional): A flag indicating whether the function is in validation mode. Defaults to False.
        penalty (float, optional): The penalty value to return if the F1 score is unavailable and not in validation mode. Defaults to -0.1.

    Returns:
        float: The computed F1 score.
    """
    # Check if the completion string does not match the predefined format
    if not check_format(completion):
        # Extract the prediction from the completion string. 
        # If extraction fails, default to an empty string.
        prediction = extract_boxed(completion) or ""
        # Calculate the F1 score by passing the prediction and ground truth answers 
        # to the scorer function from the question_answering module in scorers.
        f1 = scorers.question_answering.scorer(prediction, ground_truths)["f1"]
        # Return the computed F1 score
        return f1
    else:
        # If the completion string matches the format, extract the answer from the completion.
        prediction = extract_answer(completion)
        # Calculate the F1 score by passing the prediction and ground truth answers 
        # to the scorer function from the question_answering module in scorers.
        f1 = scorers.question_answering.scorer(prediction, ground_truths)["f1"]
        # Return the computed F1 score
        return f1

def compute_math_score(
    completion: str, 
    ground_truths: List[str], 
    is_validate: bool = False, 
    math_verify: bool = False,
) -> float:
    """
    Compute the math reward by verifying the mathematical equivalence of the prediction and ground truth.

    This function checks if the completion string matches the predefined format. 
    Depending on the format and validation status, it extracts the prediction from the completion 
    and verifies its mathematical equivalence with each ground truth answer using `scorers.math.scorer`.

    Args:
        completion (str): The input string containing the predicted answer.
        ground_truths (List[str]): A list of ground truth answers.
        is_validate (bool, optional): A flag indicating whether the function is in validation mode. Defaults to False.
        math_verify (bool, optional): If True, perform mathematical verification; otherwise, do a simple string comparison. Defaults to True.

    Returns:
        float: 1.0 if the prediction is mathematically equivalent to any ground truth, 0.0 otherwise.
    """
    # Check if the completion string does not match the predefined format
    if not check_format(completion):
        # Extract the prediction from the completion string, default to an empty string if extraction fails
        prediction = extract_boxed(completion) or ""
        # Iterate through each ground truth answer
        for ground_truth in ground_truths:
            # Verify the mathematical equivalence of the prediction and ground truth with a 10-second timeout
            if scorers.math.scorer(prediction, ground_truth, timeout=10, math_verify=math_verify)["correct"]:
                # Return 1.0 if an equivalent answer is found
                return 1.0
        # Return 0.0 if no equivalent answer is found
        return 0.0
    else:
        # If the completion string matches the format, extract the answer from the completion
        prediction = extract_answer(completion)
        # Iterate through each ground truth answer
        for ground_truth in ground_truths:
            # Verify the mathematical equivalence of the prediction and ground truth with a 10-second timeout
            if scorers.math.scorer(prediction, ground_truth, timeout=10, math_verify=math_verify)["correct"]:
                # Return 1.0 if an equivalent answer is found
                return 1.0
        # Return 0.0 if no equivalent answer is found
        return 0.0

def compute_invocation_score(completion: str) -> Dict[str, float]:
    """
    Compute the reward score for interface invocations in the given completion string.

    This function uses the `count_query` method from the `interface_zoo` module to count 
    the number of interface queries in the completion string. It then formats the counts 
    into a dictionary where keys are prefixed with '_query_score'.

    Args:
        completion (str): The input string containing potential interface invocations.

    Returns:
        Dict[str, float]: A dictionary mapping interface names with '_query_score' suffix 
                          to their corresponding invocation counts.
    """
    # Use the count_query method from the interface_zoo module to count interface queries
    # in the provided completion string. The result is a dictionary with interface names as keys
    # and their invocation counts as values.
    count = retrieval_and_code.count_query(completion)
    # Rename the keys in the count dictionary by appending '_query_score' suffix to each key.
    # This is done to standardize the key naming convention for the reward scores.
    count = {f"{k}_invocation_score": v for k, v in count.items()}
    return count

def compute_repeat_score(completion: str) -> float:
    def repeatness(s: str) -> float:
        """
        Calculate the repeatability score of a given string.

        Args:
            s (str): The input string.

        Returns:
            float: The repeatability score. A higher score indicates more repetition in the string.
        """
        def ranks(l):
            """
            Convert a list of values to a list of ranks.

            Args:
                l (list): The input list.

            Returns:
                list: The list of ranks. Each element in the returned list represents the rank of the corresponding element in the original list.
            """
            # Create a dictionary that maps each unique value in the list to its rank
            index = {v: i for i, v in enumerate(sorted(set(l)))}
            # Return a list of ranks corresponding to each value in the original list
            return [index[v] for v in l]

        def suffixArray(s):
            """
            Generate the suffix array and rank array of a given list.

            A suffix array is an array of integers containing all the starting positions of suffixes of a string in lexicographical order.
            The rank array stores the rank of each suffix in the lexicographical order.

            Args:
                s (list): The input list.

            Returns:
                tuple: A tuple containing the rank array and the suffix array.
            """
            # Convert the list to a list of ranks
            line = ranks(s)
            # Get the length of the list
            n = len(s)
            # Initialize the step size
            k = 1
            # Initialize the rank array
            ans = line
            # Initialize the suffix array
            sa = [0] * len(s)
            # Iterate until the step size is greater than or equal to the length of the list
            while k < n - 1:
                # Generate a new list of tuples by combining the current rank array with a shifted version of itself
                # This helps in comparing suffixes of different lengths
                line = ranks(list(zip_longest(line, islice(line, k, None), fillvalue=-1)))
                # Update the rank array
                ans = line
                # Double the step size
                k = k << 1
            # Fill the suffix array with the indices of the sorted suffixes
            for i, k in enumerate(ans):
                sa[k] = i
            # Return the rank array and the suffix array
            return ans, sa

        def lcp(arr, suffixArr, inv_suff):
            """
            Calculate the longest common prefix array for a given list.

            The longest common prefix (LCP) array stores the length of the longest common prefix between consecutive suffixes in the suffix array.

            Args:
                arr (list): The input list.
                suffixArr (list): The suffix array of the input list.
                inv_suff (list): The inverse suffix array of the input list. The inverse suffix array maps each suffix to its position in the suffix array.

            Returns:
                list: The longest common prefix array.
            """
            # Get the length of the list
            n = len(arr)
            # Initialize the longest common prefix array
            ans = [0] * len(arr)
            # Initialize the current longest common prefix length
            k = 0
            # Iterate through each index in the list
            for i in range(n):
                # If the current index is the last index in the inverse suffix array, reset the longest common prefix length
                if inv_suff[i] == n - 1:
                    k = 0
                    continue
                # Get the index of the next suffix in the suffix array
                j = suffixArr[inv_suff[i] + 1]
                # Calculate the longest common prefix length between the current suffix and the next suffix
                while i + k < n and j + k < n and arr[i + k] == arr[j + k]:
                    k += 1
                # Store the longest common prefix length in the longest common prefix array
                ans[inv_suff[i]] = k
                # Decrease the longest common prefix length by 1 if it is greater than 0
                if k > 0:
                    k -= 1
            # Return the longest common prefix array
            return ans

        # Convert the input string to a list of ASCII values
        arr = [ord(i) for i in s]
        # Get the length of the list
        n = len(arr)
        # If the length of the list is less than or equal to 1, return 0 as there can't be repetition
        if n <= 1:
            return 0
        # Generate the rank array and the suffix array
        c, sa = suffixArray(arr)
        # Calculate the sum of the longest common prefix array
        cnt = sum(lcp(arr, sa, c))
        # Calculate and return the repeatability score
        # The formula normalizes the sum of LCPs by the total number of possible pairs of characters in the string
        return cnt * 2 / (n * (n + 1))

    return repeatness(completion)

def compute_reflection_score(completion: str) -> float:
    def check_reflection_pattern(completion: str) -> dict[str, int]:
        """
        Check the given completion string for the presence of reflection pattern words.

        This function searches for predefined reflection pattern words in the completion string
        using regular expressions. It then counts the occurrences of each pattern and returns
        a dictionary with the patterns as keys and their counts as values.

        Args:
            completion (str): The input string to be checked for reflection pattern words.

        Returns:
            dict[str, int]: A dictionary mapping each reflection pattern word to its occurrence count.
        """
        # TODO: may need to add more pattern
        # List of regular expression patterns representing reflection pattern words
        reflection_pattern_words = [
            r"wait,",
            r"recheck[,\s]",
            r"retry",
            r"alternatively,",
            r"however,",
        ]
        # Initialize a defaultdict to store the count of each pattern, default value is 0
        res = defaultdict(int)
        # Iterate through each reflection pattern word
        for word in reflection_pattern_words:
            # Use regular expression to find all occurrences of the pattern in the completion string
            # and count them
            res[word] = len(re.findall(word, completion))
        return res

    reflection_pattern_words = check_reflection_pattern(completion)
    return sum(reflection_pattern_words.values())


class Scorer:
    def __init__(self, data_type: DataType, is_validate: bool = False):
        self.data_type = data_type
        self.is_validate = is_validate

        if self.data_type == DataType.MATH:
            self.compute_accuracy_score = compute_math_score
        elif self.data_type == DataType.MULTIHOPQA:
            self.compute_accuracy_score = compute_f1_score
        else:
            raise ValueError(f"Unsupported data type: {self.data_type}")

    def compute_score(self, completion: str, ground_truth: str | List[str]) -> Dict[str, float]:
        accuracy_score = self.compute_accuracy_score(completion, ground_truth, self.is_validate)

        format_score, invocation_score, repeat_score, reflection_score = [
            func(completion) 
            for func, completion in zip(
                [compute_format_score, compute_invocation_score, compute_repeat_score, compute_reflection_score],
                [completion] * 4
            )
        ]

        if self.is_validate:
            accuracy_score = max(accuracy_score, 0.0)

        score_details = {
            "accuracy_score": accuracy_score,
            f"{DataType.MULTIHOPQA.replace(' ', '-')}_accuracy_score": None,
            f"{DataType.MATH.replace(' ', '-')}_accuracy_score": None,
            "format_score": format_score,
            **invocation_score,
            "repeat_score": repeat_score,
            "reflection_score": reflection_score,
        }

        score_details[f"{self.data_type.replace(' ', '-')}_accuracy_score"] = accuracy_score

        return score_details
