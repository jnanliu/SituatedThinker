import re
from typing import List, Dict
from itertools import islice, zip_longest
from collections import defaultdict
from functools import partial

import scorers
from misc.constant import DataType
from interfaces import retrieval_and_code
import scorers.question_answering


# Compile a regular expression pattern to match a specific format of strings.
# This pattern is used to extract reasoning, conclusion, and answer from a given string.
format_pattern = re.compile(
    r"""
    ^                                 # Match the start of the string.
    (?!.*<conclusion>.*<conclusion>)  # Negative lookahead to ensure <conclusion> tag appears only once.
    (?!.*</conclusion>.*</conclusion>)# Negative lookahead to ensure </conclusion> tag appears only once.
    (?!.*\\boxed.*\\boxed)            # Negative lookahead to ensure \boxed appears only once.
    (?P<reasoning>.*?)                # Non-greedily capture any characters and name this group 'reasoning'.
    \n<conclusion>\n                  # Match a newline, the <conclusion> tag, and another newline.
    # Capture the conclusion part
    (?P<conclusion>                   # Start capturing the conclusion part and name this group 'conclusion'.
        (.*?(?P<answer>\\boxed\{      # Non-greedily capture any characters until "\boxed{", 
                                     # then start capturing the answer and name this group 'answer'.
    # Capture the boxed part
    (.*)                          # Capture any characters that are not curly braces.
    \}).*?)                           # Match the closing brace of \boxed{} and any remaining characters non-greedily.
    )
    \n</conclusion>                   # Match a newline, the </conclusion> tag.
    $                                 # Match the end of the string.
    """,
    re.DOTALL | re.VERBOSE
)
boxed_pattern = re.compile(
    r"""
    \\boxed\{                       # Match the literal string "\boxed{"
    # Capture the boxed part
    (.*)
    \}                              # Match the closing brace
    """,
    re.DOTALL | re.VERBOSE,
)

def parse_response(completion: str) -> Dict[str, str] | None:
    """
    Parse the given completion string to extract the reasoning process, conclusion block, and answer.

    This function checks if the completion string contains exactly one <conclusion> tag, 
    one </conclusion> tag, and the closing tag comes after the opening tag. It also checks 
    if the string contains exactly one \\boxed{} environment. If all checks pass, it uses 
    predefined regular expression patterns to extract the reasoning process, conclusion block, 
    and answer.

    Args:
        completion (str): The completion string to be parsed.

    Returns:
        Dict[str, str] | None: A dictionary containing the keys "reasoning", "conclusion", 
        and "answer", or None if parsing fails.
    """
    # Attempt to match the entire completion string using the predefined format_pattern
    # The format_pattern is designed to enforce specific structural rules on the completion string
    # such as single occurrence of <conclusion> tags and \boxed{} environment.
    match = format_pattern.match(completion)
    if not match:
        # Return None if the matching fails, indicating the string does not meet the required format.
        return None

    # Extract the reasoning process from the match result and strip leading and trailing whitespace
    # The "reasoning" group is defined in the format_pattern regular expression.
    reasoning = match.group("reasoning").strip()
    # Extract the conclusion block from the match result and strip leading and trailing whitespace
    # The "conclusion" group is defined in the format_pattern regular expression.
    conclusion_block = match.group("conclusion").strip()
    # Extract the answer from the match result and strip leading and trailing whitespace
    # The "answer" group is defined in the format_pattern regular expression.
    answer = match.group("answer").strip()
    
    # Return a dictionary containing the reasoning process, conclusion block, and answer
    return {
        "reasoning": reasoning,
        "conclusion": conclusion_block,
        "answer": answer
    }

def extract_boxed(completion: str) -> str | None:
    """
    Extract the content enclosed within the \\boxed{} environment from the given completion string.

    This function uses a predefined regular expression pattern to search for all occurrences of
    content within the \\boxed{} environment in the input string. It then returns the last match
    if any matches are found.

    Args:
        completion (str): The input string from which the \\boxed{} content is to be extracted.

    Returns:
        str | None: The extracted content if found, otherwise None.
    """
    # Use the predefined regular expression pattern `boxed_pattern` to find all occurrences
    # of the \\boxed{} environment in the completion string.
    # `findall` returns a list of all non-overlapping matches in the string.
    match = boxed_pattern.findall(completion)
    # Check if any matches were found. If so, return the content of the last match.
    # If no matches were found, return None.
    return match[-1] if len(match) == 1 else None

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
    return 1 if parse_response(completion) is not None else 0

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
    # Parse the completion string using the parse_response function
    pared_completion = parse_response(completion)
    # Check if the completion string does not match the predefined format
    if pared_completion is None:
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
        prediction = pared_completion["answer"]
        # Calculate the F1 score by passing the prediction and ground truth answers 
        # to the scorer function from the question_answering module in scorers.
        f1 = scorers.question_answering.scorer(prediction, ground_truths)["f1"]
        # Return the computed F1 score
        return f1

def compute_math_score(
    completion: str, 
    ground_truths: List[str], 
    is_validate: bool = False, 
    math_verify: bool = True,
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
        math_verify (bool, optional): If True, perform mathematical verification; otherwise, do a simple string comparison. Defaults to False.

    Returns:
        float: 1.0 if the prediction is mathematically equivalent to any ground truth, 0.0 otherwise.
    """
    # Parse the completion string using the parse_response function
    pared_completion = parse_response(completion)
    # Check if the completion string does not match the predefined format
    if pared_completion is None:
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
        prediction = pared_completion["answer"]
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
            self.compute_accuracy_score = (
                partial(compute_math_score, math_verify=True)
                if self.is_validate else
                partial(compute_math_score, math_verify=False)
            )
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
