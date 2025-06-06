import re
import string
from typing import List, Tuple, Dict

import nltk


def f1_score(normalized_prediction: str, normalized_ground_truth: str) -> Tuple[float, float, float]:
    if normalized_prediction in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return 0.0, 0.0, 0.0
    if normalized_ground_truth in ["yes", "no", "noanswer"] and normalized_prediction != normalized_ground_truth:
        return 0.0, 0.0, 0.0

    prediction_tokens = set(normalized_prediction.split())
    ground_truth_tokens = set(normalized_ground_truth.split())
    
    precision = nltk.precision(ground_truth_tokens, prediction_tokens)
    recall = nltk.recall(ground_truth_tokens, prediction_tokens)
    f1 = nltk.f_measure(ground_truth_tokens, prediction_tokens)
    return f1, precision, recall

def exact_match_score(normalized_prediction: str, normalized_ground_truth: str) -> bool:
    return normalized_prediction == normalized_ground_truth

def cover_exact_match_1_score(normalized_prediction: str, normalized_ground_truth: str) -> bool:
    pred_token_list = normalized_prediction.split(" ")
    gold_token_list = normalized_ground_truth.split(" ")
    return all(token in pred_token_list for token in gold_token_list)

def cover_exact_match_2_score(normalized_prediction: str, normalized_ground_truth: str) -> bool:
    pred_token_list = normalized_prediction.split(" ")
    gold_token_list = normalized_ground_truth.split(" ")
    for i in range(len(pred_token_list) - len(gold_token_list) + 1):
        if pred_token_list[i:i + len(gold_token_list)] == gold_token_list:
            return True
    pred_str = " ".join(pred_token_list)
    gold_str = " ".join(gold_token_list)
    if gold_str in pred_str:
        return True
    return False

def scorer(prediction: str, ground_truths: str | List[str]) -> Dict[str, float]:
    """
    Calculate multiple evaluation metrics (F1, precision, recall, exact match, etc.) 
    between a prediction and one or more ground truth answers.

    Args:
        prediction (str): The predicted answer.
        ground_truths (str | List[str]): The ground truth answer(s), either a single string or a list of strings.

    Returns:
        Dict[str, float]: A dictionary containing evaluation metrics and processed answers.
    """
    def normalize_answer(s: str) -> str:
        """
        Normalize the input string by applying a series of text processing steps.

        Args:
            s (str): The input string to be normalized.

        Returns:
            str: The normalized string.
        """
        def remove_articles(text: str) -> str:
            # Remove English articles (a, an, the) from the text
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def remove_punc(text: str) -> str:
            # Define a set of punctuation characters to be removed
            exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
            # Remove punctuation characters from the text
            return "".join(ch if ch not in exclude else " " for ch in text)

        def lower(text: str) -> str:
            # Convert the text to lowercase
            return text.lower()

        def replace_underscore(text: str) -> str:
            # Replace underscores with spaces
            return text.replace("_", " ")
        
        def white_space_fix(text: str) -> str:
            # Replace multiple consecutive whitespace characters with a single space
            return re.sub(r'\s+', ' ', text)
        
        def homogeneize_numbers(text: str) -> str:
            """
            Convert numeric words in the text to their float string representation.

            Args:
                text (str): The input text.

            Returns:
                str: The text with numeric words converted to float strings.
            """
            words = []
            for word in text.split(" "):
                try:
                    words.append(str(float(word)))
                except ValueError:
                    words.append(word)
            return " ".join(words)

        # Apply all normalization steps in sequence
        return white_space_fix(remove_articles(remove_punc(lower(s)))).strip()

    def bool_mapping(s):
        """
        Map boolean strings "True" and "False" to "yes" and "no".

        Args:
            s (str): The input string.
        Returns:
            str: The mapped string.
        """
        if s == "true":
            return "yes"
        elif s == "false":
            return "no"
        else:
            return s
    
    # Handle None prediction
    if prediction is None:
        prediction = ""
    # Convert single ground truth string to a list
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    # Normalize the prediction after boolean mapping
    normalized_prediction = normalize_answer(bool_mapping(prediction.strip()))
    # Normalize each ground truth answer after boolean mapping
    normalized_ground_truths = [normalize_answer(bool_mapping(ground_truth.strip())) for ground_truth in ground_truths]

    # Initialize F1, precision, and recall scores
    f1, precision, recall = 0., 0., 0.
    # Calculate F1, precision, and recall for each ground truth answer
    for normalized_ground_truth in normalized_ground_truths:
        f1_, precision_, recall_ = f1_score(normalized_prediction, normalized_ground_truth)
        f1, precision, recall = max(f1, f1_ or 0.0), max(precision, precision_ or 0.0), max(recall, recall_ or 0.0)
    
    # Initialize exact match score
    em = 0.
    # Calculate exact match score for each ground truth answer
    for normalized_ground_truth in normalized_ground_truths:
        em = max(em, exact_match_score(normalized_prediction, normalized_ground_truth))
    
    # Initialize cover exact match 1 score
    cem_1 = 0.
    # Calculate cover exact match 1 score for each ground truth answer
    for normalized_ground_truth in normalized_ground_truths:
        cem_1 = max(cem_1, cover_exact_match_1_score(normalized_prediction, normalized_ground_truth))

    # Initialize cover exact match 2 score
    cem_2 = 0.
    # Calculate cover exact match 2 score for each ground truth answer
    for normalized_ground_truth in normalized_ground_truths:
        cem_2 = max(cem_2, cover_exact_match_2_score(normalized_prediction, normalized_ground_truth))

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "em": em,
        "cover_em_1": cem_1,
        "cover_em_2": cem_2,
        "prediction": normalized_prediction,
        "ground_truths": normalized_ground_truths
    }
