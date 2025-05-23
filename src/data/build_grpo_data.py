import os
from pathlib import Path
import re
import random

import datasets
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv("runtime.env"))

from misc.constant import DataType
from misc.system_prompt import SYSTEM_PROMPT_WITHOUT_INTERFACE
from interfaces import retrieval_and_code
from data.dataset_loader import load_dataset


def last_boxed_only_string(string):
    idx = string.rfind('\\boxed')
    if idx < 0:
        idx = string.rfind('\\fbox')
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == '{':
            num_left_braces_open += 1
        if string[i] == '}':
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def remove_boxed(s):
    left = '\\boxed{'
    try:
        assert s[:len(left)] == left
        assert s[-1] == '}'
        return s[len(left):-1]
    except Exception:
        return None

def extract_boxed(string, strip_double_curly_brace=False):
    if string.count("\\boxed") > 1:
        return None
    boxed_string = last_boxed_only_string(string)
    if boxed_string is None:
        return None
    answer = remove_boxed(boxed_string)
    if answer is None:
        return None
    if strip_double_curly_brace:
        match = re.match('^\{(.*)\}$', answer)  # noqa: W605
        if match:
            answer = match.group(1)
    return answer

def make_map_fn(split):
    def process_fn(example, idx):
        question = example.pop("question")
        answer = example.pop("answer")
        data = {
            "data_source": example["data_source"],
            "data_type": example["data_type"],
            "prompt": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT_WITHOUT_INTERFACE + \
                        "\n\n".join([interface.prompt for interface in retrieval_and_code.interfaces])
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            "question": question,
            "answer": answer,
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                "split": split,
                "index": idx
            }
        }
        return data

    return process_fn


if __name__ == "__main__":
    random.seed(42)

    train_dataset = []

    math_dataset = datasets.load_dataset("SynthLabsAI/Big-Math-RL-Verified", split="train")
    math_dataset = math_dataset.filter(lambda example: example["llama8b_solve_rate"] is not None)
    for example in random.choices(math_dataset.filter(lambda example: example["llama8b_solve_rate"] >= 0.7), k=1000):
        train_dataset.append(
            {
                "question": example["problem"],
                "answer": [example["answer"]],
                "data_type": DataType.MATH,
                "data_source": "big-math"
            }
        )
    for example in random.choices(math_dataset.filter(lambda example: (example["llama8b_solve_rate"] > 0.3 and example["llama8b_solve_rate"] < 0.7)), k=1000):
        train_dataset.append(
            {
                "question": example["problem"],
                "answer": [example["answer"]],
                "data_type": DataType.MATH,
                "data_source": "big-math"
            }
        )
    for example in random.choices(math_dataset.filter(lambda example: example["llama8b_solve_rate"] <= 0.3), k=8000):
        train_dataset.append(
            {
                "question": example["problem"],
                "answer": [example["answer"]],
                "data_type": DataType.MATH,
                "data_source": "big-math"
            }
        )

    for config in ['algebra', 'counting_and_probability', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus']:
        math_dataset = datasets.load_dataset("EleutherAI/hendrycks_math", config, split="train")
        for example in math_dataset:
            if example["level"] not in ["Level 5", "Level 4", "Level 3"]:
                continue
            answer = extract_boxed(example["solution"])
            if answer is None:
                continue
            train_dataset.append(
                {
                    "question": example["problem"],
                    "answer": [answer],
                    "data_type": DataType.MATH,
                    "data_source": f"math_level3-5"
                }
            )

    qa_dataset = load_dataset("musique", split="train")
    for example in qa_dataset:
        train_dataset.append(
            {
                "question": example["question"],
                "answer": example["answer"],
                "data_type": example["data_type"],
                "data_source": example["data_source"]
            }
        )

    random.shuffle(train_dataset)
    train_dataset = datasets.Dataset.from_list(train_dataset)
    train_dataset = train_dataset.map(make_map_fn("train"), with_indices=True)
    print(train_dataset[0])
    train_dataset.to_parquet(Path(__file__).parents[2].joinpath("cache", "data", "grpo", "train.parquet"))
    
    val_dataset = datasets.concatenate_datasets(
        [
            load_dataset("aime24", split="train"),
            load_dataset("aime25", split="test"),
            load_dataset("math500", split="test"),
            load_dataset("bamboogle", split="test")
        ]
    )
    val_dataset = val_dataset.map(make_map_fn("val"), with_indices=True)
    print(val_dataset[0])
    val_dataset.to_parquet(Path(__file__).parents[2].joinpath("cache", "data", "grpo", "val.parquet"))
