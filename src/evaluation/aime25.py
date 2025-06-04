from argparse import ArgumentParser
from typing import List, Tuple, Dict, Any

from datasets import Dataset

from evaluation.base import BaseEvaluator
from data.dataset_loader import load_dataset
from misc.system_prompt import SYSTEM_PROMPT_WITHOUT_INTERFACE
from interfaces import retrieval_and_code
from verl.utils.reward_score.situated_thinker import extract_boxed
from scorers.math import scorer


class AIME25Evaluator(BaseEvaluator):
    def load_data(self) -> Dataset:
        eval_dataset = load_dataset("aime25", split="test")
        eval_dataset = eval_dataset.map(
            lambda example: {
                "message": [
                    {"role": "system", "content": SYSTEM_PROMPT_WITHOUT_INTERFACE + "\n\n".join([interface.prompt for interface in retrieval_and_code.interfaces])}, 
                    {"role": "user", "content": example["question"]}
                ]
            }
        )
        return eval_dataset

    def score(self, idx: int, response: str, ground_truths: str) -> Tuple[int, Dict[str, Any]]:
        prediction = extract_boxed(response) or ""

        for ground_truth in ground_truths:
            if scorer(prediction, ground_truth, timeout=10, math_verify=True):
                metrics = {
                    "accuracy": 1, 
                    "prediction": prediction, 
                    "ground_truths": ground_truths
                }
                return idx, metrics
        metrics = {
            "accuracy": 0, 
            "prediction": prediction, 
            "ground_truths": ground_truths
        }
        return idx, metrics


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = BaseEvaluator.add_args(parser)
    args = parser.parse_args()

    evaluator = AIME25Evaluator(
        checkpoint_dir=args.checkpoint_dir,
        interface_zoo=retrieval_and_code,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        n=args.n,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        output_path=args.output_path,
    )
    evaluator.evaluate()
