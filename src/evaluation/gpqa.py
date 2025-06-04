from argparse import ArgumentParser
from typing import List, Tuple, Dict, Any

from datasets import Dataset

from evaluation.base import BaseEvaluator
from data.dataset_loader import load_dataset
from misc.system_prompt import SYSTEM_PROMPT_WITHOUT_INTERFACE
from interfaces import retrieval_and_code
from verl.utils.reward_score.situated_thinker import extract_boxed


USER_PROMPT = """
Answer the following multiple choice question. Your answer should be one of ABCD and do not involve the content of options.

{question}

{options}
""".strip()


class GPQAEvaluator(BaseEvaluator):
    def load_data(self) -> Dataset:
        eval_dataset = load_dataset("gpqa", split="test")
        eval_dataset = eval_dataset.map(
            lambda example: {
                "message": [
                    {"role": "system", "content": SYSTEM_PROMPT_WITHOUT_INTERFACE + "\n\n".join([interface.prompt for interface in retrieval_and_code.interfaces])}, 
                    {"role": "user", "content": USER_PROMPT.format(question=example["question"], options=example["options"])}
                ]
            }
        )
        return eval_dataset

    def score(self, idx: int, response: str, ground_truths: str) -> Tuple[int, Dict[str, Any]]:
        prediction = extract_boxed(response) or ""

        prediction = prediction.lower()
        if prediction.endswith(")"):
            prediction = prediction.split(")")[:-1]
        ground_truth = ground_truth.lower()

        metrics = {
            "prediction": prediction,
            "ground_truths": ground_truths,
            "accuracy": any([prediction == ground_truth for ground_truth in ground_truths]),
        }
        return idx, metrics


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = BaseEvaluator.add_args(parser)
    args = parser.parse_args()

    evaluator = GPQAEvaluator(
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
