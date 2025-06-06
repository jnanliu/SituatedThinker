from argparse import ArgumentParser
from typing import List, Tuple, Dict, Any

from datasets import Dataset

from evaluation.base import BaseEvaluator
from data.dataset_loader import load_dataset
from misc.system_prompt import SYSTEM_PROMPT_WITHOUT_INTERFACE
from interfaces import retrieval_and_code
from verl.utils.reward_score.situated_thinker import extract_boxed
from scorers.question_answering import scorer


class BamboogleEvaluator(BaseEvaluator):
    def load_data(self) -> Dataset:
        eval_dataset = load_dataset("bamboogle", split="test")
        eval_dataset = eval_dataset.map(
            lambda example: {
                "message": [
                    {"role": "system", "content": SYSTEM_PROMPT_WITHOUT_INTERFACE + "\n\n".join([interface.prompt for interface in retrieval_and_code.interfaces])}, 
                    {"role": "user", "content": example["question"]}
                ]
            }
        )
        return eval_dataset

    def score(self, response: str, ground_truths: str) -> Tuple[int, Dict[str, Any]]:
        prediction = extract_boxed(response) or ""
        score_details = scorer(prediction, ground_truths)

        metrics = {
            "prediction": score_details["prediction"],
            "ground_truths": score_details["ground_truths"],
            "f1": score_details["f1"],
            "precision": score_details["precision"],
            "recall": score_details["recall"],
            "em": score_details["em"],
            "cover_em_1": score_details["cover_em_1"],
            "cover_em_2": score_details["cover_em_2"],
        }
        return metrics


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = BaseEvaluator.add_args(parser)
    args = parser.parse_args()

    evaluator = BamboogleEvaluator(
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
