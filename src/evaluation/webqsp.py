from argparse import ArgumentParser
from typing import List, Tuple, Dict, Any

from datasets import Dataset

from evaluation.base import BaseEvaluator
from data.dataset_loader import load_dataset
from misc.system_prompt import SYSTEM_PROMPT_WITHOUT_INTERFACE
from interfaces import webqsp
from verl.utils.reward_score.situated_thinker import extract_boxed
from scorers.question_answering import scorer


USER_PROMPT = """
Answer the following question given the initial entity: {q_entity}.

{question}
""".strip()


class WebQSPEvaluator(BaseEvaluator):
    def load_data(self) -> Dataset:
        eval_dataset = load_dataset("webqsp", split="test")
        eval_dataset = eval_dataset.map(
            lambda example: {
                "message": [
                    {"role": "system", "content": SYSTEM_PROMPT_WITHOUT_INTERFACE + "\n\n".join([interface.prompt for interface in webqsp.interfaces])}, 
                    {"role": "user", "content": USER_PROMPT.format(question=example["question"], q_entity=example["q_entity"])}
                ]
            }
        )
        return eval_dataset

    def score(self, idx: int, response: str, ground_truths: str) -> Tuple[int, Dict[str, Any]]:
        prediction = extract_boxed(response) or ""
        score_details = scorer(prediction, ground_truths)

        metrics = {
            "prediction": score_details["prediction"],
            "ground_truths": score_details["ground_truths"],
            "hits@1": score_details["cover_em_2"],
        }
        return idx, metrics


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = BaseEvaluator.add_args(parser)
    args = parser.parse_args()

    evaluator = WebQSPEvaluator(
        checkpoint_dir=args.checkpoint_dir,
        interface_zoo=webqsp,
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
