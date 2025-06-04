from argparse import ArgumentParser
from typing import List, Tuple, Dict, Any
from pathlib import Path

from datasets import Dataset
from textworld import gym, EnvInfos

from evaluation.base import BaseEvaluator
from data.dataset_loader import load_dataset
from misc.system_prompt import SYSTEM_PROMPT_WITHOUT_INTERFACE
from interfaces import textworld
from verl.utils.reward_score.situated_thinker import extract_boxed


USER_PROMPT = """
Game Id:
{idx}

Game Objective:
{question}

Please generate a command sequence separated by commas to win the game, e.g., command1, command2, ..., and each command is a phrase such as `go north`
""".strip()


class TextWorldEvaluator(BaseEvaluator):
    def load_data(self) -> Dataset:
        eval_dataset = load_dataset("textworld", split="test")
        eval_dataset = eval_dataset.map(
            lambda example: {
                "message": [
                    {"role": "system", "content": SYSTEM_PROMPT_WITHOUT_INTERFACE + "\n\n".join([interface.prompt for interface in textworld.interfaces])}, 
                    {"role": "user", "content": USER_PROMPT.format(question=example["question"], idx=example["idx"])}
                ]
            }
        )
        return eval_dataset

    def score(self, idx: int, response: str, ground_truths: str) -> Tuple[int, Dict[str, Any]]:
        prediction = extract_boxed(response) or ""

        env_id = gym.register_game(
            Path(__file__).parents[2].joinpath("cache", "data", "textworld", f"custom_game_{idx}.z8"),
            max_episode_steps=100,
            request_infos=EnvInfos(
                description=True,  
                inventory=True, 
                game=True, 
                facts=True, 
                admissible_commands=True
            )
        )
        
        env = gym.make(env_id) 
        obs, infos = env.reset()
        
        done = False
        for command in prediction.split(","):
            obs, score, done, infos = env.step(command)
            env.render()
            if done:
                break

        metrics = {
            "prediction": prediction,
            "pass": done
        }

        return idx, metrics


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = BaseEvaluator.add_args(parser)
    args = parser.parse_args()

    evaluator = TextWorldEvaluator(
        checkpoint_dir=args.checkpoint_dir,
        interface_zoo=textworld,
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
