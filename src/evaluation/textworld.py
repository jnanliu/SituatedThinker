from argparse import ArgumentParser
from typing import List, Tuple, Dict, Any
from pathlib import Path
from functools import partial
from collections import defaultdict
import json

import numpy as np
from datasets import Dataset
from pprint import pprint
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

    def score(self, response: str, ground_truths: str) -> Tuple[int, Dict[str, Any]]:
        prediction = extract_boxed(response) or ""

        env_id = gym.register_game(
            str(Path(__file__).parents[2].joinpath("cache", "data", "textworld", f"custom_game_{ground_truths}.z8")),
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

        return metrics
    
    def evaluate(self):
        tokenize_fn = partial(
            self.llm.get_tokenizer().apply_chat_template, 
            tokenize=True, 
            add_generation_prompt=True
        )
        eval_dataset = self.load_data()
        eval_dataset = eval_dataset.map(lambda example: {"prompt_token_ids": tokenize_fn(example["message"])})

        prompts = [
            {
                "prompt_token_ids": prompt_token_ids 
            } for prompt_token_ids in eval_dataset["prompt_token_ids"]
        ]
        extra_infos_of_prompts = [
            {
                "data_source": example["data_source"], 
                "data_type": example["data_type"]
            } for example in eval_dataset
        ]

        outputs = []
        outputs.extend(
            self.llm.generate(
                prompts, 
                sampling_params=self.sampling_params,
                extra_infos_of_prompts=extra_infos_of_prompts,
                use_tqdm=True
            )
        )
        responses = [output.outputs[0].text for output in outputs]

        repeat_item = lambda items: [item for item in items for _ in range(self.n)]
        data_types = repeat_item(eval_dataset["data_type"])
        data_sources = repeat_item(eval_dataset["data_source"])
        answers = repeat_item(eval_dataset["idx"])
        sample_idxs = repeat_item(eval_dataset["idx"])
        prompts = repeat_item(prompts)

        metrics = []
        for response, answer in zip(responses, answers):
            metrics.append(self.score(response, answer))

        data2metric = defaultdict(list)
        for data_source, metric, sample_idx, response, prompt in zip(data_sources, metrics, sample_idxs, responses, prompts):
            metric.update({"sample_idx": sample_idx})
            metric.update({"response": response})
            metric.update({"prompt": self.llm.get_tokenizer().decode(prompt["prompt_token_ids"], skip_special_tokens=True)})
            data2metric[data_source].append(metric)
        for data_source in data2metric:
            data2metric[data_source] = self.reduce_metric(data2metric[data_source], data_source)

        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(data2metric, f, ensure_ascii=False, indent=4)

        for data_source in data2metric:
            metric_to_console = defaultdict(list)
            for idx in data2metric[data_source]:
                prompt = data2metric[data_source][idx].pop("prompt")
                response = data2metric[data_source][idx].pop("response")
                prediction = data2metric[data_source][idx].pop("prediction")
                for m in data2metric[data_source][idx]:
                    metric_to_console[m].append(np.mean(data2metric[data_source][idx][m]))
            pprint(
                {
                    f"{data_source}/{m}": np.mean(metric_to_console[m]) 
                    for m in metric_to_console
                }, 
            )


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
