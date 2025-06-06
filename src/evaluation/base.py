from argparse import Namespace
from abc import ABC, abstractclassmethod
from typing import List, Tuple, Dict, Any
from pathlib import Path
from functools import partial
from collections import defaultdict
import json

from datasets import Dataset
import numpy as np
from tqdm import tqdm
from pprint import pprint
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv("runtime.env"))

from misc.utils import check_hf_ckpt_prepared, convert_fsdp_to_hf, clean_fsdp
from interfaces.base import InterfaceZoo
from inference_engine.vllm import SituatedThinkerLLM, SamplingParams


class BaseEvaluator(ABC):
    def __init__(
        self, 
        checkpoint_dir: str,
        interface_zoo: InterfaceZoo,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        n: int,
        temperature: float,
        top_p: float,
        top_k: int,
        max_tokens: int,
        output_path: str
    ):
        hf_path = Path(checkpoint_dir).joinpath("actor", "huggingface")
        if not check_hf_ckpt_prepared(hf_path):
            print("Convert fsdp checkpoints to hf checkpoints")
            convert_fsdp_to_hf(hf_path.parent)
            clean_fsdp(hf_path.parent)

        self.llm = SituatedThinkerLLM(
            interface_zoo, 
            str(hf_path), 
            tensor_parallel_size=tensor_parallel_size, 
            gpu_memory_utilization=gpu_memory_utilization
        )

        if temperature < 0.1:
            n = 1
        self.n = n

        self.sampling_params = SamplingParams(
            n=1, 
            temperature=temperature, 
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens, 
            seed=42,
        )

        self.output_path = output_path

    @staticmethod
    def add_args(parser: Namespace):
        parser.add_argument("--checkpoint_dir", type=str, required=True)
        parser.add_argument("--tensor_parallel_size", type=int, required=True)
        parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
        parser.add_argument("--n", type=int, default=1)
        parser.add_argument("--temperature", type=float, default=0.0)
        parser.add_argument("--top_p", type=float, default=0.95)
        parser.add_argument("--top_k", type=int, default=40)
        parser.add_argument("--max_tokens", type=int, default=16384)
        parser.add_argument("--output_path", type=str, required=True)
        return parser

    @abstractclassmethod
    def load_data(self) -> Dataset:
        raise NotImplementedError
    
    @abstractclassmethod
    def score(self, response: str, ground_truths: List[str]) -> Tuple[int, Dict[str, Any]]:
        raise NotImplementedError
    
    def reduce_metric(self, metrics: List[Dict[str, Any]], data_source: str) -> Dict[str, Any]:
        idx2metric = defaultdict(list)
        for metric in metrics:
            sample_idx = metric.pop("sample_idx")
            idx2metric[sample_idx].append(metric)

        def list_of_dict_to_dict_of_list(list_of_dict: list[dict]):
            if len(list_of_dict) == 0:
                return {}
            keys = list_of_dict[0].keys()
            output = {key: [] for key in keys}
            for data in list_of_dict:
                for key, item in data.items():
                    assert key in output, print(key, output, data, data_source)
                    output[key].append(item)
            return output

        for idx in idx2metric:
            idx2metric[idx] = list_of_dict_to_dict_of_list(idx2metric[idx])

        return idx2metric

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
        answers = repeat_item(eval_dataset["answer"])
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
                ground_truths = data2metric[data_source][idx].pop("ground_truths")
                for m in data2metric[data_source][idx]:
                    metric_to_console[m].append(np.mean(data2metric[data_source][idx][m]))
            pprint(
                {
                    f"{data_source}/{m}": np.mean(metric_to_console[m]) 
                    for m in metric_to_console
                }, 
            )
