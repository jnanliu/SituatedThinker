import copy
import importlib
import asyncio
import heapq
from typing import Any, Dict, List, Callable, Union
from uuid import uuid4
from dataclasses import dataclass
import traceback

from cachetools import LRUCache
from omegaconf import DictConfig
import torch
from tensordict import TensorDict
from openai import AsyncOpenAI
from openai.types.completion import Completion
import numpy as np

from verl.protocol import DataProto
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.protocol import dict_of_list_to_list_of_dict

from interfaces.base import InterfaceZoo


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


@dataclass
class CompletionOutput:
    prompt: str
    token_ids: List[int]
    result_mask: List[int]
    invoke_num: Dict[str, int]
    over_invocation: bool
    failed_invocation: bool
    over_long: bool

class SituatedThinkerCompletionScheduler:
    def __init__(
        self,
        config: DictConfig,
        model_path: str,
        server_addresses: List[str],
        max_cache_size: int = 10000,
    ):
        """
        Args:
            config: DictConfig, rollout config.
            model_path: str, model path.
            server_addresses: List[str], server addresses.
            max_cache_size: int, max cache size of request_id to address mapping.
        """
        self.config = config
        self.model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(model_path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=True)

        # Least requests load balancing
        self.weighted_addresses = [[0, address] for address in server_addresses]
        heapq.heapify(self.weighted_addresses)

        # LRU cache to map request_id to address
        self.request_id_to_address = LRUCache(maxsize=max_cache_size)

        module_path, var_name = self.config.interface_zoo.rsplit(".", 1)
        module = importlib.import_module(module_path)
        self.interface_zoo: InterfaceZoo = getattr(module, var_name)

        self.completion_sem = asyncio.Semaphore(256)

    async def submit_completions(
        self,
        callback: Callable[[Completion, Dict[str, Any], Exception], None],
        callback_additional_info: Dict[str, Any],
        **completion_request,
    ):       
        """
        Submit a chat completion request to the server with the least number of requests.

        Args:
            callback: Callable[[Completion, Dict[str, Any], Exception], None], async callback function
                to handle the response. The callback function should have the following signature:

                ```python
                async def callback(completions: Completion, info: Dict[str, Any], exception: Exception):
                    ...
                ```
                - completions: completion response from server.
                - info: user provided `callback_additional_info`.
                - exception: exception raise from OpenAI client if request failed, otherwise None.

                **CAUTION**: the callback function must be async and non-blocking, if you have any blocking operation,
                please move to seperate thread or process pool to avoid blocking the event loop.

            callback_additional_info: Dict[str, Any], additional info to pass to the callback function.

            **completion_request: dict, request parameters same as OpenAI AsyncCompletions.create.
                OpenAI API reference: https://platform.openai.com/docs/api-reference/completions/create
        """
        if "extra_headers" not in completion_request:
            completion_request["extra_headers"] = {}

        extra_headers = completion_request["extra_headers"]
        request_id = extra_headers.get("x-request-id", None)
        if request_id:
            if request_id.startswith("chatcmpl-"):
                request_id = request_id[len("chatcmpl-") :]
                extra_headers["x-request-id"] = request_id

            address = self.request_id_to_address.pop(request_id)
        else:
            address = self.weighted_addresses[0][1]
            self.weighted_addresses[0][0] += 1
            heapq.heapreplace(self.weighted_addresses, self.weighted_addresses[0])

        # use new request_id to avoid duplicate request_id problem
        request_id = uuid4().hex
        self.request_id_to_address[request_id] = address
        completion_request["extra_headers"]["x-request-id"] = request_id

        completions, exception = None, None
        try:
            # TODO: OpenAI client uses httpx, seems to have performance issue in high concurrency requests.
            async with self.completion_sem:
                completions = await self._openai_completion(address, **completion_request)
        except Exception as e:
            # Let user handle the exception
            exception = e

        await callback(completions, callback_additional_info, exception)

    async def _openai_completion(self, address: str, **completion_request) -> Completion:
        client = AsyncOpenAI(
            base_url=f"http://{address}/v1",
            api_key="token-abc123",
            timeout=None
        )
        return await client.completions.create(**completion_request)

    async def generate_sequences(self, batch: DataProto, **sampling_params) -> DataProto:
        kwargs = dict(
            n=self.config.n,
            max_tokens=self.config.response_length,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

        do_sample = batch.meta_info.get("do_sample", True)
        is_validate = batch.meta_info.get("validate", False)
        if not do_sample:
            kwargs["n"] = 1
            kwargs["best_of"] = 1
            kwargs["temperature"] = 0
            kwargs["top_p"] = 1.0
        elif is_validate:
            kwargs["n"] = 1
            kwargs["best_of"] = 1
            kwargs["temperature"] = self.config.val_kwargs.temperature
            kwargs["top_p"] = self.config.val_kwargs.top_p
        kwargs["stop"] = self.interface_zoo.end_tags
        if self.tokenizer.eos_token:
            kwargs["stop"].append(self.tokenizer.eos_token)

        kwargs.update(sampling_params)
        print(f"[SituatedThinkerScheduler] generate_sequences sampling params: {kwargs}")

        interface_kwargs = dict_of_list_to_list_of_dict(batch.non_tensor_batch)
        interface_kwargs = [interface_kwargs[i] for i in range(len(interface_kwargs)) for _ in range(kwargs["n"])]

        completion_requests = []
        for i in range(len(batch.non_tensor_batch["raw_prompt_ids"])):
            prompt_ids = batch.non_tensor_batch["raw_prompt_ids"][i]
            for _ in range(kwargs["n"]):
                additional_kwargs = copy.deepcopy(kwargs)
                additional_kwargs.update({"n": 1})
                completion_requests.append({
                    "prompt": self.tokenizer.decode(prompt_ids),
                    "model": self.model_name,
                    **additional_kwargs
                })

        completion_outputs: List[CompletionOutput] = [None] * len(completion_requests)

        async def callback(completion_obj: Completion, info: Dict[str, Any], exception: Exception):
            if exception is not None:
                raise exception
            index = info["index"]
            output: CompletionOutput = info["output"]
            request = info["request"]

            choice = completion_obj.choices[0]
            token_ids = self.tokenizer.encode(choice.text, add_special_tokens=False)
            output.token_ids.extend(token_ids)
            output.result_mask.extend([0] * len(token_ids))
            if choice.finish_reason == "stop":
                query = None
                interface = None
                lpos = [choice.text.rfind(start_tag) for start_tag in self.interface_zoo.start_tags]
                rpos = [choice.text.rfind(end_tag) for end_tag in self.interface_zoo.end_tags]
                valid = [l > r for l, r in zip(lpos, rpos)]
                valid_eng_tags = [(l, end_tag) for l, end_tag, v in zip(lpos, self.interface_zoo.end_tags, valid) if v]
                valid_eng_tags = sorted(valid_eng_tags, key=lambda x: x[0], reverse=True)
                if len(valid_eng_tags) > 0:
                    _, end_tag = valid_eng_tags[0]
                    interface = self.interface_zoo.get_interface_by_end_tag(end_tag)
                    query = interface.extract_query(choice.text + end_tag)
                    end_tag_token_ids = self.tokenizer.encode(end_tag, add_special_tokens=False)
                    output.token_ids.extend(end_tag_token_ids)
                    output.result_mask.extend([0] * len(end_tag_token_ids))
                # if pos[-1] == -1:
                #     interface = None
                # for end_tag in self.interface_zoo.end_tags:
                #     if choice.text.endswith(end_tag):
                #         interface = self.interface_zoo.get_interface_by_end_tag(end_tag)
                #         query = interface.extract_query(choice.text)
                #         break

                if interface is None:
                    completion_outputs[index] = output
                    return
                
                if query is None:
                    output.failed_invocation = True
                    result_text = (
                        f"<result> "
                        f"invocation cannot be parsed due to format error! "
                        f"</result>"
                    )
                    result_token_ids = self.tokenizer.encode(result_text, add_special_tokens=False)
                    output.token_ids.extend(result_token_ids)
                    output.result_mask.extend([1] * len(result_token_ids))
                elif output.invoke_num[end_tag] >= interface.max_invoke_num:
                    output.over_invocation = True
                    result_text = (
                        f"<result> "
                        f"this interface has been invoked for "
                        f"{output.invoke_num[end_tag]} "
                        f"times and cannot be invoked anymore! "
                        f"</result>"
                    )
                    result_token_ids = self.tokenizer.encode(result_text, add_special_tokens=False)
                    output.token_ids.extend(result_token_ids)
                    output.result_mask.extend([1] * len(result_token_ids))
                else:
                    try:
                        status, result = await interface.invoke(
                            query=query,
                            **interface_kwargs[index]
                        )
                        result_text = f"<result> {result} </result>"
                        output.failed_invocation = status == 0
                        result_token_ids = self.tokenizer.encode(result_text, add_special_tokens=False)
                        output.token_ids.extend(result_token_ids)
                        output.result_mask.extend([1] * len(result_token_ids))
                    except asyncio.TimeoutError:
                        result_text = "<result> error when invoking: timeout </result>"
                        print(f"Invocation timeout for request {index}, interface: {interface.end_tag}")
                        result_token_ids = self.tokenizer.encode(result_text, add_special_tokens=False)
                        output.token_ids.extend(result_token_ids)
                        output.result_mask.extend([1] * len(result_token_ids))
                        output.failed_invocation = True
                    except Exception as e:
                        error_msg = "\n".join(traceback.format_exc().split("\n")[-3:])
                        result_text = f"<result> error when invoking: {error_msg} </result>"
                        print(f"Invocation error for request {index}, interface: {interface.end_tag}, error: {error_msg}")
                        result_token_ids = self.tokenizer.encode(result_text, add_special_tokens=False)
                        output.token_ids.extend(result_token_ids)
                        output.result_mask.extend([1] * len(result_token_ids))
                        output.failed_invocation = True

                next_request = copy.deepcopy(request)
                next_request["prompt"] = output.prompt + self.tokenizer.decode(output.token_ids)
                next_request["max_tokens"] = kwargs["max_tokens"] - len(output.token_ids)
                if next_request["max_tokens"] <= 0:
                    output.over_long = True
                output.invoke_num[end_tag] += 1
                if output.invoke_num[end_tag] > interface.max_invoke_num:
                    output.over_invocation = True
                if not output.over_long:
                    await self.submit_completions(
                        callback=callback,
                        callback_additional_info={
                            "index": index,
                            "output": output,
                            "request": request
                        },
                        **next_request
                    )
                else:
                    completion_outputs[index] = output
            elif choice.finish_reason == "length":
                output.over_long = True
                completion_outputs[index] = output
            else:
                completion_outputs[index] = output

        tasks = []
        for index, request in enumerate(completion_requests):
            tasks.append(
                asyncio.create_task(
                    self.submit_completions(
                        callback=callback,
                        callback_additional_info={
                            "index": index,
                            "output": CompletionOutput(
                                prompt=request["prompt"],
                                token_ids=[],
                                result_mask=[],
                                invoke_num={end_tag: 0 for end_tag in self.interface_zoo.end_tags},
                                over_invocation=False,
                                failed_invocation=False,
                                over_long=False,
                            ),
                            "request": request
                        },
                        **request
                    )
                )
            )
        await asyncio.gather(*tasks)
        print("[SituatedThinkerCompletionScheduler] generate_sequences done")

        return self._postprocess(batch, completion_outputs, kwargs["n"])

    def _postprocess(self, batch: DataProto, completion_outputs: List[CompletionOutput], n: int) -> DataProto:
        # NOTE: consistent with batch version of generate_sequences in vllm_rollout_spmd.py
        # prompts: left pad
        # responses: right pad
        # input_ids: prompt + response
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]

        prompts = batch.batch["input_ids"]
        prompt_attention_mask = batch.batch['attention_mask']

        non_tensor_batch = batch.non_tensor_batch

        assert len(prompts) * n == len(completion_outputs)

        responses = [
            completion_output.token_ids[:self.config.response_length]
            for completion_output in completion_outputs
        ]
        response_result_mask = [
            completion_output.result_mask[:self.config.response_length] 
            for completion_output in completion_outputs
        ]
        over_invocation = [
            completion_output.over_invocation
            for completion_output in completion_outputs
        ]
        failed_invocation = [
            completion_output.failed_invocation
            for completion_output in completion_outputs
        ]
        over_long = [
            completion_output.over_long
            for completion_output in completion_outputs
        ]

        responses = pad_2d_list_to_length(responses, self.tokenizer.pad_token_id, self.config.response_length).to(prompts.device)
        response_result_mask = pad_2d_list_to_length(response_result_mask, 0, self.config.response_length).to(prompts.device)
        response_attention_mask = get_response_mask(response_id=responses, eos_token=self.tokenizer.eos_token_id, dtype=prompt_attention_mask.dtype)
    
        if n > 1:
            prompts = _repeat_interleave(prompts, n)
            prompt_attention_mask = _repeat_interleave(prompt_attention_mask, n)
            for k in non_tensor_batch:
                non_tensor_batch[k] = _repeat_interleave(non_tensor_batch[k], n)

        input_ids = torch.cat([prompts, responses], dim=1)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1)
        position_ids = (attention_mask.cumsum(dim=1) - 1) * attention_mask

        batch = TensorDict(
            {
                "prompts": prompts,
                "responses": responses,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "result_mask": response_result_mask,
            },
            batch_size=len(input_ids),
        )
        non_tensor_batch["over_invocation"] = np.array(over_invocation, dtype=object)
        non_tensor_batch["failed_invocation"] = np.array(failed_invocation, dtype=object)
        non_tensor_batch["over_long"] = np.array(over_long, dtype=object)

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)