import copy
import time
import traceback
import warnings
from typing import Any, Dict, List, Optional, Union, Sequence, cast
import threading
from threading import Thread
from concurrent.futures import Future
from dataclasses import dataclass

import asyncio
from tqdm import tqdm

from vllm import LLM
from vllm.inputs import PromptType, TokensPrompt, TextPrompt
from vllm.outputs import RequestOutput
from vllm.lora.request import LoRARequest
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.model_executor.guided_decoding.guided_fields import GuidedDecodingRequest, LLMGuidedOptions
from vllm.sampling_params import SamplingParams, RequestOutputKind
from vllm.distributed import parallel_state

from interfaces.base import InterfaceZoo, BaseInterface


def union_request_output(output: RequestOutput, other_output: RequestOutput) -> RequestOutput:
    assert len(output.outputs) == len(other_output.outputs)

    for index, completion_output in enumerate(other_output.outputs):
        output.outputs[index].text += completion_output.text
        output.outputs[index].token_ids += completion_output.token_ids
        
        if hasattr(output.outputs[index], 'cumulative_logprob') and output.outputs[index].cumulative_logprob is not None:
            if hasattr(completion_output, 'cumulative_logprob') and completion_output.cumulative_logprob is not None:
                output.outputs[index].cumulative_logprob += completion_output.cumulative_logprob
        
        if hasattr(output.outputs[index], 'logprobs') and output.outputs[index].logprobs is not None:
            if hasattr(completion_output, 'logprobs') and completion_output.logprobs is not None:
                output.outputs[index].logprobs += completion_output.logprobs

        output.outputs[index].finish_reason = completion_output.finish_reason
        output.outputs[index].stop_reason = completion_output.stop_reason
        
        if hasattr(completion_output, 'lora_request'):
            output.outputs[index].lora_request = completion_output.lora_request
    
    return output


class AsyncioEventLoop:
    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        loop = asyncio.new_event_loop()

        # Define a function to run the event loop in a separate thread
        def run_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        # Start a new thread to run the event loop
        Thread(target=run_loop, args=(loop,), daemon=True).start()
        self.loop = loop
    
    @classmethod
    def get_instance(cls, *args, **kwargs) -> "AsyncioEventLoop":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(*args, **kwargs)
        return cls._instance

@dataclass
class RequestInfo:
    request_id: str
    params: SamplingParams
    lora_request: Optional[LoRARequest]
    prompt_adapter_request: Optional[PromptAdapterRequest]
    priority: int
    input_extra_infos: Dict[str, Any]
    original_prompt_len: int
    invoke_num: dict[str, int]
    output: Optional[RequestOutput]

@dataclass
class InvocationOutput:
    perform_invocation: bool
    invocation_success: bool
    prefix_text: str
    prefix_token_ids: List[int]
    invocation_result_text: str
    invocation_result_token_ids: List[int]

    @property
    def success(self) -> bool:
        return (self.perform_invocation and self.invocation_success) or (not self.perform_invocation)

    @property
    def token_ids(self) -> List[int]:
        return self.prefix_token_ids + self.invocation_result_token_ids
    
    @property
    def text(self) -> str:
        return self.prefix_text + self.invocation_result_text
    
    @property
    def result_mask(self) -> List[int]:
        return [0] * len(self.prefix_token_ids) + [1] * len(self.invocation_result_token_ids)
    
@dataclass
class TaskOutput:
    continue_generation: bool
    request_id: str
    prompt: PromptType
    params: SamplingParams
    lora_request: Optional[LoRARequest]
    prompt_adapter_request: Optional[PromptAdapterRequest]
    priority: int
    output: RequestOutput
    elapsed: float


class SituatedThinkerLLM(LLM):
    def __init__(self, interface_zoo: InterfaceZoo, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interface_zoo = interface_zoo

    def generate(
        self,
        # Prompts can be a single prompt, a sequence of prompts, a string, or a list of strings
        prompts: Union[Union[PromptType, Sequence[PromptType]], Optional[Union[str, List[str]]]] = None,
        # Sampling parameters can be a single SamplingParams object or a sequence of them
        sampling_params: Optional[Union[SamplingParams, Sequence[SamplingParams]]] = None,
        # Token IDs of the prompts, can be a list of integers or a list of lists of integers
        prompt_token_ids: Optional[Union[List[int], List[List[int]]]] = None,
        # Whether to use tqdm progress bar during generation
        use_tqdm: bool = True,
        # LoRA requests can be a single LoRARequest object or a list of them
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        # Prompt adapter request
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        # Guided decoding options
        guided_options_request: Optional[Union[LLMGuidedOptions, GuidedDecodingRequest]] = None,
        # Priority of each request
        priority: Optional[List[int]] = None,
        # Extra information for each prompt
        extra_infos_of_prompts: Optional[List[Dict[str, Any]]] = None,
    ) -> List[RequestOutput]:
        # If sampling parameters are not provided, use the default sampling parameters.
        if sampling_params is None:
            # Use default sampling params.
            sampling_params = self.get_default_sampling_params()

        # Make a deep copy of the sampling parameters. If it's a sequence, copy each item.
        sampling_params = list(map(copy.deepcopy, sampling_params)) \
            if isinstance(sampling_params, Sequence) else copy.deepcopy(sampling_params)
        
        # Extract the 'n' value from each sampling parameter. 'n' represents the number of output sequences.
        n_list = (
            list(map(lambda x: getattr(x, "n"), sampling_params)) 
            if isinstance(sampling_params, Sequence) 
            else [sampling_params.n] * len(prompts)
        )
        # Modify the stop conditions and 'n' value in the sampling parameters.
        for _sampling_params in (
            sampling_params 
            if isinstance(sampling_params, Sequence) 
            else (sampling_params, )
        ):
            if _sampling_params.stop:
                # If the stop condition is already set, append the tool call end tags.
                _sampling_params.stop += self.interface_zoo.end_tags
            else:
                # If the stop condition is not set, set it to the tool call end tags.
                _sampling_params.stop = self.interface_zoo.end_tags
            # Set the number of output sequences to 1.
            _sampling_params.n = 1
            # Enable detokenization
            _sampling_params.detokenize = True

        # Get the runner type of the LLM engine
        runner_type = self.llm_engine.model_config.runner_type
        if runner_type not in ["generate", "transcription"]:
            messages = [
                "LLM.generate() is only supported for (conditional) generation "
                "models (XForCausalLM, XForConditionalGeneration).",
            ]

            supported_runner_types = self.llm_engine.model_config \
                .supported_runner_types
            if "generate" in supported_runner_types:
                messages.append(
                    "Your model supports the 'generate' runner, but is "
                    f"currently initialized for the '{runner_type}' runner. "
                    "Please initialize vLLM using `--task generate`.")

            # Raise an error if the runner type is not supported
            raise ValueError(" ".join(messages))

        if prompt_token_ids is not None:
            # Convert the input prompts and token IDs to the required format
            parsed_prompts = self._convert_v1_inputs(
                prompts=cast(Optional[Union[str, List[str]]], prompts),
                prompt_token_ids=prompt_token_ids,
            )
        else:
            # If token IDs are not provided, use the original prompts
            parsed_prompts = cast(Union[PromptType, Sequence[PromptType]],
                                  prompts)

        if isinstance(guided_options_request, dict):
            if len(guided_options_request) > 1:
                # Raise an error if multiple guided decoding options are specified
                raise ValueError(
                    "You can only use one guided decoding but multiple is "
                    f"specified: {guided_options_request}")
            # Convert the dictionary to a GuidedDecodingRequest object
            guided_options_request = GuidedDecodingRequest(
                **guided_options_request)

        def maybe_repeat(x, num_repeat: List[int]):
            """
            Repeat each item in x according to the corresponding value in num_repeat.
            If x is a single item, return it as is.

            Args:
                x: The item or sequence to repeat.
                num_repeat: A list of integers indicating the number of repetitions for each item in x.

            Returns:
                A list of repeated items if x is a sequence, otherwise return x as is.
            """
            # Check if x is a sequence (e.g., list, tuple)
            if isinstance(x, Sequence):
                # Ensure the length of x matches the length of num_repeat
                assert len(x) == len(num_repeat)
                # Repeat each item in x based on the corresponding value in num_repeat
                return [item for i, item in enumerate(x) for _ in range(num_repeat[i])]
            else:
                # If x is not a sequence, return it unchanged
                return x

        # Validate and add requests to the LLM engine.
        self._validate_and_add_requests(
            prompts=maybe_repeat(parsed_prompts, n_list),
            params=maybe_repeat(sampling_params, n_list),
            lora_request=maybe_repeat(lora_request, n_list),
            prompt_adapter_request=maybe_repeat(prompt_adapter_request, n_list),
            guided_options=maybe_repeat(guided_options_request, n_list),
            priority=maybe_repeat(priority, n_list),
            extra_infos_of_prompts=maybe_repeat(extra_infos_of_prompts, n_list)
        )
        
        # Run the LLM engine with tool calling support and get the flattened outputs.
        outputs = self._run_engine(use_tqdm=use_tqdm)

        # Validate the outputs and return them.
        return self.engine_class.validate_outputs(outputs, RequestOutput)
    
    def _validate_and_add_requests(
        self,
        prompts: Union[PromptType, Sequence[PromptType]],
        params: Union[SamplingParams, Sequence[SamplingParams]],
        lora_request: Optional[Union[Sequence[LoRARequest], LoRARequest]],
        prompt_adapter_request: Optional[PromptAdapterRequest],
        guided_options: Optional[GuidedDecodingRequest] = None,
        priority: Optional[List[int]] = None,
        extra_infos_of_prompts: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        if guided_options is not None:
            warnings.warn(
                "guided_options_request is deprecated, use "
                "SamplingParams.guided_decoding instead",
                DeprecationWarning,
                stacklevel=2,
            )

        if isinstance(prompts, (str, dict)):
            # Convert a single prompt to a list.
            prompts = [prompts]

        def _convert_prompt(prompt):
            """
            Convert the input prompt to token ids.

            Args:
                prompt (Union[dict, str, TextPrompt, Any]): The input prompt to be converted.

            Returns:
                Union[dict, TokensPrompt, Any]: The converted prompt.
            """
            tokenizer = self.get_tokenizer()
            # Check if the prompt is a dictionary
            if isinstance(prompt, dict):
                # Initialize an empty dictionary to store the converted prompt
                returns = {}
                # Iterate over each key-value pair in the dictionary
                for k in prompt:
                    # Check if the key is "prompts" and its value is a string or TextPrompt
                    if k == "prompts" and isinstance(prompt[k], (str, TextPrompt)):
                        # Encode the prompt text into token IDs without adding special tokens
                        returns["prompt_token_ids"] = tokenizer.encode(prompt[k], add_special_tokens=False)
                    else:
                        # Make a deep copy of the value and add it to the new dictionary
                        returns[k] = copy.deepcopy(prompt[k])
                return returns
            # Check if the prompt is a string or TextPrompt
            elif isinstance(prompt, (str, TextPrompt)):
                # If the prompt is a TextPrompt, access its 'prompt' attribute; otherwise, use the prompt directly
                text_to_encode = prompt.prompt if isinstance(prompt, TextPrompt) else prompt
                # Encode the text into token IDs and create a TokensPrompt object
                return TokensPrompt(prompt_token_ids=tokenizer.encode(text_to_encode, add_special_tokens=False))
            else:
                # If the prompt is neither a dictionary nor a string/TextPrompt, return it as is
                return prompt

        prompts = list(map(_convert_prompt, prompts))

        num_requests = len(prompts)
        if isinstance(params, list) and len(params) != num_requests:
            raise ValueError("The lengths of prompts and params "
                             "must be the same.")
        if isinstance(lora_request,
                      list) and len(lora_request) != num_requests:
            raise ValueError("The lengths of prompts and lora_request "
                             "must be the same.")

        for sp in params if isinstance(params, list) else (params, ):
            if isinstance(sp, SamplingParams):
                self._add_guided_params(sp, guided_options)

                # We only care about the final output
                sp.output_kind = RequestOutputKind.FINAL_ONLY

        # Add requests to the engine.
        self.request_infos: dict[str, RequestInfo] = {}
        for i, prompt in enumerate(prompts):
            self._add_request(
                prompt,
                params[i] if isinstance(params, Sequence) else params,
                lora_request=lora_request[i] if isinstance(
                    lora_request, Sequence) else lora_request,
                prompt_adapter_request=prompt_adapter_request,
                priority=priority[i] if priority else 0,
                extra_infos_of_prompt=extra_infos_of_prompts[i] \
                    if extra_infos_of_prompts else extra_infos_of_prompts,
            )
    
    def _add_request(
        self,
        # The input prompt, which can be of type PromptType
        prompt: PromptType,
        # Sampling parameters for the request
        params: SamplingParams,
        # Optional LoRA request for the model
        lora_request: Optional[LoRARequest] = None,
        # Optional prompt adapter request for the model
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        # Priority of the request, default is 0
        priority: int = 0,
        # Optional extra information associated with the prompt
        extra_infos_of_prompt: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a new request to the LLM engine and store its information.

        Args:
            prompt (PromptType): The input prompt for the LLM.
            params (SamplingParams): Sampling parameters for text generation.
            lora_request (Optional[LoRARequest]): Optional LoRA request for the model. Defaults to None.
            prompt_adapter_request (Optional[PromptAdapterRequest]): Optional prompt adapter request. Defaults to None.
            priority (int): Priority of the request. Defaults to 0.
            extra_infos_of_prompt (Optional[Dict[str, Any]]): Extra information about the prompt. Defaults to None.
        """
        # Generate a unique request ID by incrementing the request counter
        request_id = str(next(self.request_counter))
        # Create a RequestInfo object to store information about the request
        self.request_infos[request_id] = RequestInfo(
            request_id=request_id,
            # Make a deep copy of the sampling parameters to avoid modification issues
            params=copy.deepcopy(params),
            # Make a deep copy of the LoRA request to avoid modification issues
            lora_request=copy.deepcopy(lora_request),
            # Make a deep copy of the prompt adapter request to avoid modification issues
            prompt_adapter_request=copy.deepcopy(prompt_adapter_request),
            priority=priority,
            input_extra_infos=extra_infos_of_prompt,
            # Record the original length of the prompt
            original_prompt_len=len(prompt),
            # Initialize the invocation count for each interface end tag to 0
            invoke_num={end_tag: 0 for end_tag in self.interface_zoo.end_tags},
            output=None,
        )
        # Add the request to the LLM engine
        self.llm_engine.add_request(
            request_id,
            prompt,
            params,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            priority=priority,
        )

    def _run_engine(
        self, 
        *, 
        use_tqdm: bool,
    ) -> List[RequestOutput]:
        """
        Run the LLM engine, process requests, and handle tool calls asynchronously.

        Args:
            use_tqdm (bool): Whether to use the tqdm progress bar to display the processing progress.

        Returns:
            List[RequestOutput]: A list of final request outputs sorted by request ID.
        """
        # Get the number of unfinished requests in the LLM engine
        num_requests = self.llm_engine.get_num_unfinished_requests()

        # Initialize tqdm progress bar if use_tqdm is True
        if use_tqdm:
            pbar = tqdm(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} toks/s, "
                         f"output: {0:.2f} toks/s"),
                leave=False
            )

        # Initialize variables to store outputs and token counts
        outputs: List[RequestOutput] = []
        total_in_toks = 0
        total_out_toks = 0

        # Initialize a list to store tool call tasks
        invoke_tasks: List[Future] = []
        # Get the singleton instance of the asyncio event loop
        loop = AsyncioEventLoop.get_instance().loop

        # Check if any tool call tasks are completed
        completed_tasks = set()

        # Main loop to process requests until all are finished and all tool calls are completed
        while len(outputs) < num_requests:
            # Take a step in the LLM engine and get the outputs
            step_outputs = self.llm_engine.step()

            # Process each output from the current step
            for output in step_outputs:
                if output.finished:
                    # Check if the request finished due to an end-of-sequence token
                    if output.outputs[0].stop_reason not in self.interface_zoo.end_tags:
                        if self.request_infos[output.request_id].output is None:
                            final_output = output
                            # Add additional attributes to the final output
                            setattr(final_output, "result_mask", [0] * len(output.outputs[0].token_ids))
                            setattr(final_output, "over_invocation", False)
                            setattr(final_output, "over_long", False)
                            setattr(final_output, "failed_invocation", False)
                        else:
                            final_output = self.request_infos[output.request_id].output
                            # Merge the current output with the existing one
                            final_output = union_request_output(final_output, output)
                            final_output.result_mask += [0] * len(output.outputs[0].token_ids)

                        # Mask truncated response
                        if final_output.outputs[0].finish_reason != "stop":
                            final_output.overlong = True

                        # Add the output to the list of final outputs
                        outputs.append(final_output)
                        if use_tqdm:
                            if isinstance(final_output, RequestOutput):
                                # Calculate the number of input tokens
                                assert final_output.prompt_token_ids is not None
                                total_in_toks += len(final_output.prompt_token_ids)
                                in_spd = total_in_toks / pbar.format_dict["elapsed"]
                                # Calculate the number of output tokens
                                total_out_toks += sum(
                                    len(stp.token_ids) for stp in final_output.outputs)
                                out_spd = (total_out_toks /
                                           pbar.format_dict["elapsed"])
                                # Calculate the mean elapsed time of completed invocation tasks
                                invocation_mean_elapsed = (
                                    0.0 if len(completed_tasks) == 0 
                                    else sum([_task.elapsed for _task in completed_tasks]) / len(completed_tasks)
                                )
                                # Update the tqdm postfix with the estimated token speeds
                                pbar.postfix = (
                                    f"est. speed input: {in_spd:.2f} toks/s, "
                                    f"output: {out_spd:.2f} toks/s, "
                                    f"invocation task: {len(completed_tasks)} tsks, "
                                    f"invocation time: {invocation_mean_elapsed:.2f} s")
                            # Update the tqdm progress bar
                            pbar.update(1)
                    else:
                        # Get the extra information for the task
                        kwargs_for_task = self.request_infos[output.request_id].input_extra_infos
                        # Schedule the invoking coroutine in the event loop
                        task = asyncio.run_coroutine_threadsafe(
                            self._run_invoke_task(output, **kwargs_for_task), 
                            loop
                        )
                        # Add the task to the list of tasks
                        invoke_tasks.append(task)

            # Check the status of all tasks
            for task in invoke_tasks:
                if task.done() and task not in completed_tasks:
                    try:
                        # Get the result of the completed task
                        task_output: TaskOutput = task.result()
                        if task_output.continue_generation:
                            # Add the request back to the LLM engine if generation should continue
                            self.llm_engine.add_request(
                                task_output.request_id,
                                # task_output.prompt,
                                TokensPrompt(prompt_token_ids=task_output.prompt),
                                task_output.params,
                                lora_request=task_output.lora_request,
                                prompt_adapter_request=task_output.prompt_adapter_request,
                                priority=task_output.priority,
                            )
                        else:
                            # Add the output to the list of final outputs if generation is done
                            outputs.append(task_output.output)
                            if use_tqdm:
                                if isinstance(task_output.output, RequestOutput):
                                    # Calculate the number of input tokens
                                    assert task_output.output.prompt_token_ids is not None
                                    total_in_toks += len(task_output.output.prompt_token_ids)
                                    in_spd = total_in_toks / pbar.format_dict["elapsed"]
                                    # Calculate the number of output tokens
                                    total_out_toks += sum(
                                        len(stp.token_ids) for stp in task_output.output.outputs)
                                    out_spd = (total_out_toks /
                                               pbar.format_dict["elapsed"])
                                    # Calculate the mean elapsed time of completed invocation tasks
                                    invocation_mean_elapsed = (
                                        0.0 if len(completed_tasks) == 0 
                                        else sum([_task.elapsed for _task in completed_tasks]) / len(completed_tasks)
                                    )
                                    # Update the tqdm postfix with the estimated token speeds
                                    pbar.postfix = (
                                        f"est. speed input: {in_spd:.2f} toks/s, "
                                        f"output: {out_spd:.2f} toks/s, "
                                        f"invocation task: {len(completed_tasks)} tsks, "
                                        f"invocation time: {invocation_mean_elapsed:.2f} s")
                                # Update the tqdm progress bar
                                pbar.update(1)
                    except Exception as e:
                        import traceback
                        # Print the error message and traceback if the task fails
                        print(f"Invoke task failed: {e}\n\n{traceback.format_exc()}")
                        exit()

                    # Add the completed task to the set of completed tasks
                    completed_tasks.add(task)

        # Close the tqdm progress bar if it was used
        if use_tqdm:
            pbar.close()

        # Sort the outputs by request ID to ensure they are in order
        return sorted(outputs, key=lambda x: int(x.request_id))

    async def _run_invoke_task(
        self, 
        output: RequestOutput, 
        **kwargs
    ) -> TaskOutput:
        """
        Perform asynchronous tool calling based on the LLM output.

        Args:
            output (RequestOutput): The output of the LLM request.
            **kwargs: Additional keyword arguments.

        Returns:
            TaskOutput: An object containing information about whether to continue generation, 
            request ID, prompt, sampling parameters, LoRA request, prompt adapter request, 
            priority, and the final output.
        """
        # Record the start time to calculate the elapsed time later
        start_t = time.time()
        # Get the request ID from the LLM output
        request_id = output.request_id
        # Get the tokenizer instance
        tokenizer = self.get_tokenizer()

        # Get the tool instance based on the stop reason of the output
        interface = self.interface_zoo.get_interface_by_end_tag(output.outputs[0].stop_reason)
        # Extract the query from the output text
        query = interface.extract_query(output.outputs[0].text + interface.end_tag)

        async def _handle_interface_invocation(
            request_id: str,
            output: RequestOutput,
            interface: BaseInterface,
            query: str,
            interface_kwargs: dict,
        ) -> InvocationOutput:
            """
            Asynchronously handle the interface invocation based on the LLM output.

            Args:
                request_id (str): The unique identifier of the request.
                output (RequestOutput): The output of the LLM request.
                interface (BaseInterface): The interface instance to be invoked.
                query (str): The query extracted from the LLM output for the interface.
                interface_kwargs (dict): Additional keyword arguments for the interface invocation.

            Returns:
                InvocationOutput: An object containing information about the invocation, 
                including whether the invocation was performed, its success status, 
                prefix text, prefix token IDs, invocation result text, and invocation result token IDs.
            """
            # Generate the prefix text by appending the interface end tag to the output text
            prefix_text = f" {output.outputs[0].text + interface.end_tag} "
            
            # Check if the maximum invocation count for the interface has been reached
            if self.request_infos[request_id].invoke_num[interface.end_tag] >= interface.max_invoke_num:
                # Construct the result text indicating the maximum invocation count has been reached
                result_text = (
                    f"<result> "
                    f"this interface has been invoked for "
                    f"{self.request_infos[request_id].invoke_num[interface.end_tag]} "
                    f"times and cannot be invoked anymore! "
                    f"</result>"
                )
                # Return an InvocationOutput object indicating no invocation and failure
                return InvocationOutput(
                    perform_invocation=False,
                    invocation_success=False,
                    prefix_text=prefix_text,
                    prefix_token_ids=tokenizer.encode(prefix_text, add_special_tokens=False),
                    invocation_result_text=result_text,
                    invocation_result_token_ids=tokenizer.encode(result_text, add_special_tokens=False)
                )
    
            # Check if the query is None, indicating a format error
            if query is None:
                # Construct the result text indicating a format error
                result_text = (
                    f"<result> "
                    f"invocation cannot be parsed due to format error! "
                    f"</result>"
                )
                # Return an InvocationOutput object indicating invocation attempt and failure
                return InvocationOutput(
                    perform_invocation=True,
                    invocation_success=False,
                    prefix_text=prefix_text,
                    prefix_token_ids=tokenizer.encode(prefix_text, add_special_tokens=False),
                    invocation_result_text=result_text,
                    invocation_result_token_ids=tokenizer.encode(result_text, add_special_tokens=False)
                )
            
            try:
                # Asynchronously invoke the interface with the query and additional kwargs
                status, result = await interface.invoke(
                    query=query,
                    **interface_kwargs
                )
                # Construct the result text with the invocation result
                result_text = f"<result> {result} </result>"
                # Return an InvocationOutput object indicating invocation attempt and success status
                return InvocationOutput(
                    perform_invocation=True,
                    invocation_success=status == 1,
                    prefix_text=prefix_text,
                    prefix_token_ids=tokenizer.encode(prefix_text, add_special_tokens=False),
                    invocation_result_text=result_text,
                    invocation_result_token_ids=tokenizer.encode(result_text, add_special_tokens=False)
                )
            except asyncio.TimeoutError:
                # Construct the result text indicating a timeout error
                result_text = "<result> error when invoking: timeout </result>"
                # Print the timeout error message
                print(f"Invocation timeout for request {request_id}, interface: {interface.end_tag}")
                # Return an InvocationOutput object indicating invocation attempt and failure
                return InvocationOutput(
                    perform_invocation=True,
                    invocation_success=False,
                    prefix_text=prefix_text,
                    prefix_token_ids=tokenizer.encode(prefix_text, add_special_tokens=False),
                    invocation_result_text=result_text,
                    invocation_result_token_ids=tokenizer.encode(result_text, add_special_tokens=False)
                )
            except Exception as e:
                # Extract the last three lines of the traceback for error reporting
                error_msg = "\n".join(traceback.format_exc().split("\n")[-3:])
                # Construct the result text with the error message
                result_text = f"<result> error when invoking: {error_msg} </result>"
                # Print the invocation error message
                print(f"Invocation error for request {request_id}, interface: {interface.end_tag}, error: {error_msg}")
                # Return an InvocationOutput object indicating invocation attempt and failure
                return InvocationOutput(
                    perform_invocation=True,
                    invocation_success=False,
                    prefix_text=prefix_text,
                    prefix_token_ids=tokenizer.encode(prefix_text, add_special_tokens=False),
                    invocation_result_text=result_text,
                    invocation_result_token_ids=tokenizer.encode(result_text, add_special_tokens=False)
                )

        # Handle the interface invocation asynchronously
        invocation_output = await _handle_interface_invocation(
            request_id, output, interface, query, kwargs
        )
        
        # Update the prompt token IDs by appending the invocation output token IDs
        prompt_token_ids = output.prompt_token_ids + invocation_output.token_ids

        # Update the output text and token IDs with the invocation output
        output.outputs[0].text = invocation_output.text
        output.outputs[0].token_ids = invocation_output.token_ids

        if self.request_infos[request_id].output is None:
            # If the output for the request is not initialized, make a deep copy of the current output
            self.request_infos[request_id].output = copy.deepcopy(output)
            # Set additional attributes for the output
            setattr(self.request_infos[request_id].output, "result_mask", invocation_output.result_mask)
            setattr(self.request_infos[request_id].output, "over_invocation", False)
            setattr(self.request_infos[request_id].output, "failed_invocation", False)
            setattr(self.request_infos[request_id].output, "over_long", False)
        else:
            # If the output is already initialized, merge the current output with the existing one
            self.request_infos[request_id].output = union_request_output(
                self.request_infos[request_id].output, output)
            # Update the result mask
            self.request_infos[request_id].output.result_mask += invocation_output.result_mask

        # Decrease the maximum number of tokens based on the invocation output length
        self.request_infos[request_id].params.max_tokens -= len(invocation_output.token_ids)
        if self.request_infos[request_id].params.max_tokens <= 0:
            # Mark the output as over-long if the maximum tokens are exhausted
            self.request_infos[request_id].output.over_long = True

        # Increment the invocation count for the interface
        self.request_infos[request_id].invoke_num[interface.end_tag] += 1
        if self.request_infos[request_id].invoke_num[interface.end_tag] > interface.max_invoke_num:
            # Mark the output as over-invocation if the maximum invocation count is exceeded
            self.request_infos[request_id].output.over_invocation = True

        # Update the failed invocation status
        self.request_infos[request_id].output.failed_invocation |= (not invocation_output.success)

        if self.request_infos[request_id].output.over_long:
            # Return a TaskOutput indicating not to continue generation if the output is over-long
            return TaskOutput(
                continue_generation=False,
                request_id=request_id,
                prompt=prompt_token_ids,
                params=self.request_infos[request_id].params,
                lora_request=self.request_infos[request_id].lora_request,
                prompt_adapter_request=self.request_infos[request_id].prompt_adapter_request,
                priority=self.request_infos[request_id].priority,
                output=self.request_infos[request_id].output,
                elapsed=time.time() - start_t
            )

        # Return a TaskOutput indicating to continue generation
        return TaskOutput(
                continue_generation=True,
                request_id=request_id,
                prompt=prompt_token_ids,
                params=self.request_infos[request_id].params,
                lora_request=self.request_infos[request_id].lora_request,
                prompt_adapter_request=self.request_infos[request_id].prompt_adapter_request,
                priority=self.request_infos[request_id].priority,
                output=self.request_infos[request_id].output,
                elapsed=time.time() - start_t
            )

