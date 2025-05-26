from collections import defaultdict

from tqdm import tqdm
import torch
from transformers import PreTrainedTokenizer

from verl import DataProto 
from verl.protocol import DataProtoItem
from verl.utils.reward_score.situated_thinker import Scorer


class SituatedThinkerRewardManager:
    """The reward manager."""

    def __init__(
        self, 
        tokenizer: PreTrainedTokenizer, 
        num_examine: int,
        max_response_length: int,
        is_validate: bool = False
    ):
        """
        Initialize an instance of the SituatedThinkerRewardManager class.

        Args:
            tokenizer (PreTrainedTokenizer): A tokenizer used to decode token IDs into text strings.
            num_examine (int): The number of batches of decoded responses to print to the console.
            max_response_length (int): The maximum length of the response, used to calculate the length penalty score.
            is_validate (bool, optional): A boolean indicating whether the validation mode is enabled. Defaults to False.
        """
        self.tokenizer = tokenizer
        # The number of batches of decoded responses to print to the console
        self.num_examine = num_examine  
        self.max_response_length = max_response_length
        self.is_validate = is_validate

    def _score_one_sample(self, idx: int, data_item: DataProtoItem):
        """
        Process a single data sample and compute the reward score.

        Args:
            idx (int): The index of the current data sample.
            data_item (DataProtoItem): An instance of DataProtoItem containing the data sample.

        Returns:
            dict: A dictionary containing the index, prompt, valid prompt length, response, 
                  valid response length, ground truth, data source, data type, and score dictionary.
        """
        # Extract prompt IDs from the current data item
        prompt_ids = data_item.batch['prompts']
        # Get the length of the prompt
        prompt_length = prompt_ids.shape[-1]

        # Calculate the valid length of the prompt using the attention mask
        valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
        # Extract the valid part of the prompt IDs
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]

        # Extract response IDs from the current data item
        response_ids = data_item.batch['responses']
        # Calculate the valid length of the response using the attention mask
        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum().item()
        # Extract the valid part of the response IDs
        valid_response_ids = response_ids[:valid_response_length]

        # Decode the valid prompt IDs to a string
        prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
        # Decode the valid response IDs to a string
        response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

        # Extract the ground truth from the non-tensor batch
        ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

        # Extract the data source from the non-tensor batch
        data_source = data_item.non_tensor_batch['data_source']
        # Extract the data type from the non-tensor batch
        data_type = data_item.non_tensor_batch['data_type']

        # Initialize the scorer based on the data type and get the score computation function
        compute_score_fn = Scorer(data_type, self.is_validate).compute_score
        # Compute the score dictionary for the current response and ground truth
        score_dict = compute_score_fn(response_str, ground_truth)

        # Count the number of query scores to determine if there's no invocation
        invocation_count = 0
        for k, v in score_dict.items():
            if k.endswith("invocation_score"):
                invocation_count += v
        # Calculate the no-invocation penalty score
        no_invocation_penalty_reward = -1.0 if (
            invocation_count == 0 and (score_dict["accuracy_score"] <= 0)
        ) else 0.0

        # Calculate the over-invocation penalty score
        over_invocation_penalty_reward = -1.0 if data_item.non_tensor_batch["over_invocation"] else 0.0

        # Calculate the invocation failure penalty score
        failed_invocation_penalty_reward = -1.0 if data_item.non_tensor_batch["failed_invocation"] else 0.0

        # Calculate the length penalty score based on the response length
        if valid_response_length <= self.max_response_length - 2048:
            length_penalty_reward = 0.0
        else:
            length_penalty_reward = (self.max_response_length - 2048 - valid_response_length) / 2048

        # Get the accuracy score from the score dictionary
        reward = score_dict["accuracy_score"]
        # Adjust the reward if the format score is 0 and accuracy score is 0
        if not self.is_validate and score_dict["format_score"] == 0 and reward == 0: 
            reward = -0.1

        # Update the score dictionary with penalty rewards and the final reward
        score_dict.update({
            "no_invocation_penalty_reward": no_invocation_penalty_reward,
            "over_invocation_penalty_reward": over_invocation_penalty_reward,
            "failed_invocation_penalty_reward": failed_invocation_penalty_reward,
            "length_penalty_reward": length_penalty_reward,
            "reward": reward 
        })

        return {
            "idx": idx,
            "prompt": prompt_str,
            "valid_prompt_length": valid_prompt_length,
            "response": response_str,
            "valid_response_length": valid_response_length,
            "ground_truth": ground_truth,
            "data_source": data_source,
            "data_type": data_type,
            "score_dict": score_dict
        }

    def __call__(self, data: DataProto, return_dict: bool = True) -> torch.Tensor:
        """
        Compute reward scores for the given data. We will expand this function gradually based on the available datasets.

        Args:
            data (DataProto): An instance of DataProto containing the data to compute rewards for.

        Returns:
            dict: A dictionary containing the computed reward tensor and detailed score information for each data item.
        """

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        # Initialize the reward tensor with zeros, having the same shape as the responses batch
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        # Dictionary to keep track of the number of data sources already printed
        already_print_data_sources = {}
        # List to store detailed score information for each data item
        reward_extra_info_list = [None for _ in range(len(data))]

        # Iterate over each data item in the dataset with a progress bar
        for idx, data_item in tqdm(enumerate(data), total=len(data), desc="Rewarding", disable=True):
            # Process a single data sample and get the result
            result = self._score_one_sample(idx, data_item)
            # Extract relevant information from the result
            idx = result["idx"]
            prompt_str = result["prompt"]
            valid_prompt_length = result["valid_prompt_length"]
            response_str = result["response"]
            valid_response_length = result["valid_response_length"]
            ground_truth = result["ground_truth"]
            data_source = result["data_source"]
            data_type = result["data_type"]
            score_dict = result["score_dict"]
            
            # Assign the score to the last position of the valid response in the reward tensor
            reward_tensor[idx, valid_response_length - 1] = score_dict["reward"]

            # If the data source is not in the printed list, initialize its count to 0
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            # If the number of printed data sources is less than the number to examine
            if already_print_data_sources[data_source] < self.num_examine:
                # Increment the count of printed data sources
                already_print_data_sources[data_source] += 1
                # Print the prompt, response, ground truth, and score details
                print("[prompt]\n\n", prompt_str)
                print("[response]\n\n", response_str)
                print("[ground_truth]\n\n", ground_truth)
                for k, v in score_dict.items():
                    print(f"[{k}]\n\n", v)

            # Store the score details for the current data item
            reward_extra_info_list[idx] = score_dict

        # Convert the list of dictionaries to a dictionary of lists
        reward_extra_info = defaultdict(list)
        for item in reward_extra_info_list:
            for key, value in item.items():
                reward_extra_info[key].append(value)

        # If return_dict is True, return the reward tensor and extra details
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info
            }
        # Otherwise, return only the reward tensor
        else:
            return reward_tensor
