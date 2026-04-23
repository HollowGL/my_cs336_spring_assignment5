import json
import os
import random
from typing import Callable, Literal

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import SamplingParams
import wandb

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn_gsm8k, r1_zero_reward_fn
r1_zero_reward_fn = r1_zero_reward_fn_gsm8k  # 适用于gsm8k的版本
from cs336_alignment.eval import evaluate_vllm, extract_reference_answer, load_and_format_prompts
from cs336_alignment.grpo import compute_group_normalized_rewards, grpo_microbatch_train_step, masked_mean
from cs336_alignment.train_sft import VLLMServerProxy, sync_model_to_vllm_server
from cs336_alignment.utils import get_response_log_probs, tokenize_prompt_and_output

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

QWEN_MATH_BASE_PATH = "/root/autodl-tmp/assignment5-alignment/Qwen2.5-Math-1.5B"
PROMPT_PATH = "/root/autodl-tmp/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
MATH_DATA_PATH = "/root/autodl-tmp/assignment5-alignment/data/gsm8k"
OUTPUT_PATH = "/root/autodl-tmp/assignment5-alignment/output"
VLLM_SYNC_PATH = os.path.join(OUTPUT_PATH, "_vllm_eval_ckpt")

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

device_train = "cuda:0"
device_vllm = "cuda:1"


n_grpo_steps = 48
grpo_eval_freq = 4
# n_grpo_steps = 4
# grpo_eval_freq = 1

advantage_eps = 1e-6
rollout_batch_size = 64           # 一个grpo step一共多少训练多少样本
group_size = 8                    # 每个 prompt 生成 group_size 个 response，一共 rollout_batch_size // group_size 个 prompt

sampling_temperature = 1.0
sampling_top_p = 0.90
sampling_min_tokens = 4
sampling_max_tokens = 384

epochs_per_rollout_batch = 1      # 1: on policy  > 1: off policy, the same rollout batch will be reused for multiple epochs
learning_rate = 3e-5 / epochs_per_rollout_batch
train_batch_size = 64             # train_step = rollout_batch_size // train_batch_size
gradient_accumulation_steps = 8   # micro_train_batch_size = train_batch_size // gradient_accumulation_steps
gpu_memory_utilization = 0.95
loss_type: Literal[
    "no_baseline",
    "reinforce_with_baseline",
    "grpo_clip",
] = "grpo_clip"
use_std_normalization = False
cliprange = 0.2
grpo_num_eval_samples = 256


def train_grpo():
    assert train_batch_size % gradient_accumulation_steps == 0, "train_batch_size must be divisible by gradient_accumulation_steps"
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert rollout_batch_size % group_size == 0, "rollout_batch_size must be divisible by group_size"
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert train_batch_size >= group_size, "train_batch_size must be greater than or equal to group_size"
    n_microbatches_per_rollout_batch = rollout_batch_size // micro_train_batch_size

    wandb.init(project="cs336-grpo",
        name=f"grpo_{loss_type}_lr_{learning_rate}_step_{n_grpo_steps}_stdnorm_{use_std_normalization}",
        config={
            "n_grpo_steps": n_grpo_steps,
            "epochs_per_rollout_batch": epochs_per_rollout_batch,
            "rollout_batch_size": rollout_batch_size,
            "group_size": group_size,
        }
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=QWEN_MATH_BASE_PATH,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to(device_train)  # type: ignore
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(QWEN_MATH_BASE_PATH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0, betas=(0.9, 0.95))


    vllm_server = VLLMServerProxy(
        model_path=QWEN_MATH_BASE_PATH,
        device=device_vllm,
        gpu_memory_utilization=0.9,
    )

    train_prompts, train_answers = load_and_format_prompts(MATH_DATA_PATH + "/train.jsonl", PROMPT_PATH)   # the answers have not been extracted
    test_prompts, test_answers = load_and_format_prompts(MATH_DATA_PATH + "/test.jsonl", PROMPT_PATH)      # the answers will br extracted during `evaluate_vllm`
    test_prompts, test_answers = test_prompts[:grpo_num_eval_samples], test_answers[:grpo_num_eval_samples]
    train_data = [{"prompt": prompt, "answer": extract_reference_answer(answer)} for prompt, answer in zip(train_prompts, train_answers)]


    for grpo_step in range(n_grpo_steps):
        rollout_dataset = random.sample(train_data, n_prompts_per_rollout_batch)
        rollout_prompts = [data["prompt"] for data in rollout_dataset]
        rollout_answers = [data["answer"] for data in rollout_dataset]
        
        sampling_params = SamplingParams(
            seed=SEED,
            n=group_size, 
            temperature=sampling_temperature, top_p=sampling_top_p, 
            max_tokens=sampling_max_tokens, min_tokens=sampling_min_tokens,
            stop=["</answer>"], include_stop_str_in_output=True,
        )
        
        outputs = vllm_server.generate(rollout_prompts, sampling_params)

        repeated_answers = []
        responses = []
        prompts = []

        for output, answer in zip(outputs, rollout_answers):
            prompt = output.prompt          # type: ignore
            for rollout in output.outputs:  # type: ignore
                responses.append(rollout.text)
                repeated_answers.append(answer)
                prompts.append(prompt)
        tokenizations = tokenize_prompt_and_output(prompts, responses, tokenizer)

        input_ids = tokenizations["input_ids"].to(device_train)
        labels = tokenizations["labels"].to(device_train)
        response_mask = tokenizations["response_mask"].to(device_train)

        print (f"response_mask.shape: {response_mask.shape}")
        advantages_train, raw_rewards_train, reward_metadata = compute_group_normalized_rewards(
            reward_fn=r1_zero_reward_fn,
            rollout_responses=responses,
            repeated_ground_truths=repeated_answers,
            group_size=group_size,
            advantage_eps=advantage_eps,
            normalize_by_std=use_std_normalization,
        )
        advantages_train = advantages_train.to(device_train)
        raw_rewards_train = raw_rewards_train.to(device_train)

        print ("---------examples of prompt, response, answer-----------")
        for i in range(3):
            print(f"prompt:{prompts[i]}    " + 
            f"\033[4;34m rollouts:{responses[i]}\033[0m    " +
            f"answers:{repeated_answers[i]}    " +
            f"reward:{raw_rewards_train[i]}")
            print(f"reward_metadata: {reward_metadata}")
        print ("--------grpo step rollout example done")
        wandb.log({
            "sampling/avg_reward": reward_metadata["mean"],
        }, step=grpo_step)

        num_train_steps_per_epoch = rollout_batch_size // train_batch_size
        with torch.no_grad():
            old_log_probs_train = []
            for train_step in range(num_train_steps_per_epoch):
                batch_ids = train_step * train_batch_size, (train_step + 1) * train_batch_size
                for train_micro_step in range(gradient_accumulation_steps):
                    micro_batch_ids = batch_ids[0] + train_micro_step * micro_train_batch_size, batch_ids[0] + (train_micro_step + 1) * micro_train_batch_size
                    micro_input_ids = input_ids[micro_batch_ids[0]:micro_batch_ids[1]]
                    micro_labels = labels[micro_batch_ids[0]:micro_batch_ids[1]]
                    micro_response_mask = response_mask[micro_batch_ids[0]:micro_batch_ids[1]]

                    log_probs_dict = get_response_log_probs(model, micro_input_ids, micro_labels, return_token_entropy=False)
                    log_probs = log_probs_dict["log_probs"]
                    old_log_probs_train.append(log_probs.cpu())
                    assert log_probs.shape[0] == micro_batch_ids[1] - micro_batch_ids[0]
            old_log_probs_train = torch.cat(old_log_probs_train)
        print (f"grpo step {grpo_step}: complete computing log probs on the old model, old_log_probs.shape={old_log_probs_train.shape}")


        # train new model
        for train_epoch in range(epochs_per_rollout_batch):
            for train_step in range(num_train_steps_per_epoch):
                batch_ids = train_step * train_batch_size, (train_step + 1) * train_batch_size
                accumulated_token_entropy, accumulated_clip_fraction = 0, 0

                batch_response_mask = response_mask[batch_ids[0]:batch_ids[1]]
                batch_mean_response_length = batch_response_mask.sum(dim=-1).mean(dtype=torch.float32)
                for train_micro_step in range(gradient_accumulation_steps):
                    micro_batch_ids = batch_ids[0] + train_micro_step * micro_train_batch_size, batch_ids[0] + (train_micro_step + 1) * micro_train_batch_size
                    raw_rewards = raw_rewards_train[micro_batch_ids[0]:micro_batch_ids[1]].unsqueeze(-1)
                    advantages = advantages_train[micro_batch_ids[0]:micro_batch_ids[1]].unsqueeze(-1)
                    old_log_probs = old_log_probs_train[micro_batch_ids[0]:micro_batch_ids[1]].to(device_train)
                    micro_input_ids = input_ids[micro_batch_ids[0]:micro_batch_ids[1]]
                    micro_labels = labels[micro_batch_ids[0]:micro_batch_ids[1]]
                    micro_response_mask = response_mask[micro_batch_ids[0]:micro_batch_ids[1]]

                    log_probs_dict = get_response_log_probs(model, micro_input_ids, micro_labels, return_token_entropy=False)
                    log_probs = log_probs_dict["log_probs"]
                    # token_entropy = log_probs_dict["token_entropy"]

                    policy_log_probs = log_probs
                    loss, train_metadata = grpo_microbatch_train_step(
                        policy_log_probs=policy_log_probs,
                        response_mask=micro_response_mask,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                        loss_type=loss_type,
                        raw_rewards=raw_rewards,
                        advantages=advantages,
                        old_log_probs=old_log_probs,
                        cliprange=cliprange,
                    )
                    print (f"train: grpo step {grpo_step}, train epoch {train_epoch}, train step {train_step}, micro batch step {train_micro_step}, loss is {loss:.6f}")
                    # avg_token_entropy = masked_mean(token_entropy, micro_response_mask)
                    # accumulated_token_entropy += avg_token_entropy.item()
                    if loss_type == "grpo_clip":
                        accumulated_clip_fraction += masked_mean(train_metadata['clipped'], micro_response_mask).item()

                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

                wandb.log({
                    "train/loss": loss.item() * gradient_accumulation_steps,
                    # "train/avg_token_entropy": accumulated_token_entropy / gradient_accumulation_steps,
                    "train/avg_clip_fraction": accumulated_clip_fraction / gradient_accumulation_steps,
                    "train/grad_norm": grad_norm,
                    "train/mean_response_length": batch_mean_response_length
                }, step=grpo_step)

        sync_model_to_vllm_server(model, tokenizer, vllm_server, VLLM_SYNC_PATH)

        # eval
        if grpo_step % grpo_eval_freq == 0:
            print(f"\033[32m--- grpo step {grpo_step}: start evaluation ---\033[0m")  # green color
            model.eval()
            sampling_params = SamplingParams(
                seed=SEED, temperature=sampling_temperature, top_p=sampling_top_p, max_tokens=1024, stop=["</answer>"], include_stop_str_in_output=True
            )

            overview, allinfo_dict_list = evaluate_vllm(
                vllm_server, r1_zero_reward_fn, test_prompts, test_answers, sampling_params  # type: ignore
            )

            wandb.log({
                "eval/correct": overview["correct"],
                "eval/wrong answer": overview["answer_wrong"],
                "eval/wrong format": overview["format_wrong"],
                "eval/accuracy": overview["correct"] / overview["count"],
                "eval/avg_length": overview["avg_length"],
                "eval_step": grpo_step,
            })

            with open(f"./results/train_grpo_{loss_type}_lr_{learning_rate}_step_{n_grpo_steps}_results.jsonl", "w") as f:
                for i in allinfo_dict_list:
                    json.dump(i, f)
                    f.write("\n")

    model.save_pretrained(save_directory=OUTPUT_PATH+f'/model_grpo/train_grpo_{loss_type}_lr_{learning_rate}_step_{n_grpo_steps}')
    tokenizer.save_pretrained(save_directory=OUTPUT_PATH+f'/model_grpo/train_grpo_{loss_type}_lr_{learning_rate}_step_{n_grpo_steps}')

if __name__ == "__main__":
    train_grpo()