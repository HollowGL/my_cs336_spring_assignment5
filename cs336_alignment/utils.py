from typing import Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import log_softmax
from transformers import PreTrainedTokenizerBase

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer.pad_token_id must not be None")

    full_ids_list = []
    full_mask_list = []

    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        output_ids = tokenizer.encode(output, add_special_tokens=False)

        full_ids_list.append(torch.tensor(prompt_ids + output_ids, dtype=torch.long))
        full_mask_list.append(torch.tensor(
            [False] * len(prompt_ids) + [True] * len(output_ids),
            dtype=torch.bool,
        ))

    padded_ids = pad_sequence(full_ids_list, batch_first=True, padding_value=pad_id)  # type: ignore
    padded_mask = pad_sequence(full_mask_list, batch_first=True, padding_value=False)

    return {
        "input_ids": padded_ids[:, :-1],
        "labels": padded_ids[:, 1:],
        "response_mask": padded_mask[:, 1:],
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


def get_response_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids=input_ids).logits
    log_probs = log_softmax(logits, dim=-1)
    response_log_probs = torch.gather(
        log_probs, dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)

    result = {"log_probs": response_log_probs}

    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        result["token_entropy"] = token_entropy

    return result


def masked_normalize(
        tensor: torch.Tensor,
        mask: torch.Tensor,
        dim: int | None=None,
        normalize_constant: float = 1.0,
) -> torch.Tensor:
    masked_tensor = tensor * mask
    return masked_tensor.sum(dim=dim) / normalize_constant


def sft_microbatch_train_step(
        policy_log_probs: torch.Tensor,
        response_mask: torch.Tensor,
        gradient_accumulation_steps: int,
        normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    microbatch_loss = masked_normalize(policy_log_probs, response_mask, -1, normalize_constant)
    loss = - microbatch_loss.mean() / gradient_accumulation_steps
    loss.backward()

    return loss, {}


def to_float(val):
    if isinstance(val, torch.Tensor):
        return val.float().item()
    return float(val)

