from typing import Callable, Literal

from einops import repeat
import torch

def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    
    raw_rewards = torch.as_tensor(
        [
            reward_fn(rollout_response, gt_response)["reward"]
            for rollout_response, gt_response in zip(
                rollout_responses, repeated_ground_truths
            )
        ],
        dtype=torch.float32,
    )  # shape: [num_prompts * group_size]

    rewards_per_group = raw_rewards.view(-1, group_size)  # shape: [num_prompts, group_size]
    mean_reward_per_group = rewards_per_group.mean(dim=1, keepdim=True)
    advantage = rewards_per_group - mean_reward_per_group

    if normalize_by_std:
        std_reward_per_group = rewards_per_group.std(dim=1, keepdim=True)
        advantage = advantage / (std_reward_per_group + advantage_eps)

    advantage = advantage.reshape(-1)

    metadata = {
        "mean": raw_rewards.mean().item(),
        "std": raw_rewards.std(unbiased=False).item(),
        "max": raw_rewards.max().item(),
        "min": raw_rewards.min().item(),
    }

    return advantage, raw_rewards, metadata


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,  # (batch_size, 1)
    policy_log_probs: torch.Tensor,           # (batch_size, sequence_length)
) -> torch.Tensor:                            # (batch_size, sequence_length)
    batch_size, seq_len = policy_log_probs.shape
    raw_rewards_or_advantages = repeat(raw_rewards_or_advantages, 'b 1->b s', s=seq_len)
    loss = -raw_rewards_or_advantages * policy_log_probs

    return loss


def compute_grpo_clip_loss(
    advantages: torch.Tensor,        # (batch_size, 1)
    policy_log_probs: torch.Tensor,  # (batch_size, sequence_length)
    old_log_probs: torch.Tensor,     # (batch_size, sequence_length)
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:  # loss: (batch_size, sequence_length)
    """ return: clip(pi_ratio * advantages, 1-cliprange, 1+cliprange) * advantages
    """
    pi_ratio = torch.exp(policy_log_probs - old_log_probs)
    batch_size, seq_len = policy_log_probs.shape
    advantages = repeat(advantages, "b 1 -> b s", s=seq_len)
    v = pi_ratio * advantages
    v_clip = torch.clip(pi_ratio, min=1-cliprange, max=1+cliprange) * advantages

    meta = {
        "clipped": v > v_clip
    }
    return -torch.min(v, v_clip), meta


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,             # (batch_size, sequence_length)
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,    # (batch_size, 1)
    advantages: torch.Tensor | None = None,     # (batch_size, 1)
    old_log_probs: torch.Tensor | None = None,  # (batch_size, sequence_length)
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:  # loss: (batch_size, sequence_length)
    if loss_type == "no_baseline":
        assert raw_rewards is not None
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        meta = {}
    elif loss_type == "reinforce_with_baseline":
        assert advantages is not None
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        meta = {}
    elif loss_type == "grpo_clip":
        assert advantages is not None and old_log_probs is not None and cliprange is not None
        loss, meta = compute_grpo_clip_loss(advantages, policy_log_probs, old_log_probs, cliprange)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    return loss, meta


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    masked_tensor = torch.where(mask, tensor, torch.zeros_like(tensor))
    return torch.sum(masked_tensor, dim=dim) / torch.sum(mask, dim=dim)


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,             # (batch_size, sequence_length)
    response_mask: torch.Tensor,                # (batch_size, sequence_length)
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,    # (batch_size, 1)
    advantages: torch.Tensor | None = None,     # (batch_size, 1)
    old_log_probs: torch.Tensor | None = None,  # (batch_size, sequence_length)
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss, meta = compute_policy_gradient_loss(
        policy_log_probs=policy_log_probs,
        loss_type=loss_type,
        raw_rewards=raw_rewards,
        advantages=advantages,
        old_log_probs=old_log_probs,
        cliprange=cliprange,
    )
    
    loss = masked_mean(loss, response_mask)
    loss = loss / gradient_accumulation_steps
    loss.backward()

    return loss, meta
