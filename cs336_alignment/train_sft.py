import json
import math
import os
import random
import shutil
import signal
import socket
import subprocess
import time
from typing import Iterator, Optional

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from vllm import SamplingParams
import wandb

from cs336_alignment.utils import (
    get_response_log_probs,
    sft_microbatch_train_step,
    tokenize_prompt_and_output,
    to_float,
)
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.eval import evaluate_vllm, load_and_format_prompts


QWEN_MATH_BASE_PATH = "/root/autodl-tmp/assignment5-alignment/Qwen2.5-Math-1.5B"
PROMPT_PATH = "/root/autodl-tmp/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
MATH_DATA_PATH = "/root/autodl-tmp/assignment5-alignment/data/gsm8k"
OUTPUT_PATH = "/root/autodl-tmp/assignment5-alignment/output/model_sft"
VLLM_SYNC_PATH = os.path.join(OUTPUT_PATH, "_vllm_eval_ckpt")

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

device_train = "cuda:0"
device_vllm = "cuda:1"

micro_batch_size = 2
n_grad_accum_steps = 16
num_epochs = 3
eval_every_n_epochs = 1
max_length = 512
train_samples = [256, 512, 1024, 7473]
# train_samples = [128, 256, 512, 1024, 7473]

# micro_batch_size = 1
# n_grad_accum_steps = 8
# num_epochs = 1
# eval_every_n_epochs = 1
# max_length = 512
# train_samples = [10]

def load_and_format_train_data(data_path: str) -> tuple[list[str], list[str]]:
    with open(data_path, "r") as f:
        data = [json.loads(line) for line in f]

    prompts = []
    responses = []
    for item in data:
        prompts.append(item["prompt"])
        responses.append(item["response"])

    return prompts, responses


def iter_microbatches(
    tokenized_train_data: dict[str, torch.Tensor],
    batch_size: int,
    device: str,
    shuffle: bool = True,
) -> Iterator[dict[str, torch.Tensor]]:
    n = tokenized_train_data["input_ids"].shape[0]
    indices = torch.randperm(n) if shuffle else torch.arange(n)

    for start in range(0, n, batch_size):
        idx = indices[start:start + batch_size]
        yield {k: v[idx].to(device) for k, v in tokenized_train_data.items()}


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


class VLLMServerProxy:
    def __init__(
        self,
        model_path: str,
        device: str,
        gpu_memory_utilization: float = 0.9,
        port: int | None = None,
    ):
        self.model_path = model_path
        self.device = device
        self.gpu_memory_utilization = gpu_memory_utilization
        self.port = port or find_free_port()
        self.proc = None
        self.log_file = None
        self.api_base = f"http://127.0.0.1:{self.port}/v1"
        self._start_server(model_path)

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                self.proc.wait(timeout=5)
            finally:
                self.proc = None

        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None

    @staticmethod
    def _parse_cuda_index(device_str: str) -> str:
        if not device_str.startswith("cuda:"):
            raise ValueError(f"Unsupported device string: {device_str}")
        return device_str.split(":")[1]

    def _start_server(self, model_path: str):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self._parse_cuda_index(self.device)

        cmd = [
            "python",
            "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--port", str(self.port),
            "--dtype", "bfloat16",
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--disable-log-stats",
            "--generation-config", "vllm",
        ]

        self.log_file = open("/tmp/vllm_server.log", "w")
        self.proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=self.log_file,
            stderr=self.log_file,
            start_new_session=True,
        )
        self._wait_until_ready()

    def _wait_until_ready(self, timeout: float = 180.0):
        start = time.time()
        last_err = None
        while time.time() - start < timeout:
            if self.proc.poll() is not None:  # type: ignore
                raise RuntimeError("vLLM server exited unexpectedly during startup.")

            try:
                resp = requests.get(f"http://127.0.0.1:{self.port}/v1/models", timeout=2.0)
                if resp.status_code == 200:
                    print("[vLLM] server is ready.")
                    return
            except Exception as e:
                last_err = e
                print(f"[vLLM] not ready yet: {repr(e)}")

            time.sleep(10.0)

        raise RuntimeError(f"Timed out waiting for vLLM server. last_err={last_err}")

    def restart_with_model(self, model_path: str):
        self.close()
        self.model_path = model_path
        self._start_server(model_path)

    def generate(self, prompts, sampling_params, batch_size: int = 64):
        class OutputText:
            def __init__(self, text):
                self.text = text

        class RequestOutput:
            def __init__(self, prompt, texts):
                self.prompt = prompt
                self.outputs = [OutputText(t) for t in texts]

        all_outputs = []

        for start in range(0, len(prompts), batch_size):
            end = min(start + batch_size, len(prompts))
            batch_prompts = prompts[start:end]

            for prompt in batch_prompts:
                payload = {
                    "model": self.model_path,
                    "prompt": prompt,
                    "temperature": sampling_params.temperature,
                    "top_p": sampling_params.top_p,
                    "min_tokens": sampling_params.min_tokens,
                    "max_tokens": sampling_params.max_tokens,
                    "stop": sampling_params.stop,
                    "n": sampling_params.n,
                    "stream": False,
                    "include_stop_str_in_output": sampling_params.include_stop_str_in_output,
                }

                resp = requests.post(
                    f"{self.api_base}/completions",
                    json=payload,
                    timeout=1800.0,
                )
                resp.raise_for_status()
                data = resp.json()

                texts = [c["text"].strip() for c in data["choices"]]
                if len(texts) != sampling_params.n:
                    raise RuntimeError(
                        f"Expected {sampling_params.n} choices, got {len(texts)}"
                    )

                all_outputs.append(RequestOutput(prompt, texts))

        return all_outputs

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def sync_model_to_vllm_server(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    vllm_server: VLLMServerProxy,
    save_dir: str,
):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    vllm_server.model_path = save_dir
    vllm_server.restart_with_model(save_dir)


def run_training_one_setting(train_sample: int):
    print(f"Training with {train_sample} samples")

    wandb.init(
        project="cs336-alignment",
        name=f"sft_{train_sample}_samples",
        config={"train_samples": train_sample},
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=QWEN_MATH_BASE_PATH,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model = model.to(device_train) # type: ignore

    model.gradient_checkpointing_enable()

    tokenizer = AutoTokenizer.from_pretrained(QWEN_MATH_BASE_PATH)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)  # type: ignore

    train_prompts, train_responses = load_and_format_train_data(
        MATH_DATA_PATH + "/processed_train.jsonl"
    )
    test_prompts, test_answers = load_and_format_prompts(
        data_path=MATH_DATA_PATH+"/test.jsonl", 
        prompt_path=PROMPT_PATH
    )

    tokenized_train_data = tokenize_prompt_and_output(
        prompt_strs=train_prompts[:train_sample],
        output_strs=train_responses[:train_sample],
        tokenizer=tokenizer,
    )

    num_train_examples = tokenized_train_data["input_ids"].shape[0]
    steps_per_epoch = math.ceil(num_train_examples / micro_batch_size)
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / n_grad_accum_steps)

    print(f"num_train_examples={num_train_examples}")
    print(f"micro_batch_size={micro_batch_size}")
    print(f"steps_per_epoch={steps_per_epoch}")
    print(f"optimizer_steps_per_epoch={optimizer_steps_per_epoch}")

    vllm_server = VLLMServerProxy(
        model_path=QWEN_MATH_BASE_PATH,
        device=device_vllm,
        gpu_memory_utilization=0.9,
    )

    global_train_step = 0
    global_eval_step = 0

    try:
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()

            microbatch_iter = iter_microbatches(
                tokenized_train_data,
                batch_size=micro_batch_size,
                device=device_train,
                shuffle=True,
            )

            accum_counter = 0
            accum_loss = 0
            accum_entropy = 0
            accum_steps = 0
            last_loss = None
            last_entropy = torch.tensor(0.0)
            last_response_mask = torch.tensor(0.0)

            for train_batch in microbatch_iter:
                input_ids = train_batch["input_ids"]
                labels = train_batch["labels"]
                response_mask = train_batch["response_mask"]

                with amp_ctx:
                    response_log_probs = get_response_log_probs(
                        model=model,
                        input_ids=input_ids,
                        labels=labels,
                        return_token_entropy=True,
                    )
                    log_probs = response_log_probs["log_probs"]
                    entropy = response_log_probs["token_entropy"]

                    loss, _ = sft_microbatch_train_step(
                        policy_log_probs=log_probs,
                        response_mask=response_mask,
                        gradient_accumulation_steps=n_grad_accum_steps,
                    )

                accum_counter += 1
                last_loss = loss
                last_entropy = entropy
                last_response_mask = response_mask
                accum_loss += loss.item()
                accum_entropy += entropy.mean().item()
                accum_steps += 1

                if accum_counter % n_grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_train_step += 1

                    print(
                        f"Epoch {epoch + 1}, optimizer step {global_train_step}    " +
                        f"loss: {accum_loss / accum_steps:.6f}    " +
                        f"Global Entropy: {accum_entropy / accum_steps:.6f}    " +
                        f"Response Entropy: {last_entropy[last_response_mask].mean().item():.6f}    " +
                        f"Prompt Entropy: {last_entropy[~last_response_mask].mean().item():.6f}"
                    )
                    wandb.log({
                        "train/loss": accum_loss / accum_steps,
                        "train/entropy": accum_entropy / accum_steps,
                        "train/response entropy": to_float(last_entropy[last_response_mask].mean()),
                        "train/prompt entropy": to_float(last_entropy[~last_response_mask].mean()),
                        "train_step": global_train_step,
                    })

            # 处理 epoch 末尾不足 n_grad_accum_steps 的剩余梯度
            if accum_counter % n_grad_accum_steps != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_train_step += 1

                print(f"Epoch {epoch + 1}, optimizer step {global_train_step} (tail)")
                print(f"loss: {accum_loss / accum_steps:.6f}")

                wandb.log({
                    "train/loss": accum_loss / accum_steps,
                    "train/entropy": accum_entropy / accum_steps,
                    "train/response entropy": to_float(last_entropy[last_response_mask].mean()),
                    "train/prompt entropy": to_float(last_entropy[~last_response_mask].mean()),
                    "train_step": global_train_step,
                })

            if (epoch + 1) % eval_every_n_epochs == 0:
                model.eval()

                # eval 前同步一次：保存 checkpoint -> 重启 GPU1 上的 vLLM
                sync_model_to_vllm_server(
                    model=model,
                    tokenizer=tokenizer,
                    vllm_server=vllm_server,
                    save_dir=VLLM_SYNC_PATH,
                )

                sampling_params = SamplingParams(
                    seed=SEED,
                    temperature=0,
                    top_p=1.0,
                    max_tokens=1024,
                    stop=["</answer>"],
                    include_stop_str_in_output=True
                )

                overview, allinfo_dict_list = evaluate_vllm(
                    vllm_model=vllm_server,  # type: ignore
                    reward_fn=r1_zero_reward_fn,
                    prompts=test_prompts,
                    answers=test_answers,
                    eval_sampling_params=sampling_params,
                )

                accuracy = overview["correct"] / overview["count"]
                global_eval_step += 1

                print(f"Evaluation after epoch {epoch + 1}")
                print(f"Correct answer: {overview['correct']}")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Wrong answer with correct format: {overview['answer_wrong']}")
                print(f"Wrong format: {overview['format_wrong']}")

                wandb.log({
                    "eval/correct": overview["correct"],
                    "eval/wrong answer": overview["answer_wrong"],
                    "eval/wrong format": overview["format_wrong"],
                    "eval/accuracy": accuracy,
                    "eval_step": global_eval_step,
                })

                with open(f"./results/train_sft_{train_sample}_results.jsonl", "w") as f:
                    for i in allinfo_dict_list:
                        json.dump(i, f)
                        f.write("\n")   

                model.save_pretrained(save_directory=OUTPUT_PATH+f'/model_sft/train_sample_{train_sample}')
                tokenizer.save_pretrained(save_directory=OUTPUT_PATH+f'/model_sft/train_sample_{train_sample}')

    finally:
        vllm_server.close()
        wandb.finish()


def main(train_samples: list[int]) -> None:
    for train_sample in train_samples:
        run_training_one_setting(train_sample)


if __name__ == "__main__":
    main(train_samples)