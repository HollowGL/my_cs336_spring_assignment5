import argparse
import json
from typing import Callable

from datasets import load_dataset
from vllm import LLM, SamplingParams

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


QWEN_MATH_BASE_PATH = "/root/autodl-tmp/assignment5-alignment/Qwen2.5-Math-1.5B"
PROMPT_PATH = "/root/autodl-tmp/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
MATH12K_DATA_PATH = "/root/autodl-tmp/assignment5-alignment/data/math12k"


def run_vllm(vllm_model, prompts, sampling_params) -> list[str]:
    result = vllm_model.generate(prompts, sampling_params)
    texts = [output.outputs[0].text.strip() for output in result]
    return texts

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    answers: list[str],
    eval_sampling_params: SamplingParams
):
    responses = run_vllm(vllm_model, prompts, eval_sampling_params)
    allinfo_dict_list = []
    total_length = 0
    for response, answer, prompt in zip(responses, answers, prompts):
        base_reward = reward_fn(response, answer)
        reward_dict: dict[str, float | str] = {
            **base_reward,
            "response": response,
            "answer": answer,
            "prompt": prompt,
        }
        allinfo_dict_list.append(reward_dict)
        total_length += len(response)

    overview: dict[str, float] = {"correct":0, "format_wrong":0, "answer_wrong":0, "count":0}
    for reward in allinfo_dict_list:
        overview["count"] += 1
        if reward["reward"] == 1:
            overview["correct"] += 1
        elif reward["format_reward"] == 1:
            overview["answer_wrong"] += 1
        else:
            overview["format_wrong"] += 1
    overview["avg_length"] = total_length / overview["count"]
    return overview, allinfo_dict_list


def load_and_format_prompts(data_path: str, prompt_path: str):
    test_dataset  = load_dataset(data_path, split="test")
    with open(prompt_path, "r") as file:
        prompt = file.read()

    prompts = []
    answers = []

    for data in test_dataset:
        prompts.append(prompt.format(question=data["problem"]))  # type: ignore
        answers.append(data["answer"])  # type: ignore

    return prompts, answers

def build_llm_and_params(model_path: str) -> tuple[LLM, SamplingParams]:
    llm = LLM(model_path)
    sampling_params = SamplingParams(
        seed=42,
        temperature=1,
        top_p=0.9,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )
    return llm, sampling_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--choice")
    args = parser.parse_args()

    prompts, answers = load_and_format_prompts(data_path=MATH12K_DATA_PATH, prompt_path=PROMPT_PATH)

    if args.choice == "load_prompt_answer":
        for i, j in zip(prompts, answers):
            print (f"prompt:{i}, \n answer:{j}")
            break
    else:
        llm, sampling_params = build_llm_and_params(QWEN_MATH_BASE_PATH)
        # llm, sampling_params = build_llm_and_params("output/model_sft/train_sample_7473")
        # llm, sampling_params = build_llm_and_params("output/model_grpo/train_grpo_nobaseline_lr_3e-5_step_48")
        overview, allinfo_dict_list = evaluate_vllm(llm, r1_zero_reward_fn, prompts, answers, sampling_params)

        print(overview)
        print(f'\033[32m---accuracy: {overview["correct"] / overview["count"]:.4f}---\033[0m')

        # with open("./results/train_sample_7473_result_fn_gsm8k.jsonl", "w") as f:
        #     for i in allinfo_dict_list:
        #         json.dump(i, f)
        #         f.write("\n")

""" accuracy
zero-shot: 0.0500   {'correct': 25, 'format_wrong': 359, 'answer_wrong': 116, 'count': 500, 'avg_length': 724.378}
sft:       0.3320   {'correct': 166, 'format_wrong': 18, 'answer_wrong': 316, 'count': 500, 'avg_length': 506.652}
grpo:      0.5160   {'correct': 258, 'format_wrong': 102, 'answer_wrong': 140, 'count': 500, 'avg_length': 1113.124}
"""