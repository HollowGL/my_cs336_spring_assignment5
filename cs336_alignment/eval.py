from vllm import LLM, SamplingParams
from typing import Callable
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn, r1_zero_reward_fn_gsm8k
r1_zero_reward_fn = r1_zero_reward_fn_gsm8k
import re
import json
import argparse


QWEN_MATH_BASE_PATH = "/root/autodl-tmp/assignment5-alignment/Qwen2.5-Math-1.5B"
PROMPT_PATH = "/root/autodl-tmp/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
MATH_DATA_PATH = "/root/autodl-tmp/assignment5-alignment/data/gsm8k"

ANS_RE = re.compile(r"####\s*([\-0-9\.\,]+)")


def extract_reference_answer(answer: str) -> str:
    match = ANS_RE.search(answer)
    if match:
        return match.group(1).strip().replace(",", "")
    return "[invalid]"

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
        extracted_answer = extract_reference_answer(answer)
        base_reward = reward_fn(response, extracted_answer)
        reward_dict: dict[str, float | str] = {
            **base_reward,
            "response": response,
            "answer": answer,
            "prompt": prompt,
            "extracted_answer": extracted_answer
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
    with open(prompt_path, "r") as file:
        prompt = file.read()
    prompts = []
    answers = []
    with open(data_path, "r") as file:
        for line in file:
            data = json.loads(line)
            prompts.append(prompt.format(question=data["question"]))
            answers.append(data["answer"])
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

    prompts, answers = load_and_format_prompts(data_path=MATH_DATA_PATH+"/test.jsonl", prompt_path=PROMPT_PATH)

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
zero-shot: 0.0849    {'correct': 112, 'format_wrong': 900, 'answer_wrong': 307, 'count': 1319, 'avg_length': 593.5837755875664}
sft:       0.6293    {'correct': 830, 'format_wrong': 1, 'answer_wrong': 488, 'count': 1319, 'avg_length': 301.6467020470053}
grpo:      0.7475    {'correct': 986, 'format_wrong': 14, 'answer_wrong': 319, 'count': 1319, 'avg_length': 658.262319939348}
"""