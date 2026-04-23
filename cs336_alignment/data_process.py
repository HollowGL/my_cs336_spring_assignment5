import json


PROMPT_PATH = "/root/autodl-tmp/assignment5-alignment/cs336_alignment/prompts/r1_zero.prompt"
MATH_DATA_PATH = "/root/autodl-tmp/assignment5-alignment/data/gsm8k"


if __name__ == "__main__":
    """将 gsm8k 数据集处理成 SFT 训练所需的格式，输出 processed_train.jsonl 文件"""
    processed_train = []

    with open(PROMPT_PATH, "r") as f:
        prompt = f.read()
    with open(MATH_DATA_PATH + "/train.jsonl", "r") as f:
        for line in f:
            train_sample = {}
            data = json.loads(line)
            train_sample['prompt'] = prompt.format(question=data['question'])
            train_sample['response'] = data['answer'].replace("\n####", "</think> <answer>") + " </answer>"
            processed_train.append(train_sample)

    with open(MATH_DATA_PATH + "/processed_train.jsonl", "w") as f:
        for sample in processed_train:
            json.dump(sample, f)
            f.write("\n")


    # print
    with open(MATH_DATA_PATH + "/processed_train.jsonl", "r") as f:
        lines = f.readlines()
        print("data length:", len(lines))  # 7473
        for i in range(min(3, len(lines))):
            print(json.loads(lines[i]))