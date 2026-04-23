# CS336 Spring 2025 Assignment 5: Alignment

实现了 sft 和 grpo 两种算法，在 [gsm8k](./data/gsm8k/) 测试集（1319 条）上进行训练和评估，其它任务实现并不完整。

| Strategies | Accuracy   | Detail                                                       |
| ---------- | ---------- | ------------------------------------------------------------ |
| zero-shot  | 0.0849     | {'correct': 112, 'format_wrong': 900, 'answer_wrong': 307, 'avg_length': 593.583} |
| sft        | 0.6293     | {'correct': 830, 'format_wrong': 1, 'answer_wrong': 488, 'avg_length': 301.646} |
| grpo       | 0.7475 | {'correct': 986, 'format_wrong': 14, 'answer_wrong': 319, 'avg_length': 658.262} |


同时也将训练后的模型在 [math12k](https://huggingface.co/datasets/nlile/hendrycks-MATH-benchmark) 测试集（500 条）上进行评测，结果如下：
| Strategies | Accuracy | Detail                                                       |
| ---------- | -------- | ------------------------------------------------------------ |
| zero-shot  | 0.0500   | {'correct': 25, 'format_wrong': 359, 'answer_wrong': 116, 'avg_length': 724.378} |
| sft        | 0.3320   | {'correct': 166, 'format_wrong': 18, 'answer_wrong': 316, 'avg_length': 506.652} |
| grpo       | 0.5160   | {'correct': 258, 'format_wrong': 102, 'answer_wrong': 140, 'avg_length': 1113.124} |
* answer_wrong 指格式正确但答案错误


## 环境配置
在 2 块 5090 上进行实验，修改了 `pyproject.toml` 中的依赖项以适应，torch, vllm 和 flash-attn 的版本分别为 2.10.0+cu128, 0.19.0 和 2.8.3。
flash-attn 从 github 源码编译，大概花了半个小时。


## 项目结构
```.
|-- README.md
|-- cs336_alignment
|    |-- data_process.py    # 将 gsm8k 数据集处理成 sft 需要的格式
|    |-- drgrpo_grader.py   # 新增适用于 gsm8k 的打分函数
|    |-- eval.py            # 评测脚本，在 gsm8k 数据集的评测
|    |-- eval_math12k.py    # 额外的评测脚本，在 math12k 数据集的评测
|    |-- grpo.py            # grpo 算法实现
|    |-- interact.ipynb     # debug 和测试用
|    |-- prompts
|    |-- train_grpo.py      # grpo 训练脚本
|    |-- train_sft.py       # sft 训练脚本
|    `-- utils.py           # sft 算法实现和一些工具函数
|-- data
|-- demo.py                 # 检验 vllm 的脚本
|-- pyproject.toml
|-- tests
|-- uv.lock
```

## 实现说明
- gsm8k 数据集相较于 math12k 数据集更偏向应用题，会出现 `18 dollars` 这样的答案，但原有打分函数认为 `"18 dollars" != "18"` 而判错，所以新增打分函数 `r1_zero_reward_fn_gsm8k`
- 由于 vllm 版本较高，以下补丁失效：
    ```python
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch("vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None)
    ```
    因此实现 `VLLMServerProxy` 类，让 vllm 单独在另一个进程中运行，设置环境变量 `env["CUDA_VISIBLE_DEVICES"] = self._parse_cuda_index(self.device)` 来指定使用的 GPU。

## 参考
https://github.com/heng380/cs336_assignment-5