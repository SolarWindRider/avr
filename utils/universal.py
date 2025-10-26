import json
import re
import numpy as np
import random
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration
from typing import List, Dict, Any
from qwen_vl_utils import fetch_image
import os
from datasets import load_dataset

# import torch_npu

promptTemplates = json.load(open("./promptTemplates.json", "r", encoding="utf-8"))


# 设置随机数种子
def set_seed(seed=42):
    """固定 torch、numpy 和 random 的随机种子，确保实验可复现"""
    random.seed(seed)  # 固定 Python 内置的 random 模块
    np.random.seed(seed)  # 固定 numpy 的随机种子
    torch.manual_seed(seed)  # 固定 torch 的 CPU 随机种子
    torch.cuda.manual_seed(seed)  # 固定 torch 的 GPU 随机种子
    torch.cuda.manual_seed_all(seed)  # 如果使用多个 GPU，固定所有 GPU 的随机种子
    torch.backends.cudnn.deterministic = True  # 确保 cuDNN 计算是确定性的
    torch.backends.cudnn.benchmark = False  # 关闭 cuDNN 的自动优化，保证可复现性
    # torch_npu.npu.manual_seed(seed)
    # torch_npu.npu.manual_seed_all(seed)  # if using multiple NPUs


# ========== 评估准确率函数Accuracy==========
def compute_accuracy(prompts, completions, references, bench, id, issudoku):
    log = []
    cnt = 0
    for idx in range(len(id)):
        print(f"------[{bench}]    id: {id[idx]}---------------")
        correct = 0
        if not issudoku[idx]:
            try:
                clean_answer_ = completions[idx].split("{")[-1].split(":")[-1]
                if "," in clean_answer_:  # LogicVista有多选
                    clean_answer = ", ".join(re.findall("[ABCDEFGH]+", clean_answer_))
                else:
                    clean_answer = re.findall("[ABCDEFGH1234567890]+", clean_answer_)
                    if len(clean_answer) != 0:
                        clean_answer = clean_answer[0]
                        if clean_answer in references[idx].strip():
                            correct = 1
                            cnt += 1
                    else:  # PuzzleVQA的有些选项是单词或词组
                        clean_answer = re.findall("[a-z ]+", clean_answer_)[0]
                        if clean_answer == references[idx].strip():
                            correct = 1
                            cnt += 1

            except Exception as e:
                pass
        else:  # 数独的判定
            try:
                clean_answer_ = re.findall("{.*}", completions[idx])[-1]
                clean_answer = "\n".join(list(json.loads(clean_answer_).values())[0].split())
                correct = 0
                if clean_answer == references[idx].strip():
                    correct = 1
                    cnt += 1
            except Exception as e:
                pass
        print(f"Prompts: {prompts[idx]}\nPrediction: {completions[idx]}\nReference: {references[idx]}\nCorrect: {correct}")
        log.append({"ID": id[idx], "Prediction": completions[idx], "Reference": references[idx], "Correct": correct})
    acc = cnt / len(references)
    print(f"Accuracy: {acc:.3f}")
    return {"Acc": acc, "Log": log}


# ========== 加载模型与处理器 =======
def model_processor(model_path):
    if "Qwen2.5-VL" in model_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    elif "Qwen3-VL" in model_path:
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    # 使用use_fast=False来避免图像处理器的警告
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    # pad_token 兜底（与 SFT 脚本保持一致）
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
        processor.pad_token_id = processor.tokenizer.pad_token_id
        print("Set pad_token_id to eos_token_id:", processor.tokenizer.pad_token_id)

    return model, processor


def get_dataset(image_root, train_json_path, processor):
    def preprocess_to_rl(example):
        image_path = os.path.join(image_root, example["imgs"][0])
        option = f"option: {example['option']}\n" if example["option"] != "" else ""
        question = (
            example["question"] + option + promptTemplates["Naive"] + 'Write the answer into a JSON form\n```json\n{"answer": "X"}```'
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                        "resized_height": 288,
                        "resized_width": 288,
                    },
                    {"type": "text", "text": question},
                ],
            },
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs = fetch_image({"image": image_path})
        return {
            # GRPOTrainer 期望的列名：prompt / image（它会自行拼多模态模板）
            "prompt": text,
            "image": image_inputs,
            # # metadatas 里带上 gold_answer，reward 用
            "metadatas": {"gold_answer": example.get("gold_answer", None)},
        }

    raw = load_dataset("json", data_files={"train": train_json_path})["train"]
    rl_ds = raw.map(preprocess_to_rl, remove_columns=list(raw.column_names), num_proc=8)
    # 简单切分 train/val（可按需调整）
    split = rl_ds.train_test_split(test_size=0.3, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    return train_ds, eval_ds


# ======= 奖励函数（文本型 Reward；TRL 暂不支持“视觉 Reward”）=======
"""
策略：
1) 从模型生成文本里解析出 JSON 片段，读取 "answer" 字段。
2) 与 metadata.gold_answer 做对比（大小写与空白无关）。
3) 完全一致给 1.0；否则 0.0。
4) 若输出符合 JSON 且包含 "answer" 字段，给 +0.1 结构奖励（上限 1.0）。

注意：GRPOTrainer 当前只支持文本 reward，不会把图像喂给 reward。此前我们已确认这一点。
"""

answer_regex = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def extract_answer_block(text: str) -> dict:
    """解析 ```json {...} ``` 块，返回 dict，如果失败返回 None"""
    if not isinstance(text, str):
        return None
    m = answer_regex.search(text)
    if not m:
        return None
    block = m.group(1)
    try:
        return json.loads(block)
    except Exception:
        return None


def reward_fn(completions: List[str], prompts: List[str], metadatas: List[Dict[str, Any]], **kwargs) -> List[float]:
    rewards = []
    # Loop over the completions, which are a list of lists or mixed types
    for out_item, meta in zip(completions, metadatas):
        out = ""
        # Handle cases where the item is a list or a single string/dict
        if isinstance(out_item, list) and len(out_item) > 0:
            # We assume the first element of the inner list is the generated text
            out = out_item[0]
        elif isinstance(out_item, str):
            out = out_item
        else:
            # If the item is not a list or string, we cannot process it
            rewards.append(0.0)
            continue

        gold = str((meta or {}).get("gold_answer", "")).strip()
        parsed = extract_answer_block(out)

        if parsed is None:
            # Format error -> 0 points
            rewards.append(0.0)
            continue

        pred = str(parsed.get("answer", "")).strip()
        if pred.lower() == gold.lower():
            # Correct format + correct answer -> 2 points
            rewards.append(2.0)
        else:
            # Correct format + incorrect answer -> 1 point
            rewards.append(1.0)
    return rewards
