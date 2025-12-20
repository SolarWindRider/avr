import json
import re
import numpy as np
import random
import torch
from transformers import AutoProcessor
from typing import List, Dict, Any
from qwen_vl_utils import fetch_image
import os
from datasets import load_dataset
import json  # 确保 json 被导入
from peft import PeftModel

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


# ========== 融合模型 ==========================
def merge(args):
    try:
        if args.lora_path != None:
            # 由于vllm不支持加载视觉模块的lora权重，所以先做融合是最简单的方案
            # Step 1: 加载 base 模型 + LoRA adapter
            if args.lora_path:
                base_model, processor = model_processor(args.model_path)
                peft_model = PeftModel.from_pretrained(base_model, args.lora_path, trust_remote_code=True)

                # Step 2: 将 LoRA 融合进模型
                peft_model = peft_model.merge_and_unload()  # ⚠️ 注意这一步

                # Step 3: 保存融合后的模型
                args.model_path = f"merged-model-{args.log_path}"  # 更新模型路径为融合后的模型
                peft_model.save_pretrained(args.model_path)
                processor.save_pretrained(args.model_path)
                print(f"Lora 权重与模型权重融合【成功】。\nLora权重路径：{args.lora_path}\n模型权重路径：{args.model_path}")
    except Exception as e:
        print(f"Error: {e}")
        print(f"Lora 权重与模型权重融合【失败】。\nLora权重路径：{args.lora_path}\n模型权重路径：{args.model_path}")
    return args


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
        from transformers import Qwen2_5_VLForConditionalGeneration

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, trust_remote_code=True, dtype=torch.bfloat16)
    elif "Qwen3-VL" in model_path:
        from transformers import Qwen3VLForConditionalGeneration

        model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, trust_remote_code=True, dtype=torch.bfloat16)
    # 使用use_fast=False来避免图像处理器的警告
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    # pad_token 兜底（与 SFT 脚本保持一致）
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
        processor.pad_token_id = processor.tokenizer.pad_token_id
        print("Set pad_token_id to eos_token_id:", processor.tokenizer.pad_token_id)

    return model, processor


def get_dataset(image_root, train_json_path, think_process_key="gold_analysis", isGuide=False):
    def preprocess_to_rl(example, think_process_key, isGuide):
        image_path = os.path.join(image_root, example["imgs"][0])
        option = f"option: {example['option']}\n" if example["option"] != "" else ""
        question = (
            example["question"] + option + promptTemplates["Naive"] + 'Write the answer into a JSON form\n```json\n{"answer": "X"}```'
        )

        messages = [
            {
                "role": "user",
                "content": question,
            }
        ]
        if isGuide:
            messages.append(
                {
                    "role": "assistant",
                    "content": example[think_process_key].strip() + "\n",
                },
            )
        return {
            # GRPOTrainer 期望的列名：prompt / image（它会自行拼多模态模板）
            "prompt": messages,
            "image": fetch_image({"image": image_path}),
            # # metadatas 里带上 gold_answer，reward 用
            "metadatas": {"gold_answer": example.get("gold_answer", None)},
        }

    raw = load_dataset("json", data_files={"train": train_json_path})["train"]
    split = raw.train_test_split(test_size=0.3, seed=42)
    # 简单切分 train/val（可按需调整）
    raw_train = split["train"]
    raw_eval = split["test"]
    train_ds = raw_train.map(
        lambda example: preprocess_to_rl(example, think_process_key, isGuide=isGuide),
        remove_columns=list(raw_train.column_names),
        num_proc=8,
    )
    eval_ds = raw_eval.map(
        lambda example: preprocess_to_rl(example, think_process_key, isGuide=False), remove_columns=list(raw_eval.column_names), num_proc=8
    )
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
            out = out_item[0]["content"]
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


# ========== 评估准确率函数Accuracy==========


# MODIFIED: 重写 compute_accuracy 以支持 pass@k
def compute_accuracy(prompts, generation_outputs, references, bench, id, issudoku, k_values):
    """
    计算 pass@k 准确率。

    :param prompts: vLLM的输入prompt列表
    :param generation_outputs: vLLM的原始输出列表 (llm.generate的结果)
    :param references: 答案列表
    :param bench: benchmark 名称
    :param id: ID 列表
    :param issudoku: 是否为数独的布尔值列表
    :param k_values: 要计算的 k 值列表, e.g., [1, 2, 4]
    :return: 包含 Acc (字典) 和 Log 的字典
    """
    log = []
    # MODIFIED: cnt 现在是一个字典，为每个k值计数
    cnt = {k: 0 for k in k_values}

    for idx in range(len(id)):
        print(f"------[{bench}]    id: {id[idx]}---------------")

        # MODIFIED: 获取此 prompt 的所有 k 个 completions
        completions = [o.text.strip() for o in generation_outputs[idx].outputs]

        is_correct_list = []  # 存储每个 completion 是否正确 (0 或 1)

        # MODIFIED: 遍历此 prompt 的 *所有* completions
        for completion in completions:
            correct = 0  # 默认为错误
            if not issudoku[idx]:
                try:
                    clean_answer_ = completion.split("{")[-1].split(":")[-1]
                    if "," in clean_answer_:  # LogicVista有多选
                        clean_answer = ", ".join(re.findall("[ABCDEFGH]+", clean_answer_))
                    else:
                        clean_answer = re.findall("[ABCDEFGH1234567890]+", clean_answer_)
                        if len(clean_answer) != 0:
                            clean_answer = clean_answer[0]
                            if clean_answer in references[idx].strip():
                                correct = 1
                        else:  # PuzzleVQA的有些选项是单词或词组
                            clean_answer = re.findall("[a-z ]+", clean_answer_)[0]
                            if clean_answer == references[idx].strip():
                                correct = 1
                except Exception as e:
                    pass
            else:  # 数独的判定
                try:
                    clean_answer_ = re.findall("{.*}", completion)[-1]
                    clean_answer = "\n".join(list(json.loads(clean_answer_).values())[0].split())
                    if clean_answer == references[idx].strip():
                        correct = 1
                except Exception as e:
                    pass

            is_correct_list.append(correct)

        # MODIFIED: 根据 is_correct_list 计算 pass@k
        # "pass@k" = 在前 k 个样本中，是否至少有 1 个是正确的
        for k in k_values:
            # 检查 is_correct_list 的前 k 个元素
            # any(is_correct_list[:k]) 会检查 [True, False, ...] 中是否有 True
            if any(is_correct_list[:k]):
                cnt[k] += 1

        # MODIFIED: 更新打印和日志内容
        print(
            f"Prompts: {prompts[idx]['prompt']}\nPredictions: {completions}\nReference: {references[idx]}\nCorrect_List: {is_correct_list}"
        )
        log.append({"ID": id[idx], "Predictions": completions, "Reference": references[idx], "Correct_List": is_correct_list})

    # MODIFIED: 计算所有 k 值的准确率
    acc = {f"pass@{k}": cnt[k] / len(references) for k in k_values}

    print(f"Accuracy: {acc}")
    return {"Acc": acc, "Log": log}
