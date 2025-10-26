import os
import torch
from datasets import load_dataset
from peft.tuners import LoraConfig
from trl import SFTConfig, SFTTrainer
from argparse import ArgumentParser
from qwen_vl_utils import process_vision_info
from torch.nn.utils.rnn import pad_sequence
from utils.universal import set_seed, promptTemplates, model_processor


set_seed(42)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # 防止爆显存
# ==========参数===================================
parser = ArgumentParser()
parser.add_argument("--model_path", type=str, default="../Downloads/Models/Qwen/Qwen2.5-VL-7B-Instruct")
parser.add_argument("--mode", type=str, default="sft", choices=["sft", "syndata"])
parser.add_argument("--output_dir", type=str, default="output")

args = parser.parse_args()
print(args)


image_root = "../datas/VisuRiddles"
if args.mode == "sft":
    think_process_key = "gold_analysis"  # 思维过程在数据集json文件中的字段
    train_json_path = "../datas/VisuRiddles/train_dataset.json"
elif args.mode == "syndata":
    think_process_key = "explanation"
    train_json_path = "../datas/VisuRiddles/syndata.json"
# ======= 加载模型和处理器 =======
model, processor = model_processor(args.model_path)


# ======= 自定义 preprocess（支持图像） =======
def preprocess(example):
    image_path = os.path.join(image_root, example["imgs"][0])
    option = f"option: {example['option']}\n" if example["option"] != "" else ""
    question = example["question"] + option + promptTemplates["Naive"] + 'Write the answer into a JSON form\n```json\n{"answer": "X"}```'
    answer = example[think_process_key] + f'\n```json\n{{"answer": "{example["gold_answer"]}"}}```'

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
        {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info(messages)

    batch = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,  # Padding handled in collator
        return_tensors="pt",
        max_length=8192,
        truncation=True,
    )

    labels = batch["input_ids"].clone()
    assistant_token = 151644
    assistant_indices = (batch["input_ids"] == assistant_token).nonzero(as_tuple=True)[1]
    if assistant_indices.numel() > 0:
        last_assistant_idx = assistant_indices[-1]
        labels[:, : last_assistant_idx + 1] = -100

    batch["labels"] = labels
    return batch


# ======= 自定义 data_collator =======
def custom_data_collator(batch):
    input_ids = [torch.tensor(item["input_ids"]).squeeze(0) for item in batch]
    attention_mask = [torch.tensor(item["attention_mask"]).squeeze(0) for item in batch]
    labels = [torch.tensor(item["labels"]).squeeze(0) for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)

    pixel_values = [torch.tensor(item["pixel_values"]).unsqueeze(0) for item in batch]
    image_grid_thw = [torch.tensor(item["image_grid_thw"]).squeeze(0) for item in batch]

    # Pad or stack pixel_values
    if pixel_values:
        shapes = [pv.shape for pv in pixel_values]
        if len(set(shapes)) == 1:
            pixel_values = torch.stack(pixel_values)
        else:
            max_channels = max(pv.shape[0] for pv in pixel_values)
            max_height = max(pv.shape[1] for pv in pixel_values)
            max_width = max(pv.shape[2] for pv in pixel_values)
            padded_pixel_values = []
            for pv in pixel_values:
                pad_c = max_channels - pv.shape[0]
                pad_h = max_height - pv.shape[1]
                pad_w = max_width - pv.shape[2]
                padded_pv = torch.nn.functional.pad(pv, (0, pad_w, 0, pad_h, 0, pad_c), value=0)
                padded_pixel_values.append(padded_pv)
            pixel_values = torch.stack(padded_pixel_values)
    else:
        pixel_values = None

    if image_grid_thw:
        image_grid_thw = torch.stack(image_grid_thw)
    else:
        image_grid_thw = None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }


# ======= 加载数据集并拆分 =======
dataset = load_dataset("json", data_files={"train": train_json_path})["train"]
dataset = dataset.map(
    preprocess,
    batched=False,
    cache_file_name=None,
    remove_columns=list(dataset.column_names),
)


# Split dataset into train and validation (70:30)
train_test_split = dataset.train_test_split(test_size=0.3, seed=42)
train_dataset = train_test_split["train"]
val_dataset = train_test_split["test"]

# ======= LoRA 配置 =======
lora_config = LoraConfig(
    r=16,  # 默认8
    # lora_alpha=8, #默认8
    # lora_dropout=0.05, #默认0
    # bias="none",#默认none
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

# ======= SFT 配置 =======
sft_config = SFTConfig(
    output_dir=args.output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=5,
    logging_steps=5,
    save_strategy="steps",
    save_steps=20,
    save_total_limit=2,
    save_only_model=True,  # Only save model weights, not optimizer states
    eval_strategy="steps",
    per_device_eval_batch_size=1,
    eval_steps=20,
    report_to="swanlab",
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=False,
    bf16=True,
    remove_unused_columns=False,
    learning_rate=5e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    warmup_steps=1000,
    fsdp="full_shard auto_wrap",
    fsdp_config={
        "mixed_precision": "bf16",
        "forward_prefetch": True,
        "use_orig_params": False,
        "use_cpu": True,
        "offload_params": True,
        "offload_optimizer": True,
        "enable_gradient_checkpointing": True,
    },
)


# ======= 构建 Trainer =======
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=lora_config,
    data_collator=custom_data_collator,
)


# ======= 启动训练 =======
if __name__ == "__main__":
    trainer.train()
