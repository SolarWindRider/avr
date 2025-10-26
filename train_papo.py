import os
from argparse import ArgumentParser
from peft import LoraConfig
from trl.experimental.papo.papo_config import PAPOConfig
from trl.experimental.papo.papo_trainer import PAPOTrainer
from utils.universal import set_seed, get_dataset, model_processor, reward_fn

# ======= 基础环境与路径 =======
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
set_seed(42)

# ==========参数===================================
parser = ArgumentParser()
parser.add_argument("--model_path", type=str, default="../Downloads/Models/Qwen/Qwen2.5-VL-3B-Instruct")
parser.add_argument("--loss_type", type=str, default="grpo", choices=["dapo", "grpo"])  # 只有PAPO-G和PAPO-D
parser.add_argument("--output_dir", type=str, default="output")

args = parser.parse_args()
print(args)

image_root = "../datas/VisuRiddles"
train_json_path = "../datas/VisuRiddles/train_dataset.json"


# ======= 模型与处理器 =======
model, processor = model_processor(args.model_path)
train_ds, eval_ds = get_dataset(image_root, train_json_path, processor)


# ======= LoRA 配置（与你 SFT 里一致的 target_modules） =======
lora_config = LoraConfig(r=8, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], init_lora_weights=True)


# ======= PAPO 配置 =======
"""
多卡：建议用 torchrun 启动（见文末命令）。也可切换到 FSDP/Deepspeed。
这里把 remove_unused_columns=False 以保留图像列，供内部多模态拼接使用。
"""
papo_config = PAPOConfig(
    # PAPO-specific params
    perception_loss_weight=0.01,  # Weight for perception loss
    mask_ratio=0.6,  # 40% of image will be masked
    mask_type="random",  # Use patch masking (recommended)
    der_loss_weight1=0.02,
    der_loss_weight2=0.02,
    # GRPO params
    beta=0.01,
    epsilon=0.2,
    epsilon_high=0.3,
    loss_type=args.loss_type,
    output_dir=args.output_dir,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    num_generations=8,  # 每个 prompt 采样 8 条
    top_k=20,
    num_train_epochs=3,  # 可换成 num_train_epochs
    learning_rate=5e-5,  # RL 一般用更小 LR；按需调
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    logging_steps=5,
    save_strategy="steps",
    save_steps=50,
    save_only_model=True,
    eval_strategy="steps",
    eval_steps=50,
    report_to="swanlab",
    remove_unused_columns=False,
    fp16=False,
    bf16=True,
    max_prompt_length=8192,
    max_completion_length=2048,
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

# ======= 构建 PAPOTrainer =======
trainer = PAPOTrainer(
    model=model,
    processing_class=processor,  # 让内部根据 prompt+image 自动构建多模态输入
    args=papo_config,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    reward_funcs=[reward_fn],  # 也可叠加多个 reward
    peft_config=lora_config,  # LoRA 低秩微调
)

# ======= 训练 =======
if __name__ == "__main__":
    # 在开始训练前，将 LoRA 模块的参数手动转换
    from peft.peft_model import PeftModel

    if isinstance(trainer.model, PeftModel):
        for param in trainer.model.parameters():
            if param.requires_grad:
                param.data = param.data.bfloat16()
        print("Successfully converted trainable LoRA parameters to bf16.")

    trainer.train()
    # 结束时保存 LoRA 权重
    trainer.save_model()
