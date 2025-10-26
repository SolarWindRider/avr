# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A minimal Online DPO trainer adapted from TRL's SFTTrainer style, using the
sampling + online preference optimization loop from the reference
`online_dpo_trainer_qwen2.5vl.py`.

Drop this file next to your trainers and import `OnlineDPOTrainer`.

Usage example:

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl.trainer.online_dpo_config import OnlineDPOConfig
from online_dpo_trainer import OnlineDPOTrainer

model_id = "Qwen/Qwen2-0.5B-Instruct"
args = OnlineDPOConfig(output_dir="odpo-out", max_length=4096)

# Dataset must yield dicts with at least: {"prompt": str}
train_ds = load_dataset("my/prompt_only_dataset", split="train")

trainer = OnlineDPOTrainer(
    model=model_id,
    args=args,
    train_dataset=train_ds,
    processing_class=AutoTokenizer.from_pretrained(model_id),
    reward_model=AutoModelForCausalLM.from_pretrained("reward/model"),  # or pass a judge
)
trainer.train()
```
"""

from __future__ import annotations

import warnings
from functools import wraps
from typing import Any, Callable, Optional, Union

import jinja2
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoModelForCausalLM,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    is_apex_available,
)
from transformers.trainer_utils import EvalPrediction, seed_worker
from transformers.training_args import OptimizerNames
from transformers.utils import (
    is_peft_available,
    is_sagemaker_mp_enabled,
    logging,
)

from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
)
from trl.models import create_reference_model
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.judges import BasePairwiseJudge
from trl.trainer.online_dpo_config import OnlineDPOConfig
from trl.trainer.utils import (
    SIMPLE_CHAT_TEMPLATE,
    DPODataCollatorWithPadding,
    disable_dropout_in_model,
    empty_cache,
    get_reward,
    prepare_deepspeed,
    truncate_right,
)

if is_peft_available():
    from peft import PeftModel, get_peft_model

if is_apex_available():
    from apex import amp  # type: ignore

if is_sagemaker_mp_enabled():
    from smdistributed.modelparallel import __version__ as SMP_VERSION  # type: ignore
    from packaging import version as _pkg_version

    IS_SAGEMAKER_MP_POST_1_10 = _pkg_version.parse(SMP_VERSION) >= _pkg_version.parse("1.10")
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

logger = logging.get_logger(__name__)


class OnlineDPOTrainer(Trainer):
    """Online Direct Preference Optimization (DPO) Trainer.

    This variant mirrors the ergonomics of `SFTTrainer` (processing_class handling,
    optional PEFT wrapping, custom data_collator) while implementing the online
    sampling + pairwise loss loop from TRL's ODPO.

    Required train/eval dataset columns: at least a `prompt` (string) per row.
    """

    _tag_names = ["trl", "online-dpo"]

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str],
        ref_model: Union[PreTrainedModel, nn.Module, None] = None,
        reward_model: Union[PreTrainedModel, nn.Module, None] = None,
        judge: Optional[BasePairwiseJudge] = None,
        args: Optional[OnlineDPOConfig] = None,
        data_collator: Optional[DataCollator] = None,  # type: ignore
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ] = None,
        reward_processing_class: Optional[PreTrainedTokenizerBase] = None,
        peft_config: Optional[dict] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        if args is None:
            raise ValueError("`args` (OnlineDPOConfig) must be provided.")
        if processing_class is None:
            raise ValueError("`processing_class` (tokenizer/processor) must be provided.")

        # Either reward_model OR judge must be passed
        if reward_model is not None and judge is not None:
            warnings.warn(
                "Both `reward_model` and `judge` provided — ignoring `judge` and using `reward_model`.",
                UserWarning,
            )
            judge = None
        elif reward_model is None and judge is None:
            raise ValueError("Either `reward_model` or `judge` must be provided.")

        self.ref_model = ref_model
        self.reward_model = reward_model
        self.reward_processing_class = reward_processing_class
        self.judge = judge

        # Load or validate model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            if args.model_init_kwargs is not None:
                warnings.warn(
                    "`model_init_kwargs` provided but model is already instantiated — ignoring.",
                    UserWarning,
                )
        self.is_encoder_decoder = model.config.is_encoder_decoder

        # PEFT wrapping if requested
        if peft_config is not None:
            if not is_peft_available():
                raise ImportError("PEFT requested but not available. `pip install peft`. ")
            if isinstance(model, PeftModel):
                # start from a base model ref
                model = model.merge_and_unload()
            model = get_peft_model(model, peft_config)

        # Disable dropout if asked
        if args.disable_dropout:
            disable_dropout_in_model(model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Reference model handling
        if ref_model is None:
            # If PEFT, we can share weights by disabling adapters during ref forward
            self.ref_model = None if peft_config is not None else create_reference_model(model)
        else:
            self.ref_model = ref_model
            self.ref_model.eval()

        # Reward model always eval/no-grad
        if self.reward_model is not None:
            self.reward_model.eval()

        # Default collator: pairwise padding for prompts
        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(pad_token_id=processing_class.pad_token_id)

        # Basic generation params for sampling online completions
        self.max_length = args.max_length
        if args.use_vllm:
            raise NotImplementedError("vLLM generation not wired in this minimal trainer.")
        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=50,
            top_p=1.0,
            do_sample=True,
            use_cache=False if args.gradient_checkpointing else True,
        )

        # Suppress Trainer FLOPs warning (no input_ids in ODPO batches)
        model.warnings_issued["estimate_tokens"] = True

        # Initialize Trainer superclass (handles DS/FSDP/optim/scheduler)
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,  # (None, None) -> default adamw + scheduler
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Tag models that know how to store metadata
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        # Deepspeed placement for ref/reward
        if self.is_deepspeed_enabled:
            if self.reward_model is not None:
                self.reward_model = prepare_deepspeed(
                    self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )
            if self.ref_model is not None:
                self.ref_model = prepare_deepspeed(
                    self.ref_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )
        else:
            if self.ref_model is not None:
                self.ref_model = self.ref_model.to(self.accelerator.device)
            if self.reward_model is not None:
                self.reward_model = self.reward_model.to(self.accelerator.device)

        self._beta = args.beta
        self._init_stats()

    # -----------------------------
    # Dataloaders: keep columns
    # -----------------------------
    @wraps(Trainer.get_train_dataloader)
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        train_dataset = self.train_dataset
        params = {
            "batch_size": self._train_batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        if not isinstance(train_dataset, IterableDataset):
            params["sampler"] = self._get_train_sampler()
            params["drop_last"] = self.args.dataloader_drop_last
            params["worker_init_fn"] = seed_worker
            params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        return self.accelerator.prepare(DataLoader(train_dataset, **params))

    @wraps(Trainer.get_eval_dataloader)
    def get_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if hasattr(self, "_eval_dataloaders") and key in getattr(self, "_eval_dataloaders", {}) and self.args.dataloader_persistent_workers:
            return self.accelerator.prepare(self._eval_dataloaders[key])
        eds = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset if eval_dataset is not None else self.eval_dataset
        )
        params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        if not isinstance(eds, IterableDataset):
            params["sampler"] = self._get_eval_sampler(eds)
            params["drop_last"] = self.args.dataloader_drop_last
            params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        dl = DataLoader(eds, **params)
        if self.args.dataloader_persistent_workers:
            if not hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders = {}
            self._eval_dataloaders[key] = dl
        return self.accelerator.prepare(dl)

    # -----------------------------
    # Tokenization helpers
    # -----------------------------
    @staticmethod
    def tokenize_row(feature, is_encoder_decoder: bool, tokenizer: PreTrainedTokenizerBase) -> dict[str, Any]:
        if not is_encoder_decoder:
            batch = tokenizer(feature["prompt"], add_special_tokens=False)
            if tokenizer.bos_token_id is not None:
                if not batch["input_ids"] or batch["input_ids"][0] != tokenizer.bos_token_id:
                    batch["input_ids"] = [tokenizer.bos_token_id] + batch["input_ids"]
                    batch["attention_mask"] = [1] + batch["attention_mask"]
        else:
            batch = tokenizer(feature["prompt"], add_special_tokens=True)
        return {f"prompt_{k}": v for k, v in batch.items()}

    # -----------------------------
    # Generation + forward
    # -----------------------------
    def _generate(self, model, prompts):
        eos_id = self.processing_class.eos_token_id
        pad_id = self.processing_class.pad_token_id
        # format + tokenize on-the-fly so reward/policy tokenizers may differ
        items = [{"prompt": p} for p in prompts]
        items = [maybe_apply_chat_template(x, self.processing_class) for x in items]
        items = [self.tokenize_row(x, self.is_encoder_decoder, self.processing_class) for x in items]
        batch = self.data_collator(items)
        batch = self._prepare_inputs(batch)

        prompt_ids = batch["prompt_input_ids"].repeat(2, 1)
        prompt_mask = batch["prompt_attention_mask"].repeat(2, 1)
        with unwrap_model_for_generation(
            model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped:
            out = unwrapped.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                generation_config=self.generation_config,
            )
        completion_ids = out[:, prompt_ids.size(1) :]
        completion_ids, completion_mask = truncate_right(completion_ids, eos_id, pad_id)
        return prompt_ids, prompt_mask, completion_ids, completion_mask

    def _forward(self, model, prompt_ids, prompt_mask, completion_ids, completion_mask):
        # respect max_length by left-truncating prompt
        overflow = max(prompt_ids.size(1) + completion_ids.size(1) - self.max_length, 0)
        if overflow:
            prompt_ids = prompt_ids[:, overflow:]
            prompt_mask = prompt_mask[:, overflow:]
        ids = torch.cat([prompt_ids, completion_ids], dim=1)
        mask = torch.cat([prompt_mask, completion_mask], dim=1)
        out = model(ids, attention_mask=mask)
        prompt_len = prompt_ids.size(1)
        start = prompt_len - 1 if prompt_len > 0 else 0
        logits = out.logits[:, start:-1]
        logprobs = torch.take_along_dim(
            logits.log_softmax(dim=-1), completion_ids.unsqueeze(-1), dim=2
        ).squeeze(-1)
        return logprobs

    # -----------------------------
    # Training step (pairwise DPO loss)
    # -----------------------------
    @property
    def beta(self):
        b = self._beta
        return b[self.state.epoch] if isinstance(b, list) and self.state.epoch < len(b) else (b[-1] if isinstance(b, list) else b)

    def training_step(self, model: nn.Module, inputs: dict[str, Any], num_items_in_batch: Optional[int] = None) -> torch.Tensor:
        model.train()
        prompts = inputs["prompt"]
        bsz = len(prompts)

        prompt_ids, prompt_mask, completion_ids, completion_mask = self._generate(model, prompts)
        contain_eos = torch.any(completion_ids == self.processing_class.eos_token_id, dim=-1)

        logps = self._forward(model, prompt_ids, prompt_mask, completion_ids, completion_mask)
        with torch.no_grad():
            if self.ref_model is not None:
                ref_logps = self._forward(self.ref_model, prompt_ids, prompt_mask, completion_ids, completion_mask)
            else:
                with self.model.disable_adapter() if hasattr(self.model, "disable_adapter") else torch.no_grad():
                    ref_logps = self._forward(self.model, prompt_ids, prompt_mask, completion_ids, completion_mask)

        device = logps.device
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational({"prompt": prompts[0]}):
            completions = [[{"role": "assistant", "content": c}] for c in completions]

        # judge or reward
        if self.judge is not None:
            # Build plain strings to avoid tokenizer-specific tokens
            if is_conversational({"prompt": prompts[0]}):
                env = jinja2.Environment()
                tpl = env.from_string(SIMPLE_CHAT_TEMPLATE)
                str_prompts = [tpl.render(messages=p) for p in prompts]
                str_completions = [tpl.render(messages=c) for c in completions]
            else:
                str_prompts = prompts
                str_completions = completions
            ranks_first = self.judge.judge(str_prompts, list(zip(str_completions[:bsz], str_completions[bsz:])))
            mask = torch.tensor([r == 0 for r in ranks_first], device=device)
        else:
            # reward model path
            rp = 2 * prompts
            if is_conversational({"prompt": rp[0]}):
                ex = [{"prompt": p, "completion": c} for p, c in zip(rp, completions)]
                ex = [apply_chat_template(e, self.reward_processing_class) for e in ex]
                rp = [e["prompt"] for e in ex]
                completions = [e["completion"] for e in ex]
            p_ids = self.reward_processing_class(rp, padding=True, return_tensors="pt", padding_side="left")[
                "input_ids"
            ].to(device)
            ctx_len = p_ids.shape[1]
            c_ids = self.reward_processing_class(completions, padding=True, return_tensors="pt", padding_side="right")[
                "input_ids"
            ].to(device)
            pc_ids = torch.cat([p_ids, c_ids], dim=1)
            with torch.inference_mode():
                _, scores, _ = get_reward(
                    self.reward_model, pc_ids, self.reward_processing_class.pad_token_id, ctx_len
                )
                if self.args.missing_eos_penalty is not None:
                    scores[~contain_eos] -= self.args.missing_eos_penalty
            first, second = scores.split(bsz)
            mask = first >= second

        arange = torch.arange(bsz, device=device)
        chosen_idx = arange + (~mask * bsz)
        rejected_idx = arange + (mask * bsz)

        padding_mask = ~completion_mask.bool()
        cr_idx = torch.cat([chosen_idx, rejected_idx], dim=0)
        cr_logps = logps[cr_idx]
        cr_ref_logps = ref_logps[cr_idx]
        cr_pad = padding_mask[cr_idx]

        cr_logps_sum = (cr_logps * ~cr_pad).sum(1)
        cr_ref_logps_sum = (cr_ref_logps * ~cr_pad).sum(1)
        chosen_sum, rejected_sum = torch.split(cr_logps_sum, bsz)
        chosen_ref_sum, rejected_ref_sum = torch.split(cr_ref_logps_sum, bsz)

        logits = (chosen_sum - rejected_sum) - (chosen_ref_sum - rejected_ref_sum)
        if self.args.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.beta * logits)
        elif self.args.loss_type == "ipo":
            losses = (logits - 1 / (2 * self.beta)) ** 2
        else:
            raise NotImplementedError(f"Invalid loss_type {self.args.loss_type}")
        loss = losses.mean()

        # optional memory relief
        if self.args.torch_empty_cache_steps is not None and self.state.global_step % self.args.torch_empty_cache_steps == 0:
            empty_cache()

        # LOMO optimizers explicitly need LR
        kwargs = {}
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled:
                scaled.backward()
        else:
            self.accelerator.backward(loss, **kwargs)
        return loss.detach() / self.args.gradient_accumulation_steps

    # -----------------------------
    # Logging helpers (average custom stats)
    # -----------------------------
    def _init_stats(self):
        self.stats = {
            "val/contain_eos_token": [],
            "beta": [],
        }

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate=None):
        # Compact version: rely on base Trainer for evaluation/save, just ensure loss is logged.
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:
            logs: dict[str, float] = {}
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            tr_loss -= tr_loss
            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            if grad_norm is not None:
                logs["grad_norm"] = grad_norm.detach().item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            logs["learning_rate"] = learning_rate if learning_rate is not None else self._get_learning_rate()
            # push simple stats if any were filled
            for k, v in list(self.stats.items()):
                if v:
                    logs[k] = sum(v) / len(v)
            self.stats = {k: [] for k in self.stats}
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()
            self.log(logs, start_time)
        return super()._maybe_log_save_evaluate(
            tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval, start_time, learning_rate
        )
