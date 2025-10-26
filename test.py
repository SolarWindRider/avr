import os
import multiprocessing as mp
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
import json
from qwen_vl_utils import process_vision_info
from argparse import ArgumentParser
from utils.universal import promptTemplates, set_seed, compute_accuracy, model_processor
from peft import PeftModel

mp.set_start_method("spawn", force=True)
# =========固定随机种子============================
set_seed(42)

# ==========参数===================================
parser = ArgumentParser()
parser.add_argument("--model_path", type=str, default="../Downloads/Models/Qwen/Qwen2.5-VL-3B-Instruct")
parser.add_argument("--lora_path", type=str, default=None)
parser.add_argument("--tp", type=int, default=4)  # 32B及以上的模型用8，更小的模型用4
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top_k", type=int, default=20)
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument("--log_path", type=str, default="./logs")
parser.add_argument("--ptType", type=str, default="Direct,COT,Naive")
args = parser.parse_args()
print(args)


# ========== 融合模型 ==========================
def merge():
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


# ========== 构造图文 prompt ==========
def build_prompts(dataset, processor, ptType):
    prompts = []
    references = []
    id = []
    issudoku = []
    for example in dataset:
        image_path = example["image_path"]

        option = f"option: {example['option']}\n" if example["option"] != "" else ""
        question = example["question"] + option + promptTemplates[ptType] + 'Write the answer into a JSON form\n```json\n{"answer": "X"}```'
        answer = example["answer"]

        img_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_path,
                    },
                    {"type": "text", "text": question},
                ],
            },
        ]

        # 构造 Qwen2.5-VL 的 prompt
        prompt = processor.apply_chat_template(img_messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _, _ = process_vision_info(img_messages, return_video_kwargs=True)

        mm_data = {}
        if image_inputs is not None:
            mm_data["image"] = image_inputs

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": mm_data,
        }
        prompts.append(llm_inputs)
        references.append(answer)
        id.append(example["id"])
        issudoku.append(example["issudoku"])
    return prompts, references, id, issudoku


# ========== 使用 vLLM 推理 ==========
def test_model_vllm(llm, processor, dataset, bench, log_path, ptType, max_new_tokens=8000):
    prompts, references, id, issudoku = build_prompts(dataset, processor, ptType)

    sampling_params = SamplingParams(
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, max_tokens=max_new_tokens, stop=["<|im_end|>", "<|endoftext|>"]
    )

    outputs = llm.generate(prompts, sampling_params)
    completions = [o.outputs[0].text.strip() for o in outputs]

    test_log = compute_accuracy(prompts, completions, references, bench, id, issudoku)

    json.dump(test_log, open(f"{log_path}/{bench}_{ptType}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)


# ========== 加载多模态图文问答数据 ==========
def preprocess_multimodal_dataset(bench):
    if bench == "VisuRiddles":
        dsjson = json.load(open("../datas/VisuRiddles/test_dataset.json"))
        base_image_dir = "../datas/VisuRiddles"
        ds = []
        for example in dsjson:
            ds.append(
                {
                    "image_path": base_image_dir + "/" + example["imgs"][0],
                    "question": example["question"],
                    "option": example["option"],
                    "answer": example["gold_answer"],
                    "id": example["id"],
                    "issudoku": True if "sudoku" in example["class"] else False,
                }
            )
        return ds
    elif bench == "RAVEN":
        dsjson = json.load(open("../datas/RAVEN/raven_test.json"))
        base_image_dir = "../datas/RAVEN"
        ds = []
        for example in dsjson:
            ds.append(
                {
                    "image_path": base_image_dir + "/" + example["images"][0],
                    "question": "Which one of the options is the correct answer for the question?",
                    "option": "A, B, C, D, E, F, G, H.",
                    "answer": example["messages"][1]["content"],
                    "id": example["images"][0],
                    "issudoku": False,
                }
            )
        return ds
    elif bench == "MARVEL":
        ds = []
        for i in range(770):  # MARVEL总共只有770个样本
            idx = i + 1
            js = json.load(open(f"../datas/MARVEL_AVR/Json_data/{idx}/{idx}_label.json"))
            image_path = f"../datas/MARVEL_AVR/Json_data/{idx}/{idx}.png"
            ds.append(
                {
                    "image_path": image_path,
                    "question": js["avr_question"],
                    "option": "",  # MARVEL原本的question里面包含options
                    "answer": str(js["answer"]),
                    "id": idx,
                    "issudoku": False,
                }
            )
        return ds
    elif bench == "LogicVista":
        dsjson = json.load(open("../datas/LogicVista/data/dataset.json"))
        base_image_dir = "../datas/LogicVista/data/images"
        ds = []
        for key in dsjson.keys():
            ds.append(
                {
                    "image_path": base_image_dir + "/" + dsjson[key]["imagename"],
                    "question": dsjson[key]["question"],
                    "option": "",  # LogicVista 原本的question里面包含options
                    "answer": dsjson[key]["answer"],
                    "id": key,
                    "issudoku": False,
                }
            )
        return ds
    elif bench == "PuzzleVQA":
        base_image_dir = "../datas/LLM-PuzzleTest/PuzzleVQA/data"
        dsjson = []
        for each in os.listdir(base_image_dir):
            if ".json" not in each:
                continue
            with open(base_image_dir + "/" + each, "r", encoding="utf-8") as f:
                li = f.readlines()
            dsjson += li
        ds = []
        for example in dsjson:
            example = json.loads(example)
            ds.append(
                {
                    "image_path": base_image_dir + "/" + example["image"],
                    "question": example["question"],
                    "option": str(example["options"])[1:-1] + ".",
                    "answer": example["answer"],
                    "id": example["image"].split("/")[-1],
                    "issudoku": False,
                }
            )
        return ds
    elif bench == "AlgoPuzzleVQA":
        base_image_dir = "../datas/LLM-PuzzleTest/AlgoPuzzleVQA/data"
        dsjson = []
        for each in os.listdir(base_image_dir):
            if ".json" not in each:
                continue
            with open(base_image_dir + "/" + each, "r", encoding="utf-8") as f:
                li = f.readlines()
            dsjson += li
        ds = []
        for example in dsjson:
            example = json.loads(example)
            ds.append(
                {
                    "image_path": base_image_dir + "/" + example["image"],
                    "question": example["question"],
                    "option": str(example["options"])[1:-1] + ".",
                    "answer": example["answer"],
                    "id": example["image"].split("/")[-1],
                    "issudoku": False,
                }
            )
        return ds


# ========== 主程序入口 ==========
if __name__ == "__main__":
    os.makedirs(args.log_path, exist_ok=True)
    merge()

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    llm = LLM(model=args.model_path, trust_remote_code=True, tensor_parallel_size=args.tp, max_model_len=15000)  # 自动多卡推理

    for ptType_ in args.ptType.split(","):
        ptType = ptType_.strip()
        for bench in ["VisuRiddles", "RAVEN", "MARVEL", "LogicVista", "PuzzleVQA", "AlgoPuzzleVQA"]:
            dataset = preprocess_multimodal_dataset(bench)
            test_model_vllm(llm, processor, dataset, bench, args.log_path, ptType)
