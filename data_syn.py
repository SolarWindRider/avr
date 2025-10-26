import asyncio
import json
import re
from datasets import load_dataset
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
import os
import PIL
from autogen_agentchat.messages import MultiModalMessage, Image
from tqdm.asyncio import tqdm_asyncio  # ✅ 用于进度条

load_dotenv()


async def run_agent(inputs: list):
    """运行单个 agent 任务"""
    model_client = OpenAIChatCompletionClient(
        model="qwen-vl-max-2025-08-13",
        api_key=os.getenv("Bailian_API_KEY"),
        base_url=os.getenv("Bailian_BASE_URL"),
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": True,
            "family": "unknown",
            "structured_output": True,
        },
    )
    image_path, prompt = inputs[0], inputs[1]
    pil_image = PIL.Image.open(image_path)
    img = Image(pil_image)
    multi_modal_message = MultiModalMessage(content=[prompt, img], source="user")

    agent = AssistantAgent(
        "assistant",
        system_message=(
            "You are a logician who is highly adept at solving abstract visual reasoning tasks, "
            "able to discern patterns and methods of change from given images and provide the correct answer."
        ),
        model_client=model_client,
        model_client_stream=True,
    )

    for _ in range(5):  # 最多重试 5 次
        try:
            result = await agent.run(task=multi_modal_message)
            response = result.messages[-1].content
            js = json.loads(
                re.findall("{.*}", response.replace("\n", "\\n"))[-1].replace("\\n", "")
            )
            answer = list(js.values())[-1]
            break
        except Exception as e:
            print(f"Error: when getting sample: {image_path}\n {e}")
            answer = None
    return answer


async def run_agent_with_limit(semaphore: asyncio.Semaphore, inputs: list):
    """带并发限制的任务"""
    async with semaphore:
        return await run_agent(inputs)


async def run_batch(inputs_li: list[list], max_concurrency: int = 10):
    """批量并发运行，支持最大并发数限制 + tqdm进度条"""
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = [run_agent_with_limit(semaphore, inputs) for inputs in inputs_li]
    results = await tqdm_asyncio.gather(*tasks, desc="Processing", total=len(tasks))
    return results


if __name__ == "__main__":
    train_json_path = "../datas/VisuRiddles/train_dataset.json"
    rawdata = load_dataset("json", data_files={"train": train_json_path})["train"]

    inputs_li = []
    for idx in range(len(rawdata)):
        inputs_li.append(
            [
                "../datas/VisuRiddles/" + rawdata[idx]["imgs"][0],
                f"""question: {rawdata[idx]["question"]}
gold_answer: {rawdata[idx]["gold_answer"]}
As a logician, you need to carefully understand this abstract visual reasoning problem 
and explain why the gold_answer is {rawdata[idx]["gold_answer"]}. 
Then, write your explanation in JSON format:
```json
{{"explanation": "xxx"}}```""",
            ]
        )

    all_results = asyncio.run(run_batch(inputs_li, max_concurrency=10))

    # ✅ 把解释结果加回 dataset
    rawdata = rawdata.add_column("explanation", all_results)

    # ✅ 保存为 JSON 文件
    rawdata.to_json("syndata.json", force_ascii=False)
    print("结果已保存到 syndata.json ✅")
