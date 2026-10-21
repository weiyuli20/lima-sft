"""Compare the base Qwen model with its LoRA fine-tuned adapter."""

import json
import os
import random
from pathlib import Path

import instructor
import torch
from datasets import load_from_disk
from openai import OpenAI
from peft import PeftModel
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


class JudgeResult(BaseModel):
    judge: str
    reason: str


JUDGE_PROMPT = """你是中立评测裁判。
用户指令：
{prompt}

回答A：
{answer_a}

回答B：
{answer_b}

对比两个回答，判断哪个回答更好，评估指令跟随、有用性、事实准确性、流畅度。
如果答案A更好，请输出：
{{
  "reason": "简短理由",
  "judge": "a"
}}
输出格式：
{{
  "reason": "简短理由",
  "judge": "a / b / tie"
}}
"""


def call_judge(prompt, client):
    response = instructor.from_openai(client).chat.completions.create(
        model="qwen3.6-flash",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=15000,
        response_model=JudgeResult,
        extra_body={"enable_thinking": False},
    )
    return response.judge.strip().lower(), response.reason


def inference(model, tokenizer, prompts):
    if isinstance(prompts,str):
        prompts = [prompts]
    messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
    ).to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=2048,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
    res = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    print(res)
    return res


def batched_inference(model, tokenizer, prompts, batch_size=8, desc="inference"):
    """Run inference in batches and return one answer per prompt."""
    answers = []
    for start in tqdm(range(0, len(prompts), batch_size), desc=desc):
        chunk = prompts[start : start + batch_size]
        answers.extend(inference(model, tokenizer, chunk))
    return answers



def save_answers_jsonl(path, prompts, answers):
    if len(prompts) != len(answers):
        raise ValueError(
            f"Prompt/answer count mismatch: prompts={len(prompts)}, answers={len(answers)}"
        )
    with path.open("w", encoding="utf-8") as output_file:
        for sample_id, (prompt, answer) in enumerate(zip(prompts, answers)):
            output_file.write(
                json.dumps(
                    {"id": sample_id, "prompt": prompt, "answer": answer},
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_answers_jsonl(path):
    with path.open(encoding="utf-8") as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def normalized_winner(judge, swapped):
    if judge not in {"a", "b", "tie"}:
        return "tie"
    if judge == "tie":
        return "tie"
    if swapped:
        judge = "b" if judge == "a" else "a"
    return "base" if judge == "a" else "lora"


def evaluate(model_name, lora_path, test_data, judge_client, batch_size=8):
    prompts = [item["prompt"].strip().replace("USER:","").replace("ASSISTANT:","") for item in test_data]
    output_dir = Path(__file__).parent / "eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    expected_ids = list(range(len(prompts)))
    base_path = output_dir / "base_answers.jsonl"
    lora_path_jsonl = output_dir / "lora_answers.jsonl"

    def records_match(records):
        return (
            [record.get("id") for record in records] == expected_ids
            and [record.get("prompt") for record in records] == prompts
            and all("answer" in record for record in records)
        )

    base_records = load_answers_jsonl(base_path) if base_path.exists() else []
    lora_records = load_answers_jsonl(lora_path_jsonl) if lora_path_jsonl.exists() else []
    if not records_match(base_records) or not records_match(lora_records):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        # decoder-only 模型批量生成必须左填充,否则各序列的生成起点会错位
        tokenizer.padding_side = "left"
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, lora_path).eval()

        print("Generating all base-model responses...")
        with model.disable_adapter():
            base_answers = batched_inference(
                model, tokenizer, prompts, batch_size, desc="Base inference"
            )
        save_answers_jsonl(base_path, prompts, base_answers)

        print("Generating all LoRA-model responses...")
        lora_answers = batched_inference(
            model, tokenizer, prompts, batch_size, desc="LoRA inference"
        )
        save_answers_jsonl(lora_path_jsonl, prompts, lora_answers)
        base_records = load_answers_jsonl(base_path)
        lora_records = load_answers_jsonl(lora_path_jsonl)

    print("Evaluating saved responses...")

    eval_results = []
    for sample_id, base_record, lora_record in tqdm(
        zip(expected_ids, base_records, lora_records),
        total=len(prompts),
        desc="LLM judging",
    ):
        prompt = base_record["prompt"]
        base_response = base_record["answer"]
        lora_response = lora_record["answer"]

        swapped = random.random() >= 0.5
        answer_a, answer_b = (
            (lora_response, base_response) if swapped else (base_response, lora_response)
        )
        judge, reason = call_judge(
            JUDGE_PROMPT.format(prompt=prompt, answer_a=answer_a, answer_b=answer_b),
            judge_client,
        )
        eval_results.append(
            {
                "id": sample_id,
                "prompt": prompt,
                "base_response": base_response,
                "lora_response": lora_response,
                "swapped": swapped,
                "judge": judge,
                "winner": normalized_winner(judge, swapped),
                "reason": reason,
            }
        )

    output_path = output_dir / "eval_results.json"
    output_path.write_text(json.dumps(eval_results, ensure_ascii=False, indent=2))
    base_wins = sum(item["winner"] == "base" for item in eval_results)
    lora_wins = sum(item["winner"] == "lora" for item in eval_results)
    ties = sum(item["winner"] == "tie" for item in eval_results)
    print(f"Base wins: {base_wins}, LoRA wins: {lora_wins}, ties: {ties}")


if __name__ == "__main__":
    root = Path(__file__).parent
    judge_client = OpenAI(
        api_key= "sk-ws-H.PEDLPMH.ZN6k.MEUCIDOtXbGPYKTHPSBD0tTRIvk-CX7Jz4K6prvqM3_8uqg9AiEAy2HF9Wk41uF2Rma47ih7A5AI-t6sUJKD846SyFj4z0I",
        base_url="https://llm-mvatcq7dlr50w886.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    )
    test_dataset = load_from_disk(root / "data/lima")["test"]
    evaluate(
        root / "Qwen2.5-7B",
        root / "output/lima_sft_results/final_model_lr5e-5",
        test_dataset.select(range(min(30, len(test_dataset)))),
        judge_client,
    )
