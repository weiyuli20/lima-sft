from transformers import AutoTokenizer, AutoModelForCausalLM
import trl
from trl import SFTConfig,SFTTrainer
from datasets import load_dataset,load_from_disk
from pathlib import Path
import os
from peft import LoraConfig,get_peft_model
import torch


LOCAL_DATA_DIR = Path(__file__).resolve().parent / "data" / "lima"

data = load_from_disk(LOCAL_DATA_DIR)
print("original data:",data)

'''
DatasetDict({
    train: Dataset({
        features: ['prompt', 'response'],
        num_rows: 1029
    })
    test: Dataset({
        features: ['prompt', 'response'],
        num_rows: 300
    })
})

'''



LOCAL_MODEL_DIR = "./Qwen2.5-7B"

model_name = LOCAL_MODEL_DIR if os.path.exists(LOCAL_MODEL_DIR) else "Qwen/Qwen2.5-7B"
print(f"Loading base model {model_name} ...")
model = AutoModelForCausalLM.from_pretrained(model_name,dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token


def format_example(example):
    prompt = example["prompt"].strip()

    if prompt.startswith("USER:"):
        prompt = prompt[len("USER:"):].lstrip()
    if prompt.endswith("ASSISTANT:"):
        prompt = prompt[:-len("ASSISTANT:")].rstrip()

    return {
        "prompt": [{"role": "user", "content": prompt}],
        "completion": [
            {"role": "assistant", "content": example["response"]}
        ],
    }


ds = data.map(
    format_example,
    batched=False,
    remove_columns=["prompt", "response"],
)

print("processed_data:",ds)



# lima_settings={
#     "epoch":"",
#     "learning_rate":"",
#     "batch_size":""
# }

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r = 8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none",
)


model = get_peft_model(model,lora_config)

training_args  = SFTConfig(
    output_dir="./output/lima_sft_results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=20,
    per_device_eval_batch_size=2,
    eval_strategy="steps",
    eval_steps=20,
    report_to="wandb",
    run_name="lima-sft-qwen2.5-7b-lr5e-5",
    bf16 = True,
    max_length=2048,
    completion_only_loss=True,
    # deepspeed="./ds_zero3.json",
)


trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"].select(range(970)),
    eval_dataset=ds["train"].select(range(970,1000)),
    processing_class = tokenizer,
)

model.print_trainable_parameters()


#查看训练数据mask是否正确
dl = trainer.get_train_dataloader()
batch = next(iter(dl))

ids = batch["input_ids"][0]          # 一维: (seq_len,)
lab = batch["labels"][0]


tokens = tokenizer.convert_ids_to_tokens(ids.tolist())
# 标记：✅=算loss，❌=-100屏蔽
res = []
for t, l in zip(tokens, lab):
    if l == -100:
        res.append(f"❌{t}")
    else:
        res.append(f"✅{t}")

print("=== Mask检查 ===")
print(" ".join(res))
print("\n=== 原始文本解码（对照）===")
print(tokenizer.decode(ids.tolist(), skip_special_tokens=False))



print("Starting SFT training... (watch the loss curve)")
trainer.train()


save_path = "./output/lima_sft_results/final_model_lr5e-5"
trainer.save_model(save_path)
print(f"Training complete! The fine-tuned model has been saved to {save_path}.")







