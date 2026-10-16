from transformers import AutoTokenizer, AutoModelForCausalLM
import trl
from trl import SFTConfig,SFTTrainer
from datasets import load_dataset,load_from_disk
from pathlib import Path
import os


LOCAL_DATA_DIR = Path(__file__).resolve().parent / "data" / "lima"

data = load_from_disk(LOCAL_DATA_DIR)
print(data)


LOCAL_MODEL_DIR = "./Qwen2.5-7B-Instruct"

model_name = LOCAL_MODEL_DIR if os.path.exists(LOCAL_MODEL_DIR) else "Qwen/Qwen2.5-7B-Instruct"
print(f"Loading base model {model_name} ...")
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
