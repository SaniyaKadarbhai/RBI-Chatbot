import  os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_trainingl
from datasets import Dataset

# Local model path 
model_path = "./mistral-7b-instruct-local"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load model from local folder
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map= {"":0},
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Prepare for QLoRA
model = prepare_model_for_kbit_training(model)

# LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# Load tokenizer locally
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_fast=True
)
tokenizer.pad_token = tokenizer.eos_token

# Load dataset
def load_jsonl_data(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                if isinstance(obj, dict):
                    data.append(obj)
            except json.JSONDecodeError:
                continue
    return data

raw_data = load_jsonl_data("pdfQuestions.jsonl")
dataset = Dataset.from_list(raw_data)

# Tokenization
def formatting_func(batch):
    prompts = [
        f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
        for inst, inp, out in zip(batch["instruction"], batch["input"], batch["output"])
    ]
    tokenized = tokenizer(prompts, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(formatting_func, batched=True)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training config
training_args = TrainingArguments(
    output_dir="./qlora-mistral-rbi",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    fp16=True,
    save_total_limit=2,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train
trainer.train()

print("\n? Training completed successfully without errors!")

# ? Save LoRA adapter and tokenizer
model.save_pretrained("qlora-rbi-lora-adapter")
tokenizer.save_pretrained("qlora-rbi-lora-adapter")
print("? LoRA adapter and tokenizer saved!")