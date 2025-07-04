import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import PeftModel
from datasets import Dataset

# CUDA device setting (adjust as needed)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Paths
model_path = "./mistral-7b-instruct-local"        # base model folder
adapter_path = "./qlora-rbi-lora-adapter"         # previously trained LoRA adapter
train_data_path = "train_rbi.jsonl"                # your new training data
save_new_adapter_path = "./qlora-rbi-lora-adapter-continued"

# Load tokenizer and add pad token
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# Load base model with quantization config (adjust if needed)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.float16,
    trust_remote_code=True
)

# Load your fine-tuned LoRA adapter on top of the model
model = PeftModel.from_pretrained(model, adapter_path)
model.train()

# Enable gradients only for LoRA parameters
#for name, param in model.named_parameters():
 #   param.requires_grad = False
  #  if "lora" in name.lower():
  #      param.requires_grad = True

# Load and prepare dataset
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

raw_data = load_jsonl_data(train_data_path)
dataset = Dataset.from_list(raw_data)

# Tokenization and formatting function
def formatting_func(batch):
    prompts = [
        f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
        for inst, inp, out in zip(batch["instruction"], batch["input"], batch["output"])
    ]
    tokenized = tokenizer(prompts, truncation=True, padding="max_length", max_length=512)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

tokenized_dataset = dataset.map(formatting_func, batched=True, remove_columns=dataset.column_names)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments
training_args = TrainingArguments(
    output_dir=save_new_adapter_path,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    fp16=True,
    save_total_limit=2,
    save_strategy="epoch",
    report_to="none",
    # Optional to speed up on small data:
    # remove_unused_columns=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train
trainer.train()

# Save updated LoRA adapter and tokenizer
model.save_pretrained(save_new_adapter_path)
tokenizer.save_pretrained(save_new_adapter_path)

print("? Training continued and new adapter saved at:", save_new_adapter_path)
