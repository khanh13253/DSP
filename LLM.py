import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf" 
OUTPUT_DIR = "./lora_output"
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 2e-4
MAX_LENGTH = 512

# load data (jsonl with instruction/input/output)
data = load_dataset("json", data_files={"train":"train.jsonl","validation":"val.jsonl"})

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

def format_example(ex):
    inp = ex.get("input", "")
    out = ex.get("output", "")
    prompt = f"### Nhiệm vụ: Hãy an ủi và đồng cảm với lời tâm sự dưới đây.\n\n### Tâm sự:\n{inp}\n\n### Phản hồi:\n"
    return {"text": prompt + out + tokenizer.eos_token}

# map to text and tokenize
data = data.map(format_example)
def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH, padding="max_length")
data = data.map(tokenize_fn, batched=True, remove_columns=data["train"].column_names)

train_dataset = data["train"]
eval_dataset = data["validation"]

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", load_in_8bit=True)  # requires bitsandbytes

# PEFT LoRA config
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=50,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model(OUTPUT_DIR)