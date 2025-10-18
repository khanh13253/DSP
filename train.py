# train.py
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import pandas as pd
from datasets import Dataset
import torch

print("🔍 Đang tải dữ liệu...")

# Đọc dữ liệu
train_df = pd.read_csv("data/UIT_test_clean.csv", encoding="utf-8-sig")
valid_df = pd.read_csv("data/UIT_valid_clean.csv", encoding="utf-8-sig")

train_df["Emotion"] = train_df["Emotion"].str.capitalize()
valid_df["Emotion"] = valid_df["Emotion"].str.capitalize()

# ✅ LỌC CHỈ 6 CẢM XÚC HỢP LỆ
valid_emotions = {"Anger", "Disgust", "Fear", "Enjoyment", "Sadness", "Surprise", "Other"}
train_df = train_df[train_df["Emotion"].isin(valid_emotions)]
valid_df = valid_df[valid_df["Emotion"].isin(valid_emotions)]

# Đổi tên cột
train_df = train_df.rename(columns={"Clean_sentence": "Sentence", "Emotion": "Emotion"})
valid_df = valid_df.rename(columns={"Clean_sentence": "Sentence", "Emotion": "Emotion"})

# Chuẩn bị label
label2id = {"Anger":0, "Disgust":1, "Fear":2, "Enjoyment":3, "Sadness":4, "Surprise":5, "Other":6}
id2label = {v:k for k,v in label2id.items()}

train_df["Emotion"] = train_df["Emotion"].map(label2id)
valid_df["Emotion"] = valid_df["Emotion"].map(label2id)

# Ép kiểu int
train_df["Emotion"] = train_df["Emotion"].astype(int)
valid_df["Emotion"] = valid_df["Emotion"].astype(int)

# Chuyển thành Dataset
train_dataset = Dataset.from_pandas(train_df)
valid_dataset = Dataset.from_pandas(valid_df)

print(f"✅ Số mẫu train: {len(train_dataset)}")
print(f"✅ Số mẫu valid: {len(valid_dataset)}")

if len(train_dataset) == 0:
    raise ValueError("Train dataset rỗng! Không có mẫu nào thuộc 6 cảm xúc hợp lệ.")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

def tokenize_function(examples):
    return tokenizer(examples["Sentence"], truncation=True, padding=True, max_length=128)

train_tokenized = train_dataset.map(tokenize_function, batched=True)
valid_tokenized = valid_dataset.map(tokenize_function, batched=True)

# Model
model = AutoModelForSequenceClassification.from_pretrained(
    "vinai/phobert-base",
    num_labels=7,
    label2id=label2id,
    id2label=id2label
)

# Cấu hình train
training_args = TrainingArguments(
    output_dir="./phobert-emotion-model",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=50,
    save_total_limit=2,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
)

print("🚀 Bắt đầu huấn luyện...")
trainer.train()

# Lưu model
model.save_pretrained("./phobert-emotion-model-final")
tokenizer.save_pretrained("./phobert-emotion-model-final")
print("✅ Hoàn tất! Model lưu tại: ./phobert-emotion-model-final")