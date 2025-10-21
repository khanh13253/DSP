import os
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)

# === Cấu hình ===
PHOBERT_PATH = "./phobert-emotion-final"
QWEN_MODEL_NAME = "Qwen/Qwen1.5-1.8B-Chat"

# === Tải PhoBERT ===
print("🔄 Đang tải model phân tích cảm xúc (PhoBERT)...")
phobert_model = AutoModelForSequenceClassification.from_pretrained(PHOBERT_PATH)
phobert_tokenizer = AutoTokenizer.from_pretrained(PHOBERT_PATH)
id2label = phobert_model.config.id2label

# === Tải Qwen ===
print("🔄 Đang tải LLM Qwen (lần đầu có thể mất vài phút)...")
qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME, trust_remote_code=True)
qwen_model = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL_NAME,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
)

qwen_pipe = pipeline(
    "text-generation",
    model=qwen_model,
    tokenizer=qwen_tokenizer,
    device_map="auto"
)

# === Hàm hỗ trợ ===
def get_emotion(text):
    inputs = phobert_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(phobert_model.device)
    with torch.no_grad():
        logits = phobert_model(**inputs).logits
        pred_id = logits.argmax().item()
    return id2label[pred_id]

def generate_response(emotion, user_text, max_new_tokens=128):
    prompt = f"""Bạn là một người bạn đồng cảm và nhẹ nhàng. 
Hãy phản hồi ngắn gọn (1-2 câu), chân thành và an ủi người đang cảm thấy "{emotion}".
Không cần lặp lại cảm xúc, chỉ cần thể hiện sự thấu hiểu và động viên.

Lời tâm sự: "{user_text}"

Phản hồi:"""
    
    response = qwen_pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=qwen_tokenizer.eos_token_id
    )
    
    full_text = response[0]['generated_text']
    reply = full_text.split("Phản hồi:")[-1].strip()
    return reply

# === Vòng lặp chat ===
if __name__ == "__main__":
    print("💬 Xin chào! Mình là người bạn đồng cảm. Bạn có thể chia sẻ bất kỳ điều gì.")
    print("Gõ 'exit' để dừng.\n")

    while True:
        try:
            user_input = input("Bạn: ").strip()
            if user_input.lower() in ['exit', 'quit', 'thoát']:
                print("\nTạm biệt! ❤️")
                break
            if not user_input:
                continue

            emotion = get_emotion(user_input)
            reply = generate_response(emotion, user_input)

            print(f"\n🧠 Cảm xúc: {emotion}")
            print(f"💬 Qwen: {reply}\n")

        except KeyboardInterrupt:
            print("\n\nTạm biệt! ❤️")
            break
        except Exception as e:
            print(f"\n❌ Lỗi: {e}")
            print("Vui lòng thử lại.\n")