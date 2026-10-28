import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import gc
from safetensors.torch import save_file, load_file, save_model 
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

# ═══════════════════════════════════════════════════════════
# 1. الإعدادات
# ═══════════════════════════════════════════════════════════
FRIEND_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct" 
TEACHER_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct" 
TOKENIZER_NAME = FRIEND_MODEL_NAME

CACHE_PATH = "brain/fftd_cache.safetensors"
STUDENT_SAVE_PATH = "brain/qwen_student_fftd.safetensors"

texts = [
    "Hello world, how are you today?",
    "Artificial Intelligence is the future.",
    "Mathematics is the language of the universe.",
    "Who are you? And what is your origin?",
]

# ═══════════════════════════════════════════════════════════
# 2. إعدادات int4 (4-bit Quantization)
# ═══════════════════════════════════════════════════════════
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# ═══════════════════════════════════════════════════════════
# 3. دالة FFTD
# ═══════════════════════════════════════════════════════════
class FFTDLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.1, temperature=2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def forward(self, student_logits, friend_logits, teacher_logits):
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        friend_soft = F.softmax(friend_logits / self.temperature, dim=-1)
        loss_friend = F.kl_div(student_soft, friend_soft, reduction='batchmean') * (self.temperature ** 2)

        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        loss_teacher = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (self.temperature ** 2)

        sim = F.cosine_similarity(student_logits, friend_logits, dim=-1).mean()
        total_loss = (self.alpha * loss_friend) + ((1 - self.alpha) * loss_teacher) - (self.beta * sim)
        
        return total_loss, {
            "loss_friend": loss_friend.item(),
            "loss_teacher": loss_teacher.item(),
            "sim": sim.item(),
            "total": total_loss.item()
        }

# ═══════════════════════════════════════════════════════════
# 4. استخراج المخرجات (int4)
# ═══════════════════════════════════════════════════════════
def extract_and_save():
    print("⏳ تحميل النماذج (int4)...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    
    friend_model = AutoModelForCausalLM.from_pretrained(
        FRIEND_MODEL_NAME, 
        quantization_config=bnb_config,
        device_map="cpu"
    ).eval()

    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_NAME, 
        quantization_config=bnb_config,
        device_map="cpu"
    ).eval()

    print("✅ تم التحميل. جاري استخراج المخرجات...")
    tensors_to_save = {}
    
    with torch.no_grad():
        for i, text in enumerate(texts):
            inputs = tokenizer(text, return_tensors="pt", max_length=32, truncation=True)
            
            out_f = friend_model(**inputs).logits.cpu().float()
            tensors_to_save[f"friend_{i}"] = out_f

            out_t = teacher_model(**inputs).logits.cpu().float()
            tensors_to_save[f"teacher_{i}"] = out_t

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    save_file(tensors_to_save, CACHE_PATH)
    print(f"💾 تم حفظ المخرجات في {CACHE_PATH}")
    
    del friend_model, teacher_model
    gc.collect()
    print("👉 تم تحرير الرام.")

# ═══════════════════════════════════════════════════════════
# 5. التدريب (Online)
# ═══════════════════════════════════════════════════════════
def train_student():
    print("⏳ تحميل بيانات FFTD...")
    cached_tensors = load_file(CACHE_PATH)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    
    print("⏳ بناء نموذج Qwen الكامل...")
    config = AutoConfig.from_pretrained(FRIEND_MODEL_NAME)
    student = AutoModelForCausalLM.from_config(config)
    
    student = student.to(torch.float16)
    student.gradient_checkpointing_enable()
    student.train()

    print(f"✅ Total Parameters: {sum(p.numel() for p in student.parameters()):,}")

    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-5, eps=1e-6)
    fftd_loss_fn = FFTDLoss(alpha=0.5, beta=0.1, temperature=2.0)

    print("🚀 بدء التدريب...")
    
    for epoch in range(3):
        for i, text in enumerate(texts):
            friend_logits = cached_tensors[f"friend_{i}"].to(torch.float16)
            teacher_logits = cached_tensors[f"teacher_{i}"].to(torch.float16)

            inputs = tokenizer(text, return_tensors="pt", max_length=32, truncation=True)
            student_logits = student(inputs["input_ids"]).logits

            min_len = min(student_logits.size(1), friend_logits.size(1), teacher_logits.size(1))
            min_vocab = min(student_logits.size(-1), friend_logits.size(-1), teacher_logits.size(-1))

            student_logits = student_logits[:, :min_len, :min_vocab]
            friend_logits = friend_logits[:, :min_len, :min_vocab]
            teacher_logits = teacher_logits[:, :min_len, :min_vocab]

            loss, info = fftd_loss_fn(
                student_logits.float(), 
                friend_logits.float(), 
                teacher_logits.float()
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            print(f"Epoch {epoch+1}, Step {i+1}: Total={info['total']:.4f} | Friend={info['loss_friend']:.4f} | Teacher={info['loss_teacher']:.4f}")

    print("💾 حفظ النموذج الجديد (safetensors)...")
    # ✅ استخدام save_model لحل مشكلة الأوزان المشتركة (Weight Tying)
    save_model(student, STUDENT_SAVE_PATH)
    print(f"✅ تم التدريب بنجاح! الملف: {STUDENT_SAVE_PATH}")

if __name__ == "__main__":
    if not os.path.exists(CACHE_PATH):
        extract_and_save()
    
    train_student()