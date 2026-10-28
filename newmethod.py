import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import gc
import json
import warnings
from safetensors.torch import save_model
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==============================================================================
# LEGAL DISCLAIMER
# ==============================================================================
# Educational and Research use ONLY. 
# Replicating proprietary architectures without permission is PROHIBITED. 
# The author holds NO LIABILITY for misuse or copyright infringement.
# ==============================================================================

# ==============================================================================
# 1. DYNAMIC CONFIGURATION
# ==============================================================================
# Users can override these via environment variables.
FRIEND_MODEL_NAME = os.getenv("FFTD_FRIEND_MODEL", "Organization/Friend-Model")
TEACHER_MODEL_NAME = os.getenv("FFTD_TEACHER_MODEL", "Organization/Teacher-Model")
TOKENIZER_NAME = FRIEND_MODEL_NAME 

# Dataset Path (Can be a .txt or .jsonl file)
DATASET_PATH = os.getenv("FFTD_DATASET", "datasets/sample.txt")

STUDENT_SAVE_PATH = "brain/student_fftd.safetensors"

# Fallback texts if dataset is not found
FALLBACK_TEXTS = [
    "Hello world, how are you today?",
    "Artificial Intelligence is the future.",
    "Mathematics is the language of the universe.",
    "Who are you? And what is your origin?",
]

# ==============================================================================
# 2. DATASET LOADER
# ==============================================================================
def load_dataset_texts(path, max_items=20):
    """Loads texts from a file. Supports .txt and .jsonl."""
    if not os.path.exists(path):
        print(f"[WARNING] Dataset not found at '{path}'. Using fallback texts.")
        return FALLBACK_TEXTS

    texts = []
    try:
        if path.endswith(".jsonl"):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if len(texts) >= max_items: break
                    data = json.loads(line)
                    # Try to extract text from common JSON keys
                    text = data.get("text", data.get("content", data.get("raw", "")))
                    if text: texts.append(text)
        else: # Assume .txt
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    if len(texts) >= max_items: break
                    text = line.strip()
                    if len(text) > 10: # Filter out empty lines
                        texts.append(text)
        
        print(f"[INFO] Loaded {len(texts)} texts from '{path}'.")
        return texts if texts else FALLBACK_TEXTS
        
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}. Using fallback texts.")
        return FALLBACK_TEXTS

# ==============================================================================
# 3. HARDWARE DETECTION
# ==============================================================================
def get_device_and_config():
    if torch.cuda.is_available():
        print("[INFO] Hardware: CUDA GPU detected. Using GPU.")
        device = "cuda"
        use_quantization = False
        torch_dtype = torch.float16
    else:
        print("[INFO] Hardware: No CUDA GPU detected. Falling back to CPU.")
        device = "cpu"
        use_quantization = True 
        torch_dtype = torch.float32

    bnb_config = None
    if use_quantization:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        except Exception as e:
            print(f"[WARNING] BitsAndBytes init failed: {e}.")
            bnb_config = None

    return device, bnb_config, torch_dtype

# ==============================================================================
# 4. FFTD LOSS FUNCTION
# ==============================================================================
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

# ==============================================================================
# 5. MAIN EXECUTION PIPELINE
# ==============================================================================
def run_fftd():
    device, bnb_config, torch_dtype = get_device_and_config()
    
    print(f"[INFO] Initializing FFTD Pipeline...")
    print(f"       Friend Model:  {FRIEND_MODEL_NAME}")
    print(f"       Teacher Model: {TEACHER_MODEL_NAME}")
    print(f"       Dataset:       {DATASET_PATH}")
    
    # Load Dataset
    texts = load_dataset_texts(DATASET_PATH)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    
    kwargs = {"device_map": device, "low_cpu_mem_usage": True}
    if bnb_config:
        kwargs["quantization_config"] = bnb_config
    else:
        kwargs["torch_dtype"] = torch_dtype

    # --- STEP 1: Load Models ---
    print("[INFO] Loading Friend model...")
    try:
        friend_model = AutoModelForCausalLM.from_pretrained(FRIEND_MODEL_NAME, **kwargs).eval()
    except Exception:
        kwargs.pop("quantization_config", None)
        kwargs["torch_dtype"] = torch.float16
        friend_model = AutoModelForCausalLM.from_pretrained(FRIEND_MODEL_NAME, **kwargs).eval()

    print("[INFO] Loading Teacher model...")
    try:
        teacher_model = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL_NAME, **kwargs).eval()
    except Exception:
        kwargs.pop("quantization_config", None)
        kwargs["torch_dtype"] = torch.float16
        teacher_model = AutoModelForCausalLM.from_pretrained(TEACHER_MODEL_NAME, **kwargs).eval()

    # --- STEP 2: Extract Logits ---
    print("[INFO] Extracting logits...")
    cached_tensors = {}
    
    with torch.no_grad():
        for i, text in enumerate(texts):
            inputs = tokenizer(text, return_tensors="pt", max_length=32, truncation=True).to(device)
            
            out_f = friend_model(**inputs).logits.cpu().float()
            cached_tensors[f"friend_{i}"] = out_f

            out_t = teacher_model(**inputs).logits.cpu().float()
            cached_tensors[f"teacher_{i}"] = out_t

    # --- STEP 3: Free Memory ---
    print("[INFO] Freeing memory by deleting base models...")
    del friend_model, teacher_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- STEP 4: Train Student ---
    print(f"[INFO] Initializing new Student architecture from config...")
    config = AutoConfig.from_pretrained(FRIEND_MODEL_NAME)
    student = AutoModelForCausalLM.from_config(config)
    
    student = student.to(torch.float16).to(device)
    
    if hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable()
    student.train()

    print(f"[INFO] Student Total Parameters: {sum(p.numel() for p in student.parameters()):,}")

    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-5, eps=1e-6)
    fftd_loss_fn = FFTDLoss(alpha=0.5, beta=0.1, temperature=2.0)

    print("[INFO] Starting FFTD Training...")
    
    for epoch in range(3):
        for i, text in enumerate(texts):
            friend_logits = cached_tensors[f"friend_{i}"].to(device).to(torch.float16)
            teacher_logits = cached_tensors[f"teacher_{i}"].to(device).to(torch.float16)

            inputs = tokenizer(text, return_tensors="pt", max_length=32, truncation=True).to(device)
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

    # --- STEP 5: Save ---
    print(f"[INFO] Saving final Student model to {STUDENT_SAVE_PATH}...")
    os.makedirs(os.path.dirname(STUDENT_SAVE_PATH), exist_ok=True)
    
    student = student.to("cpu")
    save_model(student, STUDENT_SAVE_PATH)
    
    print("[SUCCESS] FFTD Pipeline Complete. Model saved.")

if __name__ == "__main__":
    run_fftd()
