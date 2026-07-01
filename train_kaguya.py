"""
Fine-tune Qwen3-8B as Kaguya (辉夜) — pure HuggingFace + bitsandbytes + PEFT
No Unsloth — standard QLoRA with manual training loop
"""
import os, json, shutil
os.environ['HF_HOME'] = 'D:/llm/hf_cache'

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
import time

# ============================================
# Config
# ============================================
MODEL_PATH  = "D:/llm/models/qwen/Qwen3-8B"
OUTPUT_DIR   = "D:/llm/kaguya-lora"
CKPT_DIR      = "D:/llm/kaguya-ckpt"           # Save new checkpoints here
LOAD_CKPT_DIR = "D:/llm/kaguya-ckpt-savedasepoch2"  # Load old weights from here
DATASET_FILE = "D:/llm/training/dataset_all.json"  # New data, 1 epoch only

MAX_SEQ_LENGTH = 1536
SAVE_CKPT_EVERY = 20  # Save checkpoint every N optimizer steps
LORA_R    = 32
LORA_ALPHA = 32
MICRO_BATCH = 1
GRAD_ACCUM  = 8
LEARNING_RATE = 1e-5 #from 2e-4 switch to 1e-5 at the newest step
EPOCHS = 1
WARMUP  = 10

# ============================================
# Load & merge datasets
# ============================================
print("Loading datasets...")
all_messages = []
with open(DATASET_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)
all_messages = [item["messages"] for item in data]
print(f"Loaded {len(all_messages)} conversations from {os.path.basename(DATASET_FILE)}")

# ============================================
# Load model with standard bitsandbytes 4-bit
# ============================================
print(f"\nLoading {MODEL_PATH} (bnb 4-bit)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Prepare for k-bit training + add LoRA
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.config.use_cache = False

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
alloc = torch.cuda.memory_allocated()/1024**3
total = torch.cuda.get_device_properties(0).total_memory/1024**3
print(f"VRAM: {alloc:.1f}G / {total:.1f}G | Trainable: {trainable:,}")

# ============================================
# Tokenize
# ============================================
print("Tokenizing (assistant-only loss masking)...")

IMSTART = "<|im_start|>"
IMEND   = "<|im_end|>"

def tokenize_with_mask(messages):
    """Tokenize with labels only on assistant responses.
    Builds input_ids by concatenating tokenized parts, tracking
    which positions belong to assistant responses."""
    input_ids = []
    labels = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        header = f"{IMSTART}{role}\n"

        # Tokenize header + content + end
        header_ids = tokenizer(header, add_special_tokens=False)["input_ids"]
        content_ids = tokenizer(content, add_special_tokens=False)["input_ids"]
        end_ids = tokenizer(IMEND, add_special_tokens=False)["input_ids"]

        # Add header (always masked)
        input_ids.extend(header_ids)
        labels.extend([-100] * len(header_ids))

        # Add content
        input_ids.extend(content_ids)
        if role == "assistant":
            labels.extend(content_ids)  # Only learn assistant responses
        else:
            labels.extend([-100] * len(content_ids))  # Mask system/user

        # Add end token
        input_ids.extend(end_ids)
        if role == "assistant":
            labels.extend(end_ids)  # Learn to generate <|im_end|>
        else:
            labels.extend([-100] * len(end_ids))

    # Truncate if needed
    if len(input_ids) > MAX_SEQ_LENGTH:
        input_ids = input_ids[:MAX_SEQ_LENGTH]
        labels = labels[:MAX_SEQ_LENGTH]

    return {"input_ids": input_ids, "labels": labels}

# Verify
sample = tokenize_with_mask(all_messages[0])
total_t = len(sample["input_ids"])
train_t = sum(1 for l in sample["labels"] if l != -100)
print(f"  Sample: {total_t} tokens, {train_t} trainable ({100*train_t/total_t:.0f}%)")
decoded = tokenizer.decode([tid for tid, l in zip(sample["input_ids"], sample["labels"]) if l != -100])
print(f"  Learns: {decoded[:100]}...")

tokenized = [tokenize_with_mask(m) for m in all_messages]

def collate(batch):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
    labels    = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        input_ids[i, :L] = torch.tensor(b["input_ids"], dtype=torch.long)
        labels[i, :L]    = torch.tensor(b["labels"], dtype=torch.long)
    return {"input_ids": input_ids, "labels": labels}

loader = DataLoader(tokenized, batch_size=MICRO_BATCH, shuffle=True, collate_fn=collate)

# ============================================
# Optimizer & scheduler
# ============================================
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
total_steps = (len(loader) // GRAD_ACCUM) * EPOCHS
scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP, total_steps)

# ============================================
# Checkpoint save/load
# ============================================
os.makedirs(CKPT_DIR, exist_ok=True)

def save_checkpoint(epoch, step, global_step, optimizer, scheduler, model, loss_history):
    """Save full training state for later resume."""
    torch.save({
        'epoch': epoch,
        'step': step,
        'global_step': global_step,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'loss_history': loss_history,
        'rng_state': torch.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state(),
    }, os.path.join(CKPT_DIR, 'trainer_state.pt'))
    model.save_pretrained(os.path.join(CKPT_DIR, 'lora_adapter'))
    print(f"  [Checkpoint saved at step {global_step}]")

def load_checkpoint(model, optimizer, scheduler):
    """Resume from checkpoint. CKPT_DIR first (unless completed), then LOAD_CKPT_DIR."""
    # Try CKPT_DIR — but skip if it's a completed training (epoch >= EPOCHS)
    state_file = os.path.join(CKPT_DIR, 'trainer_state.pt')
    adapter_file = os.path.join(CKPT_DIR, 'lora_adapter', 'adapter_model.safetensors')
    if os.path.exists(adapter_file):
        # Check if this checkpoint is from a completed training
        is_completed = False
        if os.path.exists(state_file):
            ckpt = torch.load(state_file, map_location='cpu', weights_only=False)
            if ckpt.get('epoch', 0) >= EPOCHS:
                is_completed = True
        if not is_completed:
            print("Resuming from CKPT_DIR checkpoint...")
            from safetensors.torch import load_file
            lora_state = load_file(adapter_file)
            fixed_state = {}
            for k, v in lora_state.items():
                new_k = k.replace(".lora_A.weight", ".lora_A.default.weight")
                new_k = new_k.replace(".lora_B.weight", ".lora_B.default.weight")
                fixed_state[new_k] = v
            model.load_state_dict(fixed_state, strict=False)
            print(f"  LoRA weights loaded ({len(fixed_state)} keys)")
            if not is_completed and os.path.exists(state_file):
                optimizer.load_state_dict(ckpt['optimizer'])
                scheduler.load_state_dict(ckpt['scheduler'])
                torch.set_rng_state(ckpt['rng_state'])
                torch.cuda.set_rng_state(ckpt['cuda_rng_state'])
                print(f"  Resumed at epoch {ckpt['epoch']+1}, step {ckpt['global_step']}")
                return model, ckpt['epoch'], ckpt['step'], ckpt['global_step'], ckpt.get('loss_history', [])
            return model, 0, 0, 0, []
        else:
            print("CKPT_DIR has completed training — loading from LOAD_CKPT_DIR instead")

    # Load weights from LOAD_CKPT_DIR (old trained weights, fresh start)
    adapter_file = os.path.join(LOAD_CKPT_DIR, 'lora_adapter', 'adapter_model.safetensors')
    if os.path.exists(adapter_file):
        print("Loading trained weights from LOAD_CKPT_DIR...")
        from safetensors.torch import load_file
        lora_state = load_file(adapter_file)
        fixed_state = {}
        for k, v in lora_state.items():
            new_k = k.replace(".lora_A.weight", ".lora_A.default.weight")
            new_k = new_k.replace(".lora_B.weight", ".lora_B.default.weight")
            fixed_state[new_k] = v
        model.load_state_dict(fixed_state, strict=False)
        print(f"  LoRA weights loaded ({len(fixed_state)} keys) — fresh optimizer")
        return model, 0, 0, 0, []

    print("No checkpoint found — starting from scratch")
    return model, 0, 0, 0, []

# ============================================
# Manual Training Loop (standard CE loss)
# ============================================
print(f"\n=== Training ===")
print(f"Model: Qwen3-8B QLoRA | Data: {len(tokenized)} samples")
print(f"Epochs: {EPOCHS} | Batch: {MICRO_BATCH}×{GRAD_ACCUM} | LR: {LEARNING_RATE}")
print(f"Steps: ~{total_steps} | Checkpoint every {SAVE_CKPT_EVERY} steps")
print(f"Stop & resume anytime — checkpoint in {CKPT_DIR}\n")

model, start_epoch, start_step, global_step, loss_history = load_checkpoint(model, optimizer, scheduler)

model.train()
accum_loss  = 0.0
start_time  = time.time()
total_micro = len(loader)  # 3038 micro-batches per epoch

for epoch in range(start_epoch, EPOCHS):
    pbar = tqdm(total=total_micro, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for step, batch in enumerate(loader):
        if epoch == start_epoch and step < start_step:
            pbar.update(1)
            continue  # Skip already-completed steps

        batch = {k: v.cuda() for k, v in batch.items()}
        pbar.update(1)

        # Standard CE loss
        outputs = model(**batch)
        loss = outputs.loss / GRAD_ACCUM
        loss.backward()
        accum_loss += loss.item()

        if (step + 1) % GRAD_ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Save checkpoint
            if global_step % SAVE_CKPT_EVERY == 0:
                save_checkpoint(epoch, step + 1, global_step, optimizer, scheduler, model, loss_history)

        # Progress
        if (step + 1) % (GRAD_ACCUM * 5) == 0:
            avg_loss = accum_loss / 5
            free = total - torch.cuda.memory_allocated()/1024**3
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "free": f"{free:.1f}G", "ckpt": global_step//SAVE_CKPT_EVERY})
            accum_loss = 0.0

    start_step = 0  # Reset for next epoch
    print(f"  Epoch {epoch+1} complete — {global_step} steps")
    save_checkpoint(epoch + 1, 0, global_step, optimizer, scheduler, model, loss_history)

elapsed = time.time() - start_time
print(f"\nTraining done in {elapsed/60:.1f} minutes")

# ============================================
# Final save + cleanup checkpoint
# ============================================
print(f"Saving final LoRA to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

import shutil
if os.path.exists(CKPT_DIR):
    shutil.rmtree(CKPT_DIR)
    print(f"Cleaned checkpoint directory (training complete)")

print("Done! LoRA adapter saved.")
print(f"\nTo use with Ollama, merge with base model first or load the adapter.")
