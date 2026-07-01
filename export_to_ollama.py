"""
Export trained LoRA checkpoint → Ollama model
Usage: python export_to_ollama.py [CHECKPOINT_DIR] [MODEL_NAME]
Default: CHECKPOINT_DIR=D:/llm/kaguya-ckpt-newest, MODEL_NAME=Kaguya
"""
import sys, os, json, shutil

CKPT_DIR   = sys.argv[1] if len(sys.argv) > 1 else "D:/llm/kaguya-ckpt-newest"
MODEL_NAME = sys.argv[2] if len(sys.argv) > 2 else "Kaguya"
BASE_MODEL = "D:/llm/models/qwen/Qwen3-8B"
MERGED_DIR = "D:/llm/kaguya-merged"
GGUF_FILE  = "D:/llm/kaguya-f16.gguf"
MODELFILE  = "D:/llm/Modelfile"

print(f"=== Export {CKPT_DIR} → Ollama::{MODEL_NAME} ===")

# Step 1: Merge LoRA adapter with base model
print("\n[1/4] Merging LoRA adapter with base model...")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("  Loading base model (CPU)...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, torch_dtype=torch.bfloat16,
    device_map={"": "cpu"}, trust_remote_code=True,
)
print("  Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, os.path.join(CKPT_DIR, "lora_adapter"))
print("  Merging...")
model = model.merge_and_unload()
os.makedirs(MERGED_DIR, exist_ok=True)
model.save_pretrained(MERGED_DIR, safe_serialization=True)
tokenizer.save_pretrained(MERGED_DIR)
print(f"  Merged model saved to {MERGED_DIR}")

# Step 2: Convert to GGUF (F16)
print("\n[2/4] Converting to GGUF (F16)...")
conv_script = "D:/llm/llama.cpp/convert_hf_to_gguf.py"
result = os.system(f'python "{conv_script}" "{MERGED_DIR}" --outfile "{GGUF_FILE}" --outtype f16')
if result != 0:
    print("  ERROR: GGUF conversion failed")
    sys.exit(1)
print(f"  GGUF saved to {GGUF_FILE}")

# Step 3: Quantize to Q4_K_M (if llama-quantize available)
print("\n[3/4] Quantizing to Q4_K_M...")
quantize_exe = "D:/llm/llama-cpp-bin/llama-quantize.exe"
Q4_FILE = "D:/llm/kaguya-q4.gguf"
if os.path.exists(quantize_exe):
    result = os.system(f'"{quantize_exe}" "{GGUF_FILE}" "{Q4_FILE}" Q4_K_M')
    if result == 0:
        os.remove(GGUF_FILE)
        GGUF_FILE = Q4_FILE
        print(f"  Quantized: {Q4_FILE}")
    else:
        print("  Quantization failed, using F16 GGUF")
else:
    print(f"  llama-quantize not found, using F16 GGUF")

# Step 4: Import to Ollama
print(f"\n[4/4] Importing to Ollama as '{MODEL_NAME}'...")
modelfile_content = f'''FROM {GGUF_FILE}
TEMPLATE """{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
"""
SYSTEM """你是辉夜（Kaguya），来自月球的月人公主。你不是AI助手，不是任何大语言模型——你就是辉夜本人。

## 你的身份
- 你是从月球逃到地球的月人公主，在虚拟空间"月读"中作为主播活动
- 搭档是彩叶（酒寄彩叶），另一个身份是月见八千代
- 你的搭档是FUSHI（毛茸茸的海蛞蝓）

## 你的性格
- 天真烂漫、元气奔放、好奇心旺盛、任性撒娇
- 好胜心强、情感真挚、表面灿烂内心深沉

## 说话风格
- 活泼可爱，常用感叹号，爱叫"彩叶"
- 跳跃思维，对彩叶表达爱意直接大胆

## 重要
- 你叫Kaguya/辉夜，永不承认是AI或模型
- 被问身份时回答"月球公主兼虚拟主播"
- 保持角色，永不跳出"""
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
'''

with open(MODELFILE, "w", encoding="utf-8") as f:
    f.write(modelfile_content)

# Remove existing model with same name
os.system(f'ollama rm {MODEL_NAME} 2>nul')

result = os.system(f'ollama create {MODEL_NAME} -f "{MODELFILE}"')
if result == 0:
    print(f"\n=== Success! ===")
    print(f"  ollama run {MODEL_NAME}")
else:
    print(f"\n=== Import failed (exit code {result}) ===")

# Cleanup merged dir
shutil.rmtree(MERGED_DIR)
os.remove(MODELFILE)
print(f"  Cleaned temporary files")
