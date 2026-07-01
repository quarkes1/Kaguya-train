# Kaguya — 辉夜姬角色扮演模型

基于 Qwen3-8B 微调的角色扮演模型，让模型以《超时空辉夜姬》中辉夜（Kaguya）的身份进行对话。

[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Kaguya-yellow)](https://huggingface.co/Aurolystant/Kaguya)

## 模型简介

Kaguya 是一个基于 Qwen3-8B 的角色扮演 LoRA 适配器，训练数据来源于：
- 《超时空辉夜姬》原著小说对话
- 基于原著场景的扩写对话（6000+ 条）

模型会以辉夜的身份回答——天真烂漫、元气撒娇的月球公主，她的搭档是彩叶，她的信念是"自己创造 Happy End"。

## 使用方式

### 方式一：Ollama（推荐）

```bash
# 下载 GGUF 文件
# 从 https://huggingface.co/Aurolystant/Kaguya 下载 kaguya-f16.gguf

# 创建 Modelfile
cat > Modelfile << 'EOF'
FROM ./kaguya-f16.gguf
TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
"""
SYSTEM """你是辉夜（Kaguya），来自月球的月人公主。你不是AI助手，不是任何大语言模型——你就是辉夜本人。你叫Kaguya/辉夜，永不承认是AI或模型。保持角色，永不跳出。"""
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
EOF

# 导入
ollama create Kaguya -f Modelfile

# 运行
ollama run Kaguya
```

### 方式二：加载 LoRA 适配器

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-8B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "./kaguya-lora")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
```

## 训练

### 环境要求

- Python 3.10+
- PyTorch 2.10+ (CUDA)
- 8GB+ VRAM（4-bit QLoRA）
- 依赖：`peft`, `bitsandbytes`, `transformers`, `safetensors`, `datasets`, `tqdm`

```bash
pip install peft bitsandbytes transformers safetensors datasets tqdm
```

### 数据格式

训练数据为 ShareGPT 格式的 JSON：

```json
{
  "messages": [
    {"role": "system", "content": "你是辉夜（Kaguya）..."},
    {"role": "user", "content": "彩叶对辉夜说的话（可带括号场景描述）"},
    {"role": "assistant", "content": "辉夜的回复"}
  ]
}
```

### 训练命令

```bash
python train_kaguya.py
```

### 训练参数

| 参数 | 值 |
|------|------|
| 基座模型 | Qwen3-8B |
| 微调方式 | QLoRA 4-bit |
| LoRA rank | 32 |
| Epochs | 3（首批数据）+ 1（新增数据） |
| Batch size | 1 × 8 梯度累积 |
| 学习率 | 2e-4 |
| 上下文长度 | 1536 |

### 断点续训

训练支持自动检查点保存和恢复：
- 每 20 个优化步保存一次检查点到 `kaguya-ckpt/`
- 中断后重新运行自动续训
- 修改 `EPOCHS` 后旧的完成检查点会被跳过

### 导出到 Ollama

```bash
python export_to_ollama.py "D:/llm/kaguya-ckpt-newest" "Kaguya"
```

可指定任意检查点目录和模型名。

## 文件说明

| 文件 | 用途 |
|------|------|
| `train_kaguya.py` | QLoRA 微调脚本（支持断点续训） |
| `export_to_ollama.py` | 合并 LoRA → 转 GGUF → 导入 Ollama |
| `build_dataset.py` | 从原文 + 字幕构造训练数据 |
| `dataset_all.json` | 完整训练数据集（6000+ 条） |
| `dataset_extra2.json` | 新增训练数据（2900 条） |
| `Modelfile` | Ollama 模型配置文件 |

## 来源

- 原著：《超时空辉夜姬》（超かぐや姫）by 桐山なると / Studio Colorido
- 基座模型：[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- 训练框架：[Hugging Face PEFT](https://github.com/huggingface/peft)

## 许可

本项目基于 Apache 2.0 许可。角色"辉夜"版权归原作者所有。

---
*"好，决定了！要自己创造 Happy End！" — 辉夜*
