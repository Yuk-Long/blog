---
title: "iMatch 项目复现记录"
date: 2026-06-01
description: "记录我复现 iMatch 项目的完整过程，从连接服务器到跑通模型的全流程"
tags: ["项目", "复现", "服务器", "Linux"]
readingTime: 8
author: "丝丝大魔王"
---


## 项目背景

iMatch（Instruction-augmented Multimodal Alignment for Image-Text and Element Matching）是 **NTIRE 2025 文本到图像生成模型质量评估挑战赛** 图像-文本对齐赛道的**冠军方法**，相关工作发表在 CVPRW 2025 上。

### 挑战赛介绍

NTIRE 2025 挑战赛聚焦于 AI 生成图像的质量评估，分为两个赛道：

- **对齐赛道（Alignment Track）**：评估生成图像与文本描述之间的匹配度，使用 **EvalMuse-40K** 数据集（约 4 万对图文，含细粒度对齐标注）
- **结构赛道（Structure Track）**：检测生成图像中的结构失真，使用 EvalMuse-Structure 数据集

比赛共吸引 **582 名注册参与者**，收到超过 3000 次提交。

### iMatch 核心思路

iMatch 的核心创新在于：

1. **QAlign 策略**：将 MOS 分数（1-5）离散化为 15 个字母等级（a-o），把回归问题转化为分类问题，并通过 softmax 加权还原连续分数
2. **多模态大模型微调**：基于 Qwen2.5-VL-7B，用 LoRA 高效微调，让模型学会图文间的细粒度对应关系
3. **数据增强**：包含四种增强策略，其中验证集增强让模型在训练过程中生成高质量伪标签再合并回训练集

我选择复现这个方法，一方面是因为它在挑战赛中的表现突出，另一方面它的技术方案（LoRA + QAlign）对硬件要求相对友好，适合个人学习。


## 第一步：连接 5880 服务器

> 参考教程：[连接 Linux 服务器](https://task.gemlab.site/w/connect-to-linux-server/)

5880 这张 GPU 卡放在机房，不能直接从外网访问，需要先通过 **跳板机（Jump Server）** 中转。

核心链路：

> 本地电脑 → 跳板机 → 5880 服务器

### 1.1 配置 SSH 公钥

先在本地生成 SSH 密钥对，然后把公钥上传到[公钥管理平台](https://task.gemlab.site/w/pubkey)。登录时用对应的**私钥**做身份验证，不需要每次输密码。

```shell
# 如果还没有密钥对，先生成一个
ssh-keygen -t ed25519 -C "你的邮箱"

# 查看公钥内容，复制到平台
cat ~/.ssh/id_ed25519.pub
```

### 1.2 配置 SSH Config

在本地 `~/.ssh/config` 里配置跳板机和目标服务器的连接信息：

```
Host jump
    HostName 跳板机地址
    User 你的用户名
    IdentityFile ~/.ssh/id_ed25519

Host gpu5880
    HostName 5880服务器内网地址
    User 你的用户名
    IdentityFile ~/.ssh/id_ed25519
    ProxyJump jump
```

这样配好后，一条命令就能直连 5880：

```shell
ssh gpu5880
```

### 1.3 验证连接

登录成功后，检查 GPU 是否可用：

```shell
nvidia-smi
```

看到 GPU 信息就说明一切正常。

---


## 第二步：创建项目目录和配置文件

连上 5880 服务器后，开始搭建项目骨架。

### 2.1 创建项目目录结构

```shell
mkdir -p ~/imatch/{configs,data,train,inference,augmentation,scripts,utils}
```

这会在 home 目录下建好 imatch 项目，包含：
- `configs/` — 配置文件
- `data/` — 数据相关
- `train/` — 训练代码
- `inference/` — 推理代码
- `augmentation/` — 数据增强
- `scripts/` — 辅助脚本
- `utils/` — 工具函数

### 2.2 创建配置文件

```yaml
data:
  root: "/home/lyl/EvalMuse-40K"
  train_json: "train.json"
  val_json: "val.json"
  image_dir: "images"

model:
  name: "Qwen/Qwen2.5-VL-7B-Instruct"
  torch_dtype: "bfloat16"
  attn_implementation: "flash_attention_2"

qalign:
  score_min: 1.0
  score_max: 5.0
  num_levels: 15
  letters: "abcdefghijklmno"

lora:
  r: 64
  alpha: 128
  dropout: 0.05
  target_modules: ["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"]

training:
  output_dir: "./outputs/imatch_run1"
  num_epochs: 3
  per_device_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-5
  max_seq_length: 2048
  logging_steps: 10
  save_steps: 200
  eval_steps: 200
  bf16: true
  tf32: true
  gradient_checkpointing: true

inference:
  batch_size: 4
  max_new_tokens: 10
```

配置关键点：
- **模型**：用的是 Qwen2.5-VL-7B-Instruct，bfloat16 精度，Flash Attention 2 加速
- **数据**：EvalMuse-40K 数据集，放在 `/home/lyl/EvalMuse-40K`
- **训练**：LoRA 微调（r=64），3 个 epoch，学习率 2e-5，开了 bf16 和梯度检查点节省显存
- **Q-Align**：分数范围 1-5 分，离散化为 15 个等级，用字母 a-o 表示

---


## 第三步：QAlign 数据处理模块

创建 `~/imatch/data/dataset.py`，实现分数到字母标签的转换、数据集加载和批处理。

### 3.1 QAlign 配置类

```python
@dataclass
class QAlignConfig:
    score_min: float = 1.0
    score_max: float = 5.0
    num_levels: int = 15
    letters: str = "abcdefghijklmno"
```

定义了分数范围（1-5）、离散化等级（15 级）和对应的字母表（a-o）。

### 3.2 核心函数

**分数 → 字母：** `score_to_letter()` 把连续的 MOS 分数（1-5）映射到 15 个字母标签之一。

**字母 → 分数：** `qalign_decode()` 在推理时把模型输出的字母 logits 通过 softmax 加权还原为连续分数，用了加权平均而非 argmax，更平滑。

### 3.3 数据集类 `IMatchDataset`

```python
class IMatchDataset(Dataset):
    def __init__(self, data_path, image_dir, processor, tokenizer, qalign_cfg, max_length=2048):
```

- 读取 JSON 格式的标注数据（图片名、prompt、分数）
- 随机选择指令模板（`INSTRUCTION_TEMPLATES`），增加训练多样性
- 用 processor 的 `apply_chat_template` 构造多模态对话格式（system + user + image）
- 标签只放在序列最后一个 token 上，其余位置用 -100 忽略

### 3.4 批处理 `collate_fn`

```python
def collate_fn(batch):
    labels[i, b["attention_mask"].sum().item() - 1] = b["label"]
```

关键设计：label 只放在每个样本的最后一个有效 token 位置（attention_mask 最后一个 1 的位置），因为 QAlign 只需要模型在序列末尾输出一个字母。

### 完整代码

```python
"""
iMatch 数据处理：分数 → 字母标签 → QAlign 解码
"""
import json, random
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset


@dataclass
class QAlignConfig:
    score_min: float = 1.0
    score_max: float = 5.0
    num_levels: int = 15
    letters: str = "abcdefghijklmno"

    @property
    def letter_list(self): return list(self.letters)

    @property
    def letter_to_idx(self): return {c: i for i, c in enumerate(self.letter_list)}


def score_to_letter(score: float, cfg: QAlignConfig) -> str:
    """连续分数 [1,5] → 字母标签 {a..o}"""
    scaled = (score - cfg.score_min) / (cfg.score_max - cfg.score_min) * (cfg.num_levels - 1) + 1
    idx = max(0, min(cfg.num_levels - 1, int(round(scaled)) - 1))
    return cfg.letter_list[idx]


def qalign_decode(logits: torch.Tensor, cfg: QAlignConfig, tokenizer) -> float:
    """QAlign 软解码：字母 logits → softmax 概率加权 → [1,5] 连续分数"""
    letter_ids = tokenizer.encode(cfg.letters, add_special_tokens=False)
    if logits.dim() == 3:
        logits = logits[0, -1, :]
    elif logits.dim() == 2:
        logits = logits[-1, :]
    letter_logits = logits[letter_ids]
    probs = torch.softmax(letter_logits.float(), dim=-1)
    weights = torch.arange(1, cfg.num_levels + 1, device=probs.device).float()
    score_scaled = (probs * weights).sum().item()
    return cfg.score_min + (score_scaled - 1) / (cfg.num_levels - 1) * (cfg.score_max - cfg.score_min)


SYSTEM_MESSAGE = "You are an expert in evaluating the alignment between images and text descriptions."

INSTRUCTION_TEMPLATES = [
    "Rate the alignment between this image and the text: {prompt}",
    "How well does this image align with the description: {prompt}?",
]


def format_instruction(prompt: str, element_labels=None) -> str:
    template = random.choice(INSTRUCTION_TEMPLATES)
    if element_labels:
        el_str = ", ".join(f"{k}={v:.1f}" for k, v in element_labels.items())
        return f"Rate the alignment: {prompt}\nElement scores: {el_str}\nOverall score?"
    return template.format(prompt=prompt)


class IMatchDataset(Dataset):
    def __init__(self, data_path, image_dir, processor, tokenizer, qalign_cfg, max_length=2048):
        self.image_dir = Path(image_dir)
        self.processor = processor
        self.tokenizer = tokenizer
        self.qalign_cfg = qalign_cfg
        self.max_length = max_length
        with open(data_path, "r") as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} samples")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        from PIL import Image
        sample = self.data[idx]
        image = Image.open(self.image_dir / sample["image"]).convert("RGB")
        prompt = sample["prompt"]
        score = float(sample["score"])
        instruction = format_instruction(prompt)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": instruction},
            ]},
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True, max_length=self.max_length, truncation=True)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["label"] = self.tokenizer.encode(score_to_letter(score, self.qalign_cfg), add_special_tokens=False)[0]
        return inputs


def collate_fn(batch):
    from torch.nn.utils.rnn import pad_sequence
    input_ids = pad_sequence([b["input_ids"] for b in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
    labels = torch.full_like(input_ids, -100)
    for i, b in enumerate(batch):
        labels[i, b["attention_mask"].sum().item() - 1] = b["label"]
    result = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    if "pixel_values" in batch[0]:
        result["pixel_values"] = torch.stack([b["pixel_values"] for b in batch])
    if batch[0].get("image_grid_thw") is not None:
        result["image_grid_thw"] = torch.stack([b["image_grid_thw"] for b in batch])
    return result
```

---


## 第四步：训练脚本

创建 `~/imatch/train/train.py`，实现 LoRA 微调 Qwen2.5-VL-7B 的完整训练流程。

### 4.1 整体流程

脚本分四步走：

1. **加载模型**：Qwen2.5-VL-7B + LoRA 适配器，确保 `lm_head` 可训练
2. **加载数据**：用 HuggingFace `Dataset` 做预处理，把分数转为字母标签
3. **训练**：Trainer 自动管理梯度累积、checkpoint、评估
4. **保存**：最终模型和 processor 一起存到 `output_dir/final_model`

### 4.2 关键设计

- **`lm_head` 强制可训练**：LoRA 默认不训练 `lm_head`，但 QAlign 需要在序列末尾分类字母，所以手动解开
- **`collate` 函数**：label 只放在每个样本最后一个有效 token，避免整个序列都参与损失计算
- **训练配置**：cosine 学习率调度、warmup 3%、权重衰减 0.01、保留最优 3 个 checkpoint

### 完整代码

```python
"""
iMatch 训练：QAlign 分类 + LoRA 微调 Qwen2.5-VL-7B
"""
import os, sys, argparse, yaml, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset as HFDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset import QAlignConfig, score_to_letter, format_instruction, SYSTEM_MESSAGE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--train_data", required=True)
    p.add_argument("--val_data", required=True)
    p.add_argument("--image_dir", required=True)
    return p.parse_args()


def preprocess(examples, processor, qalign_cfg, max_length, image_dir):
    from PIL import Image
    texts, images_list, letters = [], [], []
    for prompt, score, img_name in zip(examples["prompt"], examples["score"], examples["image"]):
        instruction = format_instruction(prompt)
        img = Image.open(os.path.join(image_dir, img_name)).convert("RGB")
        letter = score_to_letter(float(score), qalign_cfg)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": instruction}]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
        images_list.append(img)
        letters.append(letter)
    return {"text": texts, "images": images_list, "letter": letters}


def collate(batch, processor, max_length=2048):
    texts = [b["text"] for b in batch]
    images = [b["images"] for b in batch]
    letters = [b["letter"] for b in batch]
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    labels = torch.full_like(inputs["input_ids"], -100)
    for i, (ltr, mask) in enumerate(zip(letters, inputs["attention_mask"])):
        tok_id = processor.tokenizer.encode(ltr, add_special_tokens=False)[0]
        labels[i, mask.sum().item() - 1] = tok_id
    inputs["labels"] = labels
    return inputs


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    qalign_cfg = QAlignConfig(**cfg["qalign"])
    model_cfg, train_cfg = cfg["model"], cfg["training"]

    print("[1/4] Loading model & processor...")
    attn = model_cfg.get("attn_implementation", "flash_attention_2")
    dtype = getattr(torch, model_cfg["torch_dtype"])
    processor = AutoProcessor.from_pretrained(model_cfg["name"], trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_cfg["name"], torch_dtype=dtype, attn_implementation=attn, trust_remote_code=True, device_map="auto")

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"], lora_dropout=lora_cfg["dropout"],
                              target_modules=lora_cfg["target_modules"], bias="none", task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, peft_config)
    for n, p in model.named_parameters():
        if "lm_head" in n:
            p.requires_grad = True
    model.print_trainable_parameters()

    print("[2/4] Loading data...")
    with open(args.train_data) as f:
        train_raw = json.load(f)
    with open(args.val_data) as f:
        val_raw = json.load(f)
    train_ds = HFDataset.from_list(train_raw)
    val_ds = HFDataset.from_list(val_raw)

    pp = lambda x: preprocess(x, processor, qalign_cfg, train_cfg.get("max_seq_length", 2048), args.image_dir)
    train_ds = train_ds.map(pp, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(pp, batched=True, remove_columns=val_ds.column_names)
    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_ds.set_format(type="dict")
    val_ds.set_format(type="dict")

    print("[3/4] Training...")
    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        eval_steps=train_cfg["eval_steps"],
        save_total_limit=3,
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="tensorboard",
        logging_dir=os.path.join(train_cfg["output_dir"], "logs"),
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=lambda b: collate(b, processor, train_cfg.get("max_seq_length", 2048)),
        processing_class=processor.tokenizer,
    )
    trainer.train()

    print("[4/4] Saving model...")
    final_path = os.path.join(train_cfg["output_dir"], "final_model")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    print(f"Done! Model saved to {final_path}")


if __name__ == "__main__":
    main()
```

---


## 第五步：推理脚本

创建 `~/imatch/inference/predict.py`，用训练好的模型对测试集打分。

### 5.1 核心流程

1. **加载模型**：从保存路径加载 LoRA 微调后的 Qwen2.5-VL-7B
2. **逐样本推理**：每张图片 + prompt → 模型生成字母 → QAlign 软解码回分数
3. **结果保存**：输出 `predictions.json`，包含图片名、prompt 和预测分数

### 5.2 关键细节

- **`predict_single`**：用 `model.generate()` 生成 1 个字母，取生成 token 的 logits 做 QAlign 软解码，而非直接 argmax
- **`do_sample=False, temperature=None`**：贪心解码，确保推理结果确定性
- **`qalign_decode`**：复用数据处理模块的软解码函数，分数精度到 4 位小数

### 完整代码

```python
"""
iMatch 推理：QAlign 软解码
"""
import os, sys, json, argparse, yaml
from pathlib import Path
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.dataset import QAlignConfig, qalign_decode, format_instruction, SYSTEM_MESSAGE


def load_model(model_path, device="auto"):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    print(f"Loading model from {model_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
        trust_remote_code=True, device_map=device)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    return model, processor


def predict_single(model, processor, image, prompt, qalign_cfg):
    instruction = format_instruction(prompt)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MESSAGE}]},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": instruction}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False, temperature=None,
                                  return_dict_in_generate=True, output_logits=True,
                                  pad_token_id=processor.tokenizer.pad_token_id)
    return qalign_decode(outputs.logits[-1][0], qalign_cfg, processor.tokenizer)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--test_data", required=True)
    p.add_argument("--output", default="predictions.json")
    p.add_argument("--image_dir", required=True)
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    qalign_cfg = QAlignConfig(**cfg["qalign"])
    model, processor = load_model(args.model)

    with open(args.test_data) as f:
        test_data = json.load(f)

    results = []
    for sample in tqdm(test_data, desc="Inference"):
        img = Image.open(os.path.join(args.image_dir, sample["image"])).convert("RGB")
        score = predict_single(model, processor, img, sample["prompt"], qalign_cfg)
        results.append({"image": sample["image"], "prompt": sample.get("prompt", ""), "predicted_score": round(score, 4)})

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDone! {len(results)} predictions saved to {args.output}")


if __name__ == "__main__":
    main()
```

---


## 第六步：安装依赖

所有脚本准备好后，安装 Python 依赖。

```shell
cd ~/imatch
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

分步解释：

- **`python3 -m venv venv`** — 创建虚拟环境，隔离项目依赖，不影响系统 Python
- **`source venv/bin/activate`** — 激活虚拟环境，终端前会出现 `(venv)` 提示
- **`pip install -r requirements.txt`** — 根据依赖清单安装所有包（torch、transformers、peft 等）

之后每次登录服务器，都要先 `source venv/bin/activate` 激活环境再操作。

---


## 第七步：准备数据集 EvalMuse-40K

数据集托管在 HuggingFace，需要下载、解压、并适配代码的字段结构。

### 7.1 下载数据集

```shell
git clone https://huggingface.co/datasets/DY-Evalab/EvalMuse
```

LFS 大文件约 51GB，需要合并分片后解压：

```shell
cat images.zip.part-* > images_full.zip
unzip -d ~/EvalMuse-40K/images images_full.zip
cp train_list.json test.json ~/EvalMuse-40K/
```

### 7.2 数据字段映射

JSON 中的实际字段需要和代码一致：

| JSON 字段        |含义                  | 处理方式                    |
|-----------------|----------------------|----------------------------|
| `img_path`      | 图片路径              | 拼接 `image_dir`          |
| `total_score`   | 3 个标注员分数数组     | `sum/3` 取平均             |
| `element_score` | 元素级二值标注         | 平均后注入 prompt 做 CoT    |
| `type`          | real / synthetic      | 注入 prompt 类型前缀         |

### 7.3 切分训练/验证集

`test.json` 无标签，不做验证。从训练集随机切 10% 作为验证集：

```python
import json, random

with open('train_list.json') as f:
    data = json.load(f)

random.seed(42)
random.shuffle(data)

split = int(len(data) * 0.9)
json.dump(data[:split], open('train_split.json', 'w'))
json.dump(data[split:], open('val_split.json', 'w'))
```


---

## 第八步：数据预处理与 QAlign 分数转换

## 8.1 数据字段对齐

EvalMuse-40K 的实际字段与代码约定存在差异，需要手动映射：

| JSON 字段 | 含义 | 处理方式 |
|-----------|------|----------|
| `img_path` | 图片路径，如 `SDXL-Turbo/00110.png` | 拼接 `image_dir` 基路径 |
| `total_score` | 三个标注员的打分数组，如 `[4,3,3]` | `sum / 3` 取均值 |
| `element_score` | 元素级二值标注，如 `{"puffin": [1,0,1]}` | 均值化后用于元素增强 |
| `type` | `"real"` 或 `"synthetic"` | 注入 prompt 前缀做类型增强 |
| `fidelity_label` | 保真度标签，如 `"真实场景-动物"` | 本次复现暂未使用 |

## 8.2 划分训练 / 验证集

`test.json` 中的 `total_score` 字段全部为 `null`（测试集无标签），无法直接用做验证。从 `train_list.json` 按 9:1 随机切分：

```bash
python3 -c "
import json, random
with open('train_list.json') as f: data = json.load(f)
random.seed(42); random.shuffle(data)
split = int(len(data) * 0.9)
json.dump(data[:split], open('train_split.json','w'))
json.dump(data[split:], open('val_split.json','w'))
"
```

切分后：**训练集 29,445 条，验证集 3,272 条**。

## 8.3 QAlign 分数转换

### 原理

iMatch 的核心设计之一：**将回归问题转化为分类问题**。

1. **训练时**：原始分数 `[1, 5]` 线性映射到 `[1, 15]` 的 15 个离散等级，对应字母 `{a, b, ..., o}`，作为分类标签训练（CrossEntropy Loss）。
2. **推理时**：取最后一个 token 对应 15 个字母的 logits → 闭集 softmax 得到概率分布 → 概率加权求和插值出连续分数。

### 示例

```
连续分数 3.72 → 缩放 [1,15] → 10.52 → 四舍五入 → letter "k"
训练目标：模型输出 token "k"（CE Loss）
推理：{a..o} 15 个 logits → softmax → Σ(prob_i × i) → 映射回 [1,5]
```

### 相比传统回归头的优势

传统回归头使用 `MLP(768 → 1)` 配合 MSE Loss，在小范围分数（1-5）上训练不稳定、对标注噪声敏感。QAlign 利用了大模型天然的离散输出能力，用分类目标替代回归目标，既能复用预训练权重，又通过概率插值保留了连续分数的精度。

### 实现代码

```python
LETTERS = "abcdefghijklmno"

def score_to_letter(score: float) -> str:
    """[1,5] → scaled [1,15] → letter {a..o}"""
    scaled = (score - 1) / 4 * 14 + 1
    idx = max(0, min(14, int(round(scaled)) - 1))
    return LETTERS[idx]

def qalign_decode(logits, tokenizer) -> float:
    """推理时：15个字母logits → 概率加权 → [1,5]连续分数"""
    letter_ids = tokenizer.encode(LETTERS, add_special_tokens=False)
    if logits.dim() == 3: logits = logits[0, -1, :]
    elif logits.dim() == 2: logits = logits[-1, :]
    probs = torch.softmax(logits[letter_ids].float(), dim=-1)
    weights = torch.arange(1, 16, device=probs.device).float()
    score_scaled = (probs * weights).sum().item()  # [1, 15]
    return 1 + (score_scaled - 1) / 14 * 4           # 映射回 [1, 5]
```

## 8.4 图片分辨率限制与显存优化

Qwen2.5-VL-7B 全模型 bf16 单卡训练约需 45GB 显存。为在 48GB RTX 5880 Ada 上安全运行，将最大图片分辨率限制为 `256 × 28²`（约 200K pixels）：

```python
processor = AutoProcessor.from_pretrained(
    model_name,
    min_pixels=56 * 28 * 28,
    max_pixels=256 * 28 * 28
)
```

图片被切分为约 8 × 8 个 patch，对物体级判断任务影响轻微，同时显存占用显著下降。辅助措施：

| 策略 | 作用 |
|------|------|
| `gradient_checkpointing_enable()` | 用计算换显存，前向不缓存中间激活 |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | 防止显存碎片化导致 OOM |
| `batch_size=1` + 梯度累积 8 步 | 等效 batch = 8，单步显存足够 |
| `bf16` 混合精度 | 比 fp32 节省一半显存 |

## 8.5 踩坑记录

| 问题 | 原因 | 解决 |
|------|------|------|
| `huggingface-cli` 命令失效 | 新版 HF CLI 已替换为 `hf` | 改用 `hf download` |
| 数据集仓库名返回 401 | 实际为 `DY-Evalab/EvalMuse`，非论文中列出的名称 | 通过 GitHub 官方仓库 clone |
| 51GB 分片无法直接使用 | LFS 大文件分片存储 | `cat images.zip.part-* > images.zip` 合并后解压 |
| HFDataset.map() 预处理 OOM | Arrow 格式无法存储变长 `pixel_values` | 改为边读边训的流式处理 |
| `torch.stack(pixel_values)` 报错 | 多图 patches 形状不统一 | 改用 `torch.cat` 拼接，通过 `image_grid_thw` 区分 |
| Flash Attention 不可用 | 服务器未安装编译依赖 | 回退为 `sdpa`，训练速度影响不大 |
| GPU 0 硬件故障 | 该卡持续报 `ERR!` | 运行时自动检测显存并切换到空闲卡 |
| 服务器频繁崩溃 | 共享集群资源竞争 + GPU 0 硬件问题 | tmux 后台运行，checkpoint 每 400 步自动保存 |

---

> 下一步：Step 9 模型训练（Qwen2.5-VL-7B + LoRA + QAlign 分类目标）


---

## 第九步：模型训练 — 完整训练记录

## 背景

复现 NTIRE 2025 图文对齐赛道冠军方案 **iMatch** 的训练阶段。基础模型 **Qwen2.5-VL-7B-Instruct**，数据集 **EvalMuse-40K**（~29K训练 / ~3.2K验证）。

完整项目仓库：[GitHub - DYEvaLab/EvalMuse](https://github.com/DYEvaLab/EvalMuse)

---

## 核心方法：QAlign

传统做法在 MLLM 输出层后接 MLP 回归头直接预测分数，iMatch 把回归问题**转化为分类问题**：

```
原始分数 [1,5] → 线性缩放 [1,15] → 映射字母 {a..o}
```

训练时输出字母做 CrossEntropy Loss，推理时对字母 logits 做 softmax 概率加权还原连续分数。**视觉特征直接用 Qwen2.5-VL 内置 ViT，不需要额外编码器。**

---

## 模型配置

| 参数 | 值 |
|------|-----|
| 基础模型 | `Qwen/Qwen2.5-VL-7B-Instruct` |
| 微调方式 | LoRA (r=64, alpha=128) |
| 可训练参数 | 1.9 亿 / 85 亿 (2.24%) |
| 计算精度 | bfloat16 |
| Attention | SDPA |

## 训练配置

| 参数 | 值 |
|------|-----|
| Epochs | 2 |
| 有效 Batch | 8 (1×8 梯度累积) |
| 学习率 | 2e-5, Cosine |
| 图片分辨率 | max 256×28×28 |
| GPU | RTX 5880 Ada (48GB) |

---

## 显存踩坑全记录

Qwen2.5-VL-7B 默认前向 ~45GB，加 LoRA 梯度容易 OOM。三个关键措施：

1. **限制图片分辨率** `max_pixels=256×28×28`，显存 45GB→25GB
2. **Gradient Checkpointing** 用计算换显存
3. **Expandable Segments** 防止碎片化

---

## 完整训练代码

### train_final.py

```python
"""终极版：限分辨率 + 碎片整理，稳定不 OOM"""
import os, torch, json, random
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
TRAIN_JSON = os.path.expanduser("~/EvalMuse-40K/train_split.json")
VAL_JSON = os.path.expanduser("~/EvalMuse-40K/val_split.json")
IMG_DIR = os.path.expanduser("~/EvalMuse-40K/images/dataset/images")
OUT = "outputs/run1"
EPOCHS=2; ACC=8; LR=2e-5
LETTERS = "abcdefghijklmno"
SYS = "You are an expert in image-text alignment evaluation."
MAX_PIXELS = 256 * 28 * 28

def letter(s): return LETTERS[max(0,min(14,int(round((s-1)/4*14))))]

# === 自动选空闲 GPU ===
import subprocess, re
r = subprocess.run(["nvidia-smi","--query-gpu=index,memory.free",
  "--format=csv,noheader"], capture_output=True, text=True)
gpu = sorted([(int(m), i) for i,m in re.findall(
  r"(\d+), (\d+) MiB", r.stdout)], reverse=True)[0][1]
dev = f"cuda:{gpu}"; print(f"GPU {gpu}")

os.makedirs(OUT, exist_ok=True)
with open(TRAIN_JSON) as f: train_raw = json.load(f)
with open(VAL_JSON) as f: val_raw = json.load(f)
print(f"Train:{len(train_raw)} Val:{len(val_raw)}")

# processor 限制分辨率
proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True,
  min_pixels=56*28*28, max_pixels=MAX_PIXELS)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
  MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
  trust_remote_code=True).to(dev)
model = get_peft_model(model, LoraConfig(
  r=64, lora_alpha=128, lora_dropout=0.05,
  target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"],
  bias="none", task_type=TaskType.CAUSAL_LM))
model.print_trainable_parameters()
model.gradient_checkpointing_enable()

opt = torch.optim.AdamW(model.parameters(), lr=LR)
model.train(); step=0

for ep in range(EPOCHS):
    print(f"\n=== Epoch {ep+1}/{EPOCHS} ===")
    random.shuffle(train_raw)
    running=0.0; opt.zero_grad()
    
    for i, s in enumerate(train_raw):
        ip = os.path.join(IMG_DIR, s["img_path"])
        if not os.path.exists(ip):
            ip = os.path.join(IMG_DIR, s["img_path"].lower())
        img = Image.open(ip).convert("RGB")
        score = sum(s["total_score"])/3
        lb = letter(score)
        
        msgs = [
            {"role":"system","content":[{"type":"text","text":SYS}]},
            {"role":"user","content":[
                {"type":"image","image":img},
                {"type":"text","text":f'Rate: "{s["prompt"]}"'},
            ]},
        ]
        text = proc.apply_chat_template(msgs, tokenize=False,
          add_generation_prompt=False)
        inp = proc(text=[text], images=[img], return_tensors="pt", padding=True)
        lbs = torch.full(inp["input_ids"].shape, -100, device=dev)
        lbs[0,-1] = proc.tokenizer.encode(lb, add_special_tokens=False)[0]
        batch = {k: v.to(dev) for k,v in inp.items()}
        batch["labels"] = lbs
        
        with torch.autocast(dev, dtype=torch.bfloat16):
            loss = model(**batch).loss / ACC
        loss.backward(); running+=loss.item()
        
        if (i+1)%ACC==0:
            opt.step(); opt.zero_grad(); step+=1
            if step%20==0:
                print(f"  step{step:5d}  loss={running/20:.4f}")
                running=0.0
            if step%400==0:
                model.save_pretrained(f"{OUT}/ckpt_{step}")
                print(f"  >>> saved ckpt_{step}")
        elif (i+1)%2==0:
            torch.cuda.empty_cache()

    ck = f"{OUT}/epoch_{ep+1}"
    model.save_pretrained(ck); proc.save_pretrained(ck)
    print(f"Saved {ck}")

print(f"\nDone! {OUT}")
```

### resume.py（断点恢复）

```python
"""从 checkpoint 恢复训练"""
import os, torch, json, random, glob
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
TRAIN_JSON = os.path.expanduser("~/EvalMuse-40K/train_split.json")
IMG_DIR = os.path.expanduser("~/EvalMuse-40K/images/dataset/images")
CKPT_DIR = "outputs/run1"; OUT = "outputs/run1"
EPOCHS=2; ACC=8; LR=2e-5
LETTERS = "abcdefghijklmno"
SYS = "You are an expert in image-text alignment evaluation."
MAX_PIXELS = 256 * 28 * 28

# 自动找最新 checkpoint
ckpts = sorted(glob.glob(f"{CKPT_DIR}/ckpt_*"),
               key=lambda x: int(x.split("_")[-1]))
CKPT = ckpts[-1]
START_STEP = int(CKPT.split("_")[-1]) // 8
SKIP = START_STEP * ACC
print(f"Resume from {CKPT} (step {START_STEP}, skip {SKIP} samples)")

def letter(s):
    return LETTERS[max(0,min(14,int(round((s-1)/4*14))))]

import subprocess, re
r = subprocess.run(["nvidia-smi","--query-gpu=index,memory.free",
  "--format=csv,noheader"], capture_output=True, text=True)
gpu = sorted([(int(m), i) for i,m in re.findall(
  r"(\d+), (\d+) MiB", r.stdout)], reverse=True)[0][1]
dev = f"cuda:{gpu}"; print(f"GPU {gpu}")

with open(TRAIN_JSON) as f: train_raw = json.load(f)
print(f"Train: {len(train_raw)}")

proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True,
  min_pixels=56*28*28, max_pixels=MAX_PIXELS)
base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
  MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
  trust_remote_code=True).to(dev)
model = PeftModel.from_pretrained(base, CKPT)
model.gradient_checkpointing_enable()

opt = torch.optim.AdamW(model.parameters(), lr=LR)
model.train(); step = START_STEP

for ep in range(EPOCHS):
    print(f"\n=== Epoch {ep+1}/{EPOCHS} ===")
    random.shuffle(train_raw)
    running=0.0; opt.zero_grad()
    for i, s in enumerate(train_raw):
        if ep == 0 and i < SKIP:
            if (i+1)%(ACC*20)==0: print(f"  skip {i+1}/{SKIP}")
            continue
        ip = os.path.join(IMG_DIR, s["img_path"])
        if not os.path.exists(ip):
            ip = os.path.join(IMG_DIR, s["img_path"].lower())
        img = Image.open(ip).convert("RGB")
        score = sum(s["total_score"])/3; lb = letter(score)
        msgs = [
            {"role":"system","content":[{"type":"text","text":SYS}]},
            {"role":"user","content":[
                {"type":"image","image":img},
                {"type":"text","text":f'Rate: "{s["prompt"]}"'},
            ]},
        ]
        text = proc.apply_chat_template(msgs, tokenize=False,
          add_generation_prompt=False)
        inp = proc(text=[text], images=[img], return_tensors="pt", padding=True)
        lbs = torch.full(inp["input_ids"].shape, -100, device=dev)
        lbs[0,-1] = proc.tokenizer.encode(lb, add_special_tokens=False)[0]
        batch = {k: v.to(dev) for k,v in inp.items()}
        batch["labels"] = lbs
        with torch.autocast(dev, dtype=torch.bfloat16):
            loss = model(**batch).loss / ACC
        loss.backward(); running+=loss.item()
        if (i+1)%ACC==0:
            opt.step(); opt.zero_grad(); step+=1
            if step%20==0:
                print(f"  step{step:5d}  loss={running/20:.4f}")
                running=0.0
            if step%400==0:
                model.save_pretrained(f"{OUT}/ckpt_{step}")
                print(f"  >>> saved ckpt_{step}")
        elif (i+1)%2==0:
            torch.cuda.empty_cache()
    ck = f"{OUT}/epoch_{ep+1}"
    model.save_pretrained(ck); proc.save_pretrained(ck)
    print(f"Saved {ck}")

print(f"\nDone! {OUT}")
```



> **说明**：esume.py 与 	rain_final.py 的核心逻辑（读图、构造输入、前向/反向、梯度累积、保存 checkpoint）完全一致，区别仅在于 esume.py 多了从最新 checkpoint 恢复并跳过已训练数据继续的逻辑。
### 启动命令

```bash
cd ~/imatch && source venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
tmux new -s train
python train_final.py
# Ctrl+B D 断开
# 断点恢复: python resume.py
```

---

## 训练曲线

```
step   20  loss=11.5881  ← 初始随机
step   40  loss=2.8887   ← 快速下降
step  200  loss=2.2911
step  400  loss=2.1070
step 1600  loss=2.0034
step 3200  loss=1.9868
step 5600  loss=2.0920
step 7360  loss=2.0587   ← 训练完成
```

> 注：loss 绝对值高是因为全词表分类（~15万token），模型只需在 15 个字母中区分即可。`run_all.py` 版本在第 40 步时就已经降到了 2.9，说明模型很快学会了 QAlign 打分。

---

## 训练耗时

| 阶段 | 耗时 |
|------|------|
| Epoch 1 | ~8h（3680 steps） |
| Epoch 2 | ~8h |
| **总计** | **~16h**（单卡 RTX 5880 Ada） |

## 断点恢复机制

- 每 **400 steps** 自动保存 checkpoint
- 每个 epoch 保存完整模型 + processor
- `resume.py` 自动找最新 checkpoint，跳过已训练数据继续
- `tmux` 保持进程不受 SSH 断连影响

## 训练产物

```
outputs/run1/
├── ckpt_400 ~ ckpt_7200/   # 18 个中间存档
├── epoch_1/                # epoch 1 完整模型
└── epoch_2/                # epoch 2 完整模型
```

## 下一步

用 `epoch_2` 在验证集做 QAlign 推理，计算 SRCC/PLCC：

| 方法 | SRCC | PLCC |
|------|------|------|
| FGA-BLIP2（基线） | 0.6491 | 0.6947 |
| iMatch（冠军） | 0.8249 | 0.8485 |
| **本复现** | **待测** | **待测** |


---

（后续步骤待补充）


