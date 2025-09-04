import os, json, pathlib, random, inspect
from dotenv import load_dotenv
from huggingface_hub import login
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, TrainingArguments,
    DataCollatorForLanguageModeling, Trainer
)
from peft import (
    LoraConfig, get_peft_model,
    prepare_model_for_kbit_training
)

# ===================== 0) Auth & config =====================
print("[0/7] loading env & auth")
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not found in .env — add HF_TOKEN=hf_*** to your .env")
login(HF_TOKEN)

HF_ID = "microsoft/Phi-3.5-mini-instruct"   # open, small, Windows-friendly
OUTPUT_DIR = "lora_out"
DATA_TRAIN = "data/train.jsonl"
DATA_DEV   = "data/dev.jsonl"
SRC_DATA   = pathlib.Path("data.jsonl")

# modest max length to fit 8GB VRAM
MAX_LEN = 512
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ===================== 1) Prepare dataset =====================
if (not pathlib.Path(DATA_TRAIN).exists()) or (not pathlib.Path(DATA_DEV).exists()):
    print("[1/7] splitting dataset from data.jsonl")
    if not SRC_DATA.is_file():
        raise FileNotFoundError("data.jsonl not found in project root")
    rows = []
    for line in SRC_DATA.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            p = (obj.get("prompt") or "").strip()
            y = (obj.get("completion") or "").strip()
            if p and y:
                rows.append({"prompt": p, "completion": y})
        except Exception:
            pass

    from collections import defaultdict
    dd = defaultdict(list)
    for r in rows:
        dd[r["completion"].strip()].append(r)

    random.seed(42)
    train, dev = [], []
    for _, items in dd.items():
        random.shuffle(items)
        k = max(1, int(0.9 * len(items)))
        train += items[:k]
        dev   += items[k:]

    pathlib.Path("data").mkdir(exist_ok=True)
    with open(DATA_TRAIN, "w", encoding="utf-8") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with open(DATA_DEV, "w", encoding="utf-8") as f:
        for r in dev:
            f.write(json.dumps(r) + "\n")
    print(f"[1/7] split complete → train={len(train)} dev={len(dev)}")
else:
    print("[1/7] dataset split already present")

# ===================== 2) Tokenizer =====================
print("[2/7] loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(HF_ID, use_fast=True, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ===================== 3) Base model (prefer QLoRA 4-bit) =====================
print("[3/7] loading base model (QLoRA if available)")
has_cuda = torch.cuda.is_available()
use_bnb = False
model = None

if has_cuda:
    try:
        from transformers import BitsAndBytesConfig
        compute_dtype = torch.bfloat16  # compute dtype on GPU
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            HF_ID,
            token=HF_TOKEN,
            quantization_config=bnb_cfg,
            device_map="auto",
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        use_bnb = True
        print("[3/7] loaded with bitsandbytes in 4-bit")
    except Exception as e:
        print(f"[3/7] WARNING: 4-bit load failed ({e}); falling back to non-4bit")

if model is None:
    # CPU or non-4bit GPU fallback
    dtype = torch.float16 if has_cuda else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        HF_ID,
        token=HF_TOKEN,
        torch_dtype=dtype,
        device_map=None,
    )
    device = "cuda" if has_cuda else "cpu"
    model.to(device)
    try:
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
    except Exception:
        pass
    print(f"[3/7] loaded on {device} without 4-bit")

# ===================== 4) Prepare for k-bit training + LoRA =====================
print("[4/7] preparing model for k-bit training & attaching LoRA")
# IMPORTANT: ensure grads flow to LoRA in k-bit
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj"
    ],
)
model = get_peft_model(model, lora_cfg)

# ===================== 5) Build training texts =====================
print("[5/7] loading & formatting dataset")
def format_example(ex):
    prompt = (
        'Respond with exactly one JSON object and nothing else. '
        'Use this exact key: "category". The value must be one of: '
        'Token Swap, NFT Purchase, DeFi Interaction, Simple Transfer, Contract Deployment, Unknown.\n\n'
        f'{ex["prompt"].strip()}\n'
        'Answer: '
    )
    label = ex["completion"].strip()
    answer = json.dumps({"category": label})
    return {"text": prompt + answer}

ds = load_dataset("json", data_files={"train": DATA_TRAIN, "validation": DATA_DEV})
ds = ds.map(format_example, remove_columns=ds["train"].column_names)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN, padding=False)

tok = ds.map(tokenize, batched=True, remove_columns=["text"])
print(f"[5/7] tokenized: train={len(tok['train'])} dev={len(tok['validation'])}")

# ===================== 6) TrainingArguments =====================
print("[6/7] creating TrainingArguments")
def make_training_args(output_dir="lora_out"):
    common = dict(
        output_dir=output_dir,
        per_device_train_batch_size=1,        # tiny to fit 8GB
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,       # effective batch = 16
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=20,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        no_cuda=not has_cuda,                 # we already placed the model
    )
    # If NOT using bnb 4-bit, enable mixed precision where possible
    if has_cuda and not use_bnb:
        # prefer bf16 on modern GPUs; else fp16
        try:
            common["bf16"] = True
        except Exception:
            common["fp16"] = True

    # Add eval hooks only if supported (saves VRAM to keep it off initially)
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in params:
        common["evaluation_strategy"] = "no"

    return TrainingArguments(**common)

args = make_training_args(OUTPUT_DIR)
collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# ===================== 7) Train =====================
print("[7/7] building Trainer & starting training…")
if has_cuda:
    torch.cuda.empty_cache()

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tok["train"],
    eval_dataset=tok["validation"],
    data_collator=collator,
)
trainer.train()

print("[done] saving adapter & tokenizer")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Saved LoRA adapter to", OUTPUT_DIR)


