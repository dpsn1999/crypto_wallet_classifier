# hf_infer.py
import os
import threading
from typing import Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE_ID       = os.getenv("BASE_ID", "microsoft/Phi-3.5-mini-instruct")
ADAPTER_DIR   = os.getenv("ADAPTER_DIR", "lora_out")
USE_GPU       = os.getenv("USE_GPU", "1") == "1"
USE_4BIT      = os.getenv("USE_4BIT", "1") == "1"      # <- turn on 4-bit by default
OFFLOAD_DIR   = os.getenv("OFFLOAD_DIR", "./offload")
TEMP_STR      = os.getenv("GEN_TEMP", "0.0")
MAX_NEW       = int(os.getenv("MAX_NEW_TOKENS", "16")) # <- lower default
MAX_INPUT_TOK = int(os.getenv("MAX_INPUT_TOKENS", "640"))

try:
    TEMP = float(TEMP_STR)
except Exception:
    TEMP = 0.0

_device = "cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu"
_dtype  = torch.float16 if _device == "cuda" else torch.float32

_tokenizer = None
_model     = None
_ready     = False
_gen_lock  = threading.Lock()  # serialize generations

def _load_once():
    global _tokenizer, _model, _ready, _device, _dtype

    if _ready:
        return

    print(f"[hf_infer] BASE_ID={BASE_ID}")
    print(f"[hf_infer] ADAPTER_DIR={ADAPTER_DIR}")
    print(f"[hf_infer] device={_device} dtype={_dtype}")
    print(f"[hf_infer] USE_4BIT={USE_4BIT} OFFLOAD_DIR={OFFLOAD_DIR}")

    _tokenizer = AutoTokenizer.from_pretrained(BASE_ID, use_fast=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    quant_cfg = None
    device_map = None
    kwargs = dict(torch_dtype=_dtype)

    if _device == "cuda" and USE_4BIT:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        device_map = "auto"
        kwargs.update(dict(
            quantization_config=quant_cfg,
            device_map=device_map,
            offload_folder=OFFLOAD_DIR,
        ))
        os.makedirs(OFFLOAD_DIR, exist_ok=True)

    print("[hf_infer] loading base model …")
    base = AutoModelForCausalLM.from_pretrained(BASE_ID, **kwargs)

    print("[hf_infer] attaching LoRA adapter …")
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)

    # If device_map is not set (CPU or non-quantized GPU), move explicitly
    if device_map is None:
        try:
            model.to(_device)
        except Exception as e:
            print(f"[hf_infer] device move failed ({e}); falling back to CPU")
            _device = "cpu"
            _dtype = torch.float32
            model.to("cpu")

    model.eval()
    _model = model
    _ready = True
    print("[hf_infer] model ready ✅")


def _gen(prompt: str) -> str:
    """Generate with lock + OOM fallback."""
    global _device

    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOK)
    if _device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=MAX_NEW,
        eos_token_id=_tokenizer.eos_token_id,
        pad_token_id=_tokenizer.pad_token_id,
    )
    if TEMP > 0.0:
        gen_kwargs.update(do_sample=True, temperature=TEMP)
    else:
        gen_kwargs.update(do_sample=False)

    with _gen_lock:
        try:
            out = _model.generate(**inputs, **gen_kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and _device == "cuda":
                # clear and fallback to CPU for this request
                torch.cuda.empty_cache()
                print("[hf_infer] CUDA OOM → retrying on CPU for this call")
                cpu_inputs = {k: v.to("cpu") for k, v in inputs.items()}
                out = _model.to("cpu").generate(**cpu_inputs, **gen_kwargs)
                _model.to("cuda")  # keep base state
            else:
                raise
    text = _tokenizer.decode(out[0], skip_special_tokens=True)
    return text


@torch.inference_mode()
def classify_with_lora(tx: Dict) -> str:
    _load_once()

    prompt = (
        'You are a strict classifier. Respond ONLY with JSON of the form '
        '{"category":"<one of: Token Swap, NFT Purchase, DeFi Interaction, Simple Transfer, Contract Deployment, Unknown>"}\n\n'
        f"From: {tx.get('from')}\n"
        f"To: {tx.get('to')}\n"
        f"Value (wei): {tx.get('value')}\n"
        f"Input: {tx.get('input')}\n\n"
        "Return JSON now:"
    )

    text = _gen(prompt)
    raw = text.split("Return JSON now:")[-1].strip()

    import json, re
    try:
        obj = json.loads(raw)
        label = str(obj.get("category", "")).strip()
    except Exception:
        m = re.search(r'"category"\s*:\s*"([^"]+)"', raw, flags=re.I)
        label = m.group(1).strip() if m else raw.strip()

    norm = label.lower()
    if   "swap" in norm:                                      return "Token Swap"
    elif "nft" in norm or "purchase" in norm:                 return "NFT Purchase"
    elif "defi" in norm or "liquidity" in norm or "stake" in norm: return "DeFi Interaction"
    elif "deploy" in norm or "creation" in norm:              return "Contract Deployment"
    elif "transfer" in norm or "send" in norm:                return "Simple Transfer"
    elif label in {
        "Token Swap","NFT Purchase","DeFi Interaction",
        "Simple Transfer","Contract Deployment","Unknown"
    }:
        return label
    return "Unknown"


def warmup_once():
    _load_once()

