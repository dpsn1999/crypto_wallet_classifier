import argparse, json, re, pathlib, sys
from collections import defaultdict, Counter

# Evaluate the *same* entrypoint as the API uses
import llm_classifier as L

CATS = [
    "Token Swap","NFT Purchase","DeFi Interaction",
    "Simple Transfer","Contract Deployment","Unknown"
]

def parse_tx_from_prompt(prompt: str):
    """
    Your jsonl has fields: prompt, completion
    Prompts look like:
      From: 0x...
      To: 0x...
      Value (wei): 123
      Input: 0x....
    """
    def grab(pat, default=""):
        m = re.search(pat, prompt, flags=re.I)
        return m.group(1).strip() if m else default

    from_addr = grab(r"From:\s*([^\s]+)")
    to_addr   = grab(r"To:\s*([^\s]+)")
    value     = grab(r"Value\s*\(wei\):\s*([0-9]+)", "0")
    data      = grab(r"Input:\s*(0x[0-9A-Fa-f]*)", "0x")
    return {"from": from_addr, "to": to_addr, "value": value, "input": data}

def norm_label(s: str):
    s = (s or "").strip()
    lo = s.lower()
    for c in CATS:
        if lo == c.lower():
            return c
    # light aliasing so minor drifts don’t tank metrics
    if "swap" in lo: return "Token Swap"
    if "nft" in lo or "purchase" in lo: return "NFT Purchase"
    if any(k in lo for k in ["defi","liquidity","stake","lend","borrow"]): return "DeFi Interaction"
    if "transfer" in lo or "send" in lo: return "Simple Transfer"
    if "deploy" in lo or "creation" in lo: return "Contract Deployment"
    return "Unknown"

def load_jsonl(path: pathlib.Path):
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line: 
            continue
        try:
            yield json.loads(line)
        except Exception as e:
            print(f"[warn] bad json on line {ln}: {e}", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="data/dev.jsonl", help="Path to eval jsonl (prompt/completion)")
    ap.add_argument("--save_csv", default="", help="Optional path to write predictions CSV")
    ap.add_argument("--no_heuristics", action="store_true", help="Disable quick rules; use LoRA only")
    args = ap.parse_args()

    p = pathlib.Path(args.path)
    if not p.is_file():
        print(f"[error] file not found: {p}")
        sys.exit(1)

    rows = list(load_jsonl(p))
    if not rows:
        print("[error] no rows found")
        sys.exit(1)

    total = 0
    correct = 0
    by_actual = Counter()
    by_pred   = Counter()
    conf = defaultdict(lambda: Counter())
    skipped = 0

    # optional CSV
    csvf = None
    if args.save_csv:
        csvf = open(args.save_csv, "w", encoding="utf-8", newline="")
        csvf.write("actual,pred,from,to,value,input\n")

    for r in rows:
        prompt = (r.get("prompt") or "").strip()
        gold   = norm_label(r.get("completion") or "")
        if not prompt or not gold:
            skipped += 1
            continue

        tx = parse_tx_from_prompt(prompt)
        # call classifier (same as API path)
        pred = L.classify_transaction(
            tx,
            use_heuristics=(not args.no_heuristics)
        )
        pred = norm_label(pred)

        total += 1
        by_actual[gold] += 1
        by_pred[pred]   += 1
        conf[gold][pred] += 1
        if pred == gold:
            correct += 1

        if csvf:
            csvf.write(f"{gold},{pred},{tx['from']},{tx['to']},{tx['value']},{tx['input']}\n")

    if csvf:
        csvf.close()

    if total == 0:
        print("[error] nothing to evaluate (all skipped)")
        sys.exit(1)

    acc = correct / total
    print(f"\nEvaluated {total} examples (skipped {skipped})")
    print(f"Overall accuracy: {acc:.3f}\n")

    # per-class metrics
    print("Per-class metrics:")
    print(f"{'class':22} {'support':>8} {'precision':>10} {'recall':>8} {'f1':>8}")
    for c in CATS:
        tp = conf[c][c]
        support = by_actual[c]
        pred_as_c = by_pred[c]
        prec = (tp / pred_as_c) if pred_as_c else 0.0
        rec  = (tp / support) if support else 0.0
        f1   = (2*prec*rec/(prec+rec)) if (prec+rec)>0 else 0.0
        print(f"{c:22} {support:8d} {prec:10.3f} {rec:8.3f} {f1:8.3f}")

    # simple confusion table
    print("\nConfusion matrix (rows=actual, cols=pred):")
    header = " " * 22 + " | " + " | ".join(f"{c[:14]:14}" for c in CATS)
    print(header)
    print("-" * len(header))
    for a in CATS:
        row = f"{a:22} | " + " | ".join(f"{conf[a][b]:14d}" for b in CATS)
        print(row)

if __name__ == "__main__":
    main()
