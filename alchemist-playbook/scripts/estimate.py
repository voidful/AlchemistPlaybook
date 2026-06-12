#!/usr/bin/env python3
"""Training-run arithmetic for the alchemist-playbook skill.

Subcommands:
  flops       FLOPs (6ND), GPU-hours, wall-clock, optional cost
  chinchilla  Chinchilla-optimal tokens<->params check (20 tok/param)
  batch       tokens/step from (gpus, micro, seq, accum); warmup sizing
  lr          peak-LR ladder lookup by size and stage, with sources

No third-party dependencies. All outputs label their assumptions.
"""
import argparse
import sys

GPU_PEAK_TFLOPS = {  # dense BF16/FP16 peak, no sparsity
    "h100": 989.0,
    "h200": 989.0,
    "a100": 312.0,
    "a6000": 155.0,
    "v100": 125.0,  # fp16
    "4090": 165.0,
    "mi250x": 383.0,
    "b200": 2250.0,
}


def fmt(x: float) -> str:
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(x) >= div:
            return f"{x / div:.3g}{unit}"
    return f"{x:.3g}"


def cmd_flops(a):
    flops = 6.0 * a.params * a.tokens
    print(f"Training FLOPs (6ND): {flops:.3e}")
    peak = GPU_PEAK_TFLOPS[a.gpu] * 1e12
    eff = peak * a.mfu
    gpu_seconds = flops / eff
    gpu_hours = gpu_seconds / 3600
    print(f"Assumptions: {a.gpu.upper()} peak {peak/1e12:.0f} TFLOPs dense, MFU {a.mfu:.0%}")
    print(f"GPU-hours: {gpu_hours:,.0f}  (= {gpu_hours/24:,.1f} GPU-days)")
    if a.gpus:
        days = gpu_hours / a.gpus / 24
        print(f"Wall-clock on {a.gpus} GPUs: {days:,.2f} days")
    if a.cost_per_hour:
        print(f"Cost @ ${a.cost_per_hour}/GPU-hr [ASSUMED PRICE]: ${gpu_hours * a.cost_per_hour:,.0f}")


def cmd_chinchilla(a):
    if a.params and not a.tokens:
        opt = 20.0 * a.params
        print(f"Chinchilla-optimal tokens for {fmt(a.params)} params: ~{fmt(opt)} (20 tok/param)")
        print("Reminder: deployment-optimal is usually FAR above this "
              "(LLaMA-7B 140x, SmolLM2-1.7B ~6500x). Below 20x = undertrained.")
    elif a.tokens and not a.params:
        opt = a.tokens / 20.0
        print(f"Chinchilla-optimal params for {fmt(a.tokens)} tokens: ~{fmt(opt)}")
    elif a.params and a.tokens:
        ratio = a.tokens / a.params
        print(f"tokens/param = {ratio:,.0f}  (Chinchilla-optimal ~20)")
        if ratio < 15:
            print("UNDERTRAINED for compute-optimality; add tokens or shrink model.")
        elif ratio <= 30:
            print("Near compute-optimal. Fine for research; small deployed models usually overtrain.")
        else:
            print(f"Overtraining regime ({ratio/20:.0f}x Chinchilla) - standard for deployed small models.")
    else:
        sys.exit("Provide --params and/or --tokens")


def cmd_batch(a):
    tps = a.gpus * a.micro * a.seq * a.accum
    print(f"tokens/step = {a.gpus} gpus x {a.micro} micro x {a.seq} seq x {a.accum} accum = {tps:,} ({fmt(tps)})")
    print(f"sequences/step = {a.gpus * a.micro * a.accum:,}")
    if a.target_tokens:
        steps = a.target_tokens / tps
        print(f"Steps for {fmt(a.target_tokens)} tokens: {steps:,.0f}")
        w = min(max(round(steps * 0.01), 1000), 8000)
        print(f"Warmup suggestion: ~{w} steps (~1% of steps, clamped to [1000, 8000]; "
              f"= {fmt(w * tps)} tokens). Anchors: LLaMA 2000, OLMo-2 ~2000 (8.4B tok), Llama-3-405B 8000.")
    bands = [(2e9, "<=2B: ~2M tokens/step (SmolLM2)"),
             (15e9, "7-13B: ~4M (LLaMA, OLMo-2-7B)"),
             (8e10, "30-70B: 4-8M (OLMo-2-13B/32B 8.4M)"),
             (1e12, ">=100B: 8-16M with ramp (Llama-3-405B, DeepSeek-V3)")]
    if a.params:
        for lim, msg in bands:
            if a.params <= lim:
                print(f"Published batch band for {fmt(a.params)}: {msg}")
                break


LADDER = {
    "pretrain": [
        (2e9, "4e-4 - 1e-3", "SmolLM2-1.7B 5e-4; OLMo-1B 4e-4; Pythia small ~1e-3"),
        (15e9, "3e-4", "LLaMA-7B/13B; OLMo-1/2 7B & 13B (config)"),
        (8e10, "1.5e-4 - 6e-4", "LLaMA-33/65B & Llama-2-70B 1.5e-4; OLMo-2-32B 6e-4 (QK-norm stack)"),
        (1e15, "8e-5 - 2.2e-4", "Llama-3-405B 8e-5; DeepSeek-V3 MoE 2.2e-4"),
    ],
    "sft": [
        (15e9, "5e-6 - 2e-5", "Tulu-3-8B 5e-6; Zephyr & Llama-2-chat 2e-5"),
        (1e15, "2e-6", "Tulu-3 70B/405B"),
    ],
    "dpo": [
        (15e9, "5e-7 (beta 0.01 std / beta 5 length-norm)", "Zephyr; Tulu-3-8B"),
        (1e15, "2e-7", "Tulu-3 70B/405B"),
    ],
    "rl": [
        (1e15, "3e-7 - 1e-6", "Tulu-3 PPO/RLVR 3e-7; DeepSeekMath GRPO 1e-6"),
    ],
    "lora": [
        (15e9, "1e-4 - 2e-4 (start 2e-4)", "QLoRA 7-13B"),
        (1e15, "1e-4", "QLoRA 33-65B"),
    ],
}


def cmd_lr(a):
    stage = a.stage.lower()
    if stage not in LADDER:
        sys.exit(f"stage must be one of {list(LADDER)}")
    for lim, lr, src in LADDER[stage]:
        if a.params <= lim:
            print(f"{stage.upper()} peak LR for {fmt(a.params)} params: {lr}")
            print(f"Source runs: {src}")
            break
    if stage == "pretrain":
        print("Pair with: AdamW (0.9, 0.95), eps 1e-8, wd 0.1, clip 1.0, "
              "warmup ~2000 steps, cosine->10% or WSD.")
    if stage == "dpo":
        print("NEVER copy beta across loss types: 0.01 (standard) vs 5 (length-normalized).")
    if stage == "lora":
        print("LoRA LR is ~10-20x full-FT SFT LR. Target ALL linear modules.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("flops", help="FLOPs / GPU-hours / cost")
    f.add_argument("--params", type=float, required=True)
    f.add_argument("--tokens", type=float, required=True)
    f.add_argument("--gpus", type=int, default=0)
    f.add_argument("--gpu", choices=sorted(GPU_PEAK_TFLOPS), default="h100")
    f.add_argument("--mfu", type=float, default=0.40, help="Llama-3 reached 0.38-0.43")
    f.add_argument("--cost-per-hour", type=float, default=0.0)
    f.set_defaults(func=cmd_flops)

    c = sub.add_parser("chinchilla", help="compute-optimal check")
    c.add_argument("--params", type=float)
    c.add_argument("--tokens", type=float)
    c.set_defaults(func=cmd_chinchilla)

    b = sub.add_parser("batch", help="token-batch arithmetic")
    b.add_argument("--gpus", type=int, required=True)
    b.add_argument("--micro", type=int, required=True, help="sequences per GPU per fwd/bwd")
    b.add_argument("--seq", type=int, required=True)
    b.add_argument("--accum", type=int, default=1)
    b.add_argument("--target-tokens", type=float, default=0)
    b.add_argument("--params", type=float, default=0)
    b.set_defaults(func=cmd_batch)

    l = sub.add_parser("lr", help="peak-LR ladder lookup")
    l.add_argument("--params", type=float, required=True)
    l.add_argument("--stage", required=True, help="pretrain|sft|dpo|rl|lora")
    l.set_defaults(func=cmd_lr)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
