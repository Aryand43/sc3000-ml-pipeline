import argparse
import inspect
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------- paths/config

ROOT = Path(__file__).resolve().parent.parent
DPO_DIR = ROOT / "dpo"
TERMINATOR = "\n\n\n\n"  # supplied preprocessing appends four newlines
PAD_ID = 0  # supplied padding id; also the id of '\n', hence the explicit masks


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="DPO fine-tuning (Task 2)")
    # Defaults mirror the supplied Step 2 configuration.
    p.add_argument("--pairs", default=str(DPO_DIR / "pos_neg_pairs.json"))
    p.add_argument("--dev-pairs", default=str(DPO_DIR / "task1_dev_pairs.json"))
    p.add_argument("--checkpoint", default=str(ROOT / "sft" / "gpt.pt"))
    p.add_argument("--out", default=str(DPO_DIR / "dpo.pt"))
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup-frac", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    # beta 2.0 on mean-normalised log-probs == the supplied hint's "/ beta" at
    # beta=0.5; see the Task 2 notes in the notebook.
    p.add_argument("--beta", type=float, default=2.0)
    p.add_argument("--lambda-sft", type=float, default=0.1)
    p.add_argument("--reference", choices=["sft", "none"], default="sft")
    p.add_argument("--loss-scope", choices=["response", "full"], default="response")
    p.add_argument("--length-norm", choices=["mean", "sum"], default="mean")
    p.add_argument("--seed", type=int, default=3000)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None, help="use the first N training pairs")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-best", action="store_true", default=True)
    p.add_argument("--no-save-best", dest="save_best", action="store_false")
    # Objective: "dpo" is the Task 2 preference objective; "sft" is plain
    # supervised training on the positives only, for the SFT-then-DPO pipeline.
    p.add_argument("--objective", choices=["dpo", "sft"], default="dpo")
    # Generated-answer evaluation (Task 3 scoring, reused verbatim).
    p.add_argument("--gen-per-category", type=int, default=20)
    p.add_argument("--gen-seed", type=int, default=3003)
    p.add_argument("--gen-batch-size", type=int, default=16)
    p.add_argument("--gen-max-new-tokens", type=int, default=64)
    p.add_argument("--no-gen-eval", dest="gen_eval", action="store_false", default=True,
                   help="skip generated-answer evaluation (preference metrics only)")
    p.add_argument("--overwrite", action="store_true",
                   help="allow --out to replace an existing checkpoint")
    p.add_argument("--smoke", action="store_true",
                   help="random tiny model, 512 pairs, 1 epoch: checks the pipeline only")
    args = p.parse_args(argv)
    if args.smoke:
        args.epochs, args.limit, args.batch_size = 1, 512, 32
        args.log_every = 5
        args.gen_per_category = 2
        # The smoke model's context is max_length, so keep prompt + budget inside it.
        args.gen_max_new_tokens = min(args.gen_max_new_tokens, 24)
        if args.out == str(DPO_DIR / "dpo.pt"):  # keep the real checkpoint intact
            args.out = str(DPO_DIR / "smoke_dpo.pt")
    return args


def pick_device(requested=None):
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ------------------------------------------------------------------- tokeniser

def load_tokeniser(root=ROOT):
    with (root / "sft" / "meta.pkl").open("rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]
    return stoi, itos, meta["vocab_size"]


# ---------------------------------------------------------------- Step 5: data

def load_pairs(path, limit=None):
    with open(path) as f:
        pairs = json.load(f)
    assert isinstance(pairs, list) and pairs, f"no records in {path}"
    for rec in pairs:
        assert set(rec) == {"positive", "negative"}, f"unexpected schema: {sorted(rec)}"
    return pairs[:limit] if limit else pairs


def prompt_length(pair):
    question = pair["positive"].split("?", 1)[0] + "?"
    for key in ("positive", "negative"):
        assert pair[key].startswith(question + " "), f"question is not a shared prefix: {pair}"
    return len(question) + 1


def encode_pair(pair, stoi, max_length, loss_scope):
    start = prompt_length(pair) if loss_scope == "response" else 0
    out = []
    for key in ("negative", "positive"):
        ids = [stoi[c] for c in pair[key] + TERMINATOR]
        # No-truncation invariant from Task 1 (longest record is 62 tokens).
        assert len(ids) <= max_length, f"{len(ids)} > max_length={max_length}: {pair[key]!r}"
        mask = [start <= i < len(ids) for i in range(max_length)]
        out.append((ids + [PAD_ID] * (max_length - len(ids)), mask))
    return out


def get_batches(pairs, batch_size, stoi, max_length, loss_scope, device, rng=None):
    order = list(range(len(pairs)))
    if rng is not None:
        rng.shuffle(order)
    for i in range(0, len(order), batch_size):
        chunk = [pairs[j] for j in order[i:i + batch_size]]
        encoded = [encode_pair(p, stoi, max_length, loss_scope) for p in chunk]
        tensors = []
        for slot in (0, 1):  # 0 = negative, 1 = positive
            tensors.append(torch.tensor([e[slot][0] for e in encoded], dtype=torch.long, device=device))
            tensors.append(torch.tensor([e[slot][1] for e in encoded], dtype=torch.bool, device=device))
        yield tuple(tensors)


# ----------------------------------------------------------------- model/logps

def load_gpt(GPT, GPTConfig, checkpoint_path, device, smoke=False, vocab_size=74, max_length=64):
    if smoke:
        config = GPTConfig(n_layer=2, n_head=2, n_embd=64, block_size=max_length,
                           bias=False, vocab_size=vocab_size, dropout=0.0)
        model, model_args = GPT(config), dict(
            n_layer=2, n_head=2, n_embd=64, block_size=max_length,
            bias=False, vocab_size=vocab_size, dropout=0.0)
        return model.to(device), model_args
    with open(checkpoint_path, "rb") as f:
        if f.read(80).startswith(b"version https://git-lfs.github.com/spec/"):
            raise RuntimeError(f"{checkpoint_path} is an LFS pointer; run `git lfs pull` "
                               "or pass --checkpoint with the real weights")
    # Trusted local checkpoint; torch >= 2.6 defaults to weights_only=True.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_args = ckpt["model_args"]
    model = GPT(GPTConfig(**model_args))
    # The SFT checkpoint stores "model"; our own runs store "model_state_dict",
    # so accept either and resume from a DPO checkpoint as well.
    state = ckpt.get("model_state_dict", ckpt.get("model"))
    if state is None:
        raise KeyError(f"{checkpoint_path}: no 'model' or 'model_state_dict' key "
                       f"(found {sorted(ckpt)})")
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in state.items()})
    return model.to(device), model_args


def forward_logits(model, inputs):
    if "full_seq" in inspect.signature(type(model).forward).parameters:
        logits, _ = model(inputs, full_seq=True)
    else:
        logits, _ = model(inputs, targets=inputs)
    return logits


def sequence_logprob(model, ids, mask, length_norm="mean"):
    inputs, targets = ids[:, :-1], ids[:, 1:]
    logits = forward_logits(model, inputs)
    B, T, V = logits.shape
    token_logprob = -F.cross_entropy(
        logits.reshape(-1, V).float(), targets.reshape(-1), reduction="none"
    ).view(B, T)
    scored = mask[:, 1:].to(token_logprob.dtype)
    total = (token_logprob * scored).sum(dim=1)
    if length_norm == "sum":
        return total
    return total / scored.sum(dim=1).clamp(min=1.0)


def dpo_loss(pol_pos, pol_neg, ref_pos, ref_neg, beta, lambda_sft):
    pos_ratio = pol_pos - ref_pos
    neg_ratio = pol_neg - ref_neg
    logits = beta * (pos_ratio - neg_ratio)
    preference = -F.logsigmoid(logits).mean()
    sft = -pol_pos.mean()
    stats = dict(
        dpo_loss=preference.item(),
        sft_loss=sft.item(),
        accuracy=(logits > 0).float().mean().item(),
        margin=(pos_ratio - neg_ratio).mean().item(),
        pos_reward=pos_ratio.mean().item(),
        neg_reward=neg_ratio.mean().item(),
    )
    return preference + lambda_sft * sft, stats


# --------------------------------------------------------- Step 6: optim/sched

def build_optimizer(model, lr, weight_decay):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)
    groups = [
        dict(params=decay, weight_decay=weight_decay),
        dict(params=no_decay, weight_decay=0.0),
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


def build_scheduler(optimizer, total_steps, warmup_frac, min_lr_ratio):
    warmup = max(1, int(total_steps * warmup_frac))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


# ------------------------------------------------------------ Step 7: training

@torch.no_grad()
def evaluate(policy, reference, pairs, args, stoi, device):
    was_training = policy.training
    policy.eval()
    totals, batches = {}, 0
    for neg_ids, neg_mask, pos_ids, pos_mask in get_batches(
        pairs, args.batch_size, stoi, args.max_length, args.loss_scope, device
    ):
        pol_pos = sequence_logprob(policy, pos_ids, pos_mask, args.length_norm)
        pol_neg = sequence_logprob(policy, neg_ids, neg_mask, args.length_norm)
        ref_pos, ref_neg = reference_logprobs(reference, pos_ids, pos_mask, neg_ids, neg_mask, args)
        _, stats = dpo_loss(pol_pos, pol_neg, ref_pos, ref_neg, args.beta, args.lambda_sft)
        for k, v in stats.items():
            totals[k] = totals.get(k, 0.0) + v
        batches += 1
    if was_training:
        policy.train()
    return {k: v / max(1, batches) for k, v in totals.items()}


def reference_logprobs(reference, pos_ids, pos_mask, neg_ids, neg_mask, args):
    if reference is None:
        zero = torch.zeros(pos_ids.size(0), device=pos_ids.device)
        return zero, zero
    with torch.no_grad():
        return (sequence_logprob(reference, pos_ids, pos_mask, args.length_norm),
                sequence_logprob(reference, neg_ids, neg_mask, args.length_norm))


def run_generation_eval(policy, questions, stoi, itos, args, label):
    """Task 3 scoring, imported unchanged from math_evaluation."""
    from math_evaluation import evaluate_generated_math
    report = evaluate_generated_math(policy, questions, stoi, itos, device=args.device,
                                     batch_size=args.gen_batch_size,
                                     max_new_tokens=args.gen_max_new_tokens)
    metrics = report["summary"]["overall"]
    wrong = [r for r in report["results"] if not r["answer_correct"]]
    # Separate a wrong number from a response we could not parse or that ran out
    # of budget: they call for different fixes.
    modes = dict(wrong_answer=sum(r["parsed_answer"] is not None for r in wrong),
                 unparsed=sum(r["parsed_answer"] is None for r in wrong),
                 truncated=sum(r["truncated"] for r in wrong))
    print(f"  {label}: generated answer accuracy "
          f"{metrics['answer_correct']}/{metrics['count']} ({metrics['answer_accuracy']:.1%}), "
          f"fully verified {metrics['fully_verified_accuracy']:.1%} | failures: "
          f"{modes['wrong_answer']} wrong math, {modes['unparsed']} unparsed, "
          f"{modes['truncated']} truncated")
    cases = [{k: r[k] for k in ("question", "expected_answer", "category", "response",
                                "parsed_answer", "stop_reason", "truncated",
                                "explanation_status")} for r in wrong]
    return report, metrics, dict(stage=label, failure_modes=modes, cases=cases)


def main(argv=None):
    args = parse_args(argv)
    args.device = pick_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    sys.path.insert(0, str(ROOT))
    from model import GPT, GPTConfig  # supplied, unchanged

    stoi, itos, vocab_size = load_tokeniser()
    pairs = load_pairs(args.pairs, args.limit)
    dev_pairs = load_pairs(args.dev_pairs) if Path(args.dev_pairs).exists() else []
    canonical = lambda rec: "".join(rec["positive"].split("?", 1)[0].split())
    train_questions = {canonical(p) for p in pairs}
    overlap = sum(canonical(p) in train_questions for p in dev_pairs)
    assert overlap == 0, f"{overlap} dev questions leaked into training"

    policy, model_args = load_gpt(GPT, GPTConfig, args.checkpoint, args.device,
                                  args.smoke, vocab_size, args.max_length)
    policy.train()
    reference = None
    if args.objective == "dpo" and args.reference == "sft":
        reference, _ = load_gpt(GPT, GPTConfig, args.checkpoint, args.device,
                                args.smoke, vocab_size, args.max_length)
        if args.smoke:  # the ablation needs identical weights, not a second init
            reference.load_state_dict(policy.state_dict())
        reference.eval()
        reference.requires_grad_(False)

    steps_per_epoch = math.ceil(math.ceil(len(pairs) / args.batch_size) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    optimizer = build_optimizer(policy, args.lr, args.weight_decay)
    scheduler = build_scheduler(optimizer, total_steps, args.warmup_frac, args.min_lr_ratio)
    use_bf16 = args.device == "cuda" and torch.cuda.is_bf16_supported()

    print(f"device={args.device} pairs={len(pairs)} dev={len(dev_pairs)} "
          f"steps/epoch={steps_per_epoch} total_steps={total_steps} bf16={use_bf16}")
    print(f"beta={args.beta} lambda_sft={args.lambda_sft} reference={args.reference} "
          f"loss_scope={args.loss_scope} length_norm={args.length_norm}")

    history = {"config": vars(args), "model_args": model_args, "train": [], "dev": [],
               "generated_dev": []}
    best = (-float("inf"), -float("inf"))
    best_generated = -1.0
    failures_log = []
    out_path = Path(args.out)
    if out_path.exists() and not (args.overwrite or args.smoke):
        raise SystemExit(f"{out_path} already exists. Pass --overwrite to replace it, "
                         "or choose another --out so existing checkpoints are preserved.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    generated_best_path = out_path.with_name(out_path.stem + "_generated_best.pt")

    # Select the generated-evaluation questions ONCE, from development pairs
    # only, and reuse the identical list and decoding for every evaluation.
    gen_questions = []
    if args.gen_eval and dev_pairs:
        from math_evaluation import select_development_questions
        gen_questions = select_development_questions(dev_pairs, args.gen_per_category,
                                                     args.gen_seed)
        print(f"generated-answer evaluation: {len(gen_questions)} development questions "
              f"({args.gen_per_category}/category, seed {args.gen_seed})")
        _, baseline_metrics, baseline_failures = run_generation_eval(
            policy, gen_questions, stoi, itos, args, "before training")
        history["generated_dev"].append(dict(epoch=0, **baseline_metrics))
        failures_log.append(dict(epoch=0, **baseline_failures))

    for epoch in range(args.epochs):
        rng = random.Random(args.seed + epoch)
        batcher = get_batches(pairs, args.batch_size, stoi, args.max_length,
                              args.loss_scope, args.device, rng)
        try:
            from tqdm import tqdm
            batcher = tqdm(batcher, total=math.ceil(len(pairs) / args.batch_size),
                           desc=f"epoch {epoch + 1}/{args.epochs}")
        except ImportError:
            pass
        # Two accumulators: one for the epoch average, one for the logged window.
        # Counts are per key because grad_norm only exists on step boundaries.
        running, running_n, window, window_n = {}, {}, {}, {}
        seen, step, t0 = 0, 0, time.time()
        optimizer.zero_grad(set_to_none=True)
        for micro, (neg_ids, neg_mask, pos_ids, pos_mask) in enumerate(batcher):
            autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 \
                else torch.autocast("cpu", enabled=False)
            with autocast:
                pol_pos = sequence_logprob(policy, pos_ids, pos_mask, args.length_norm)
                if args.objective == "dpo":
                    pol_neg = sequence_logprob(policy, neg_ids, neg_mask, args.length_norm)
            if args.objective == "dpo":
                ref_pos, ref_neg = reference_logprobs(reference, pos_ids, pos_mask,
                                                      neg_ids, neg_mask, args)
                loss, stats = dpo_loss(pol_pos, pol_neg, ref_pos, ref_neg,
                                       args.beta, args.lambda_sft)
            else:
                # Supervised training on the positives alone, response-only loss.
                # The negatives are not used and no reference model is needed.
                loss = -pol_pos.mean()
                stats = dict(sft_loss=loss.item())
            (loss / args.grad_accum).backward()
            last_micro = micro + 1 == math.ceil(len(pairs) / args.batch_size)
            if (micro + 1) % args.grad_accum == 0 or last_micro:
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                stats["grad_norm"] = float(grad_norm)
            stats["loss"] = loss.item()
            seen += 1
            for k, v in stats.items():
                running[k] = running.get(k, 0.0) + v
                running_n[k] = running_n.get(k, 0) + 1
                window[k] = window.get(k, 0.0) + v
                window_n[k] = window_n.get(k, 0) + 1
            if seen % args.log_every == 0:
                avg = {k: window[k] / window_n[k] for k in window}
                window, window_n = {}, {}
                record = dict(epoch=epoch + 1, step=step, lr=scheduler.get_last_lr()[0], **avg)
                history["train"].append(record)
                if hasattr(batcher, "set_postfix"):
                    shown = {k: f"{avg[k]:.4f}" for k in ("loss", "accuracy", "margin")
                             if k in avg}
                    batcher.set_postfix(**shown)
        epoch_avg = {k: running[k] / running_n[k] for k in running}
        shown = " ".join(f"{k}={epoch_avg[k]:.4f}" for k in
                         ("loss", "dpo_loss", "sft_loss", "accuracy", "margin")
                         if k in epoch_avg)
        print(f"epoch {epoch + 1}: {shown} ({time.time() - t0:.0f}s)")

        dev_stats = (evaluate(policy, reference, dev_pairs, args, stoi, args.device)
                     if dev_pairs and args.objective == "dpo" else {})
        if dev_stats:
            history["dev"].append(dict(epoch=epoch + 1, **dev_stats))
            print(f"  preference dev: acc={dev_stats['accuracy']:.3f} "
                  f"margin={dev_stats['margin']:.4f} dpo={dev_stats['dpo_loss']:.4f}")

        gen_metrics = {}
        if gen_questions:
            _, gen_metrics, gen_failures = run_generation_eval(
                policy, gen_questions, stoi, itos, args, f"epoch {epoch + 1}")
            history["generated_dev"].append(dict(epoch=epoch + 1, **gen_metrics))
            failures_log.append(dict(epoch=epoch + 1, **gen_failures))

        payload = {
            "model_state_dict": policy.state_dict(),  # key expected by Step 8
            "model_args": model_args,
            "epoch": epoch + 1,
            "config": vars(args),
            "train_metrics": epoch_avg,
            "dev_metrics": dev_stats,
            "generated_dev_metrics": gen_metrics,
        }
        torch.save(payload, out_path)
        print(f"Saved checkpoint to {out_path}")
        if not args.save_best:
            pass
        elif gen_metrics:
            # Select on what Task 3 actually measures. Preference accuracy
            # saturates inside one epoch and would always pick the first.
            if gen_metrics["answer_accuracy"] > best_generated:
                best_generated = gen_metrics["answer_accuracy"]
                torch.save(payload, generated_best_path)
                print(f"  new best generated accuracy ({best_generated:.1%}) "
                      f"-> {generated_best_path}")
        else:
            # Only when generated evaluation is switched off. Format the tuple
            # element-wise; "{tuple:.3f}" raises TypeError.
            score = (dev_stats.get("accuracy", epoch_avg.get("accuracy", 0.0)),
                     dev_stats.get("margin", epoch_avg.get("margin", 0.0)))
            if score > best:
                best = score
                best_path = out_path.with_name(out_path.stem + "_best.pt")
                torch.save(payload, best_path)
                print(f"  new best preference (acc={score[0]:.3f}, "
                      f"margin={score[1]:.4f}) -> {best_path}")

    if failures_log:
        failures_path = out_path.with_name("task2_generation_failures.json")
        failures_path.write_text(json.dumps(failures_log, indent=2))
        print(f"Wrote {failures_path}")
    history_path = out_path.with_name("task2_history.json")
    history_path.write_text(json.dumps(history, indent=2, default=str))
    print(f"Wrote {history_path}")
    plot_history(history, out_path.with_name("task2_training_curves.png"))
    plot_generated_accuracy(history, out_path.with_name("task2_generated_accuracy.png"))
    if history["generated_dev"]:
        first, last = history["generated_dev"][0], history["generated_dev"][-1]
        print(f"generated answer accuracy: {first['answer_accuracy']:.1%} (epoch "
              f"{first['epoch']}) -> {last['answer_accuracy']:.1%} (epoch {last['epoch']}); "
              f"best {best_generated:.1%}")
    return history


def plot_generated_accuracy(history, path):
    rows = history.get("generated_dev") or []
    if len(rows) < 2:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.figure()
    plt.plot([r["epoch"] for r in rows], [100 * r["answer_accuracy"] for r in rows],
             "o-", label="answer accuracy")
    plt.plot([r["epoch"] for r in rows], [100 * r["fully_verified_accuracy"] for r in rows],
             "s--", label="fully verified")
    plt.xlabel("Epoch (0 = before training)")
    plt.ylabel("Generated development accuracy (%)")
    plt.ylim(0, 100)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Wrote {path}")


def plot_history(history, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipped the curve")
        return
    train = history["train"]
    if not train:
        return
    xs = list(range(len(train)))
    # The sft objective logs no preference metrics, so draw only what exists.
    have = lambda key: all(key in r for r in train)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(xs, [r["loss"] for r in train], label="total")
    for key, label in (("dpo_loss", "dpo"), ("sft_loss", "positive NLL")):
        if have(key):
            axes[0].plot(xs, [r[key] for r in train], label=label)
    axes[0].set_title("loss"); axes[0].legend()
    if have("accuracy"):
        axes[1].plot(xs, [r["accuracy"] for r in train], label="train")
        if history["dev"]:
            step = max(1, len(xs) // max(1, len(history["dev"])))
            axes[1].plot([min(len(xs) - 1, (i + 1) * step - 1) for i in range(len(history["dev"]))],
                         [r["accuracy"] for r in history["dev"]], "o-", label="dev")
        axes[1].legend()
    axes[1].set_title("preference accuracy")
    if have("margin"):
        axes[2].plot(xs, [r["margin"] for r in train])
    axes[2].set_title("implicit reward margin")
    for ax in axes:
        ax.set_xlabel(f"logged step (every {history['config']['log_every']} batches)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
