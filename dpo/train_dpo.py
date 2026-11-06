import argparse
import inspect
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
DPO_DIR = ROOT / "dpo"
TERMINATOR = "\n\n\n\n"  # supplied preprocessing appends four newlines
PAD_ID = 0  # supplied padding id; also the id of '\n', hence the explicit masks
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/"
SMOKE_MAX_NEW_TOKENS = 24
SMOKE_MODEL_ARGS = dict(n_layer=2, n_head=2, n_embd=64, bias=False, dropout=0.0)
EPOCH_SUMMARY_KEYS = ("loss", "dpo_loss", "sft_loss", "accuracy", "margin")
PROGRESS_KEYS = ("loss", "accuracy", "margin")


def add_path_arguments(parser):
    # Defaults mirror the supplied Step 2 configuration.
    parser.add_argument("--pairs", default=str(DPO_DIR / "pos_neg_pairs.json"))
    parser.add_argument("--dev-pairs", default=str(DPO_DIR / "task1_dev_pairs.json"))
    parser.add_argument("--checkpoint", default=str(ROOT / "sft" / "gpt.pt"))
    parser.add_argument("--out", default=str(DPO_DIR / "dpo.pt"))
    parser.add_argument("--overwrite", action="store_true",
                        help="allow --out to replace an existing checkpoint")


def add_optimisation_arguments(parser):
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-frac", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)


def add_objective_arguments(parser):
    # beta 2.0 on mean-normalised log-probs == the supplied hint's "/ beta" at
    # beta=0.5; see the Task 2 notes in the notebook.
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--lambda-sft", type=float, default=0.1)
    parser.add_argument("--reference", choices=["sft", "none"], default="sft")
    parser.add_argument("--loss-scope", choices=["response", "full"], default="response")
    parser.add_argument("--length-norm", choices=["mean", "sum"], default="mean")
    # "dpo" is the Task 2 preference objective; "sft" is plain supervised
    # training on the positives only, for the SFT-then-DPO pipeline.
    parser.add_argument("--objective", choices=["dpo", "sft"], default="dpo")


def add_run_arguments(parser):
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="use the first N training pairs")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-best", action="store_true", default=True)
    parser.add_argument("--no-save-best", dest="save_best", action="store_false")
    parser.add_argument("--smoke", action="store_true",
                        help="random tiny model, 512 pairs, 1 epoch: checks the pipeline only")


def add_generation_arguments(parser):
    # Generated-answer evaluation (Task 3 scoring, reused verbatim).
    parser.add_argument("--gen-per-category", type=int, default=20)
    parser.add_argument("--gen-seed", type=int, default=3003)
    parser.add_argument("--gen-batch-size", type=int, default=16)
    parser.add_argument("--gen-max-new-tokens", type=int, default=64)
    parser.add_argument("--no-gen-eval", dest="gen_eval", action="store_false", default=True,
                        help="skip generated-answer evaluation (preference metrics only)")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="DPO fine-tuning (Task 2)")
    add_path_arguments(parser)
    add_optimisation_arguments(parser)
    add_objective_arguments(parser)
    add_run_arguments(parser)
    add_generation_arguments(parser)
    args = parser.parse_args(argv)
    if args.smoke:
        apply_smoke_defaults(args)
    return args


def apply_smoke_defaults(args):
    args.epochs, args.limit, args.batch_size = 1, 512, 32
    args.log_every = 5
    args.gen_per_category = 2
    # The smoke model's context is max_length, so keep prompt + budget inside it.
    args.gen_max_new_tokens = min(args.gen_max_new_tokens, SMOKE_MAX_NEW_TOKENS)
    if args.out == str(DPO_DIR / "dpo.pt"):  # keep the real checkpoint intact
        args.out = str(DPO_DIR / "smoke_dpo.pt")


def pick_device(requested=None):
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_tokeniser(root=ROOT):
    with (root / "sft" / "meta.pkl").open("rb") as f:
        meta = pickle.load(f)
    return meta["stoi"], meta["itos"], meta["vocab_size"]


def load_pairs(path, limit=None):
    with open(path) as f:
        pairs = json.load(f)
    assert isinstance(pairs, list) and pairs, f"no records in {path}"
    for record in pairs:
        assert set(record) == {"positive", "negative"}, f"unexpected schema: {sorted(record)}"
    return pairs[:limit] if limit else pairs


def prompt_length(pair):
    question = pair["positive"].split("?", 1)[0] + "?"
    for key in ("positive", "negative"):
        assert pair[key].startswith(question + " "), f"question is not a shared prefix: {pair}"
    return len(question) + 1


def encode_pair(pair, stoi, max_length, loss_scope):
    start = prompt_length(pair) if loss_scope == "response" else 0
    encoded = []
    for key in ("negative", "positive"):
        ids = [stoi[c] for c in pair[key] + TERMINATOR]
        # No-truncation invariant from Task 1 (longest record is 62 tokens).
        assert len(ids) <= max_length, f"{len(ids)} > max_length={max_length}: {pair[key]!r}"
        mask = [start <= i < len(ids) for i in range(max_length)]
        encoded.append((ids + [PAD_ID] * (max_length - len(ids)), mask))
    return encoded


def get_batches(pairs, batch_size, stoi, max_length, loss_scope, device, rng=None):
    order = list(range(len(pairs)))
    if rng is not None:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        chunk = [pairs[j] for j in order[start:start + batch_size]]
        encoded = [encode_pair(p, stoi, max_length, loss_scope) for p in chunk]
        tensors = []
        for slot in (0, 1):  # 0 = negative, 1 = positive
            tensors.append(torch.tensor([e[slot][0] for e in encoded],
                                        dtype=torch.long, device=device))
            tensors.append(torch.tensor([e[slot][1] for e in encoded],
                                        dtype=torch.bool, device=device))
        yield tuple(tensors)


def count_training_batches(pairs, batch_size):
    return math.ceil(len(pairs) / batch_size)


def reject_lfs_pointer(checkpoint_path):
    with open(checkpoint_path, "rb") as f:
        if f.read(80).startswith(LFS_POINTER_PREFIX):
            raise RuntimeError(f"{checkpoint_path} is an LFS pointer; run `git lfs pull` "
                               "or pass --checkpoint with the real weights")


def build_smoke_model(GPT, GPTConfig, device, vocab_size, max_length):
    model_args = dict(SMOKE_MODEL_ARGS, block_size=max_length, vocab_size=vocab_size)
    return GPT(GPTConfig(**model_args)).to(device), model_args


def load_gpt(GPT, GPTConfig, checkpoint_path, device, smoke=False, vocab_size=74, max_length=64):
    if smoke:
        return build_smoke_model(GPT, GPTConfig, device, vocab_size, max_length)
    reject_lfs_pointer(checkpoint_path)
    # Trusted local checkpoint; torch >= 2.6 defaults to weights_only=True.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_args = checkpoint["model_args"]
    model = GPT(GPTConfig(**model_args))
    # The SFT checkpoint stores "model"; our own runs store "model_state_dict",
    # so accept either and resume from a DPO checkpoint as well.
    state = checkpoint.get("model_state_dict", checkpoint.get("model"))
    if state is None:
        raise KeyError(f"{checkpoint_path}: no 'model' or 'model_state_dict' key "
                       f"(found {sorted(checkpoint)})")
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
    batch, time_steps, vocab = logits.shape
    token_logprob = -F.cross_entropy(
        logits.reshape(-1, vocab).float(), targets.reshape(-1), reduction="none"
    ).view(batch, time_steps)
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
    supervised = -pol_pos.mean()
    stats = dict(
        dpo_loss=preference.item(),
        sft_loss=supervised.item(),
        accuracy=(logits > 0).float().mean().item(),
        margin=(pos_ratio - neg_ratio).mean().item(),
        pos_reward=pos_ratio.mean().item(),
        neg_reward=neg_ratio.mean().item(),
    )
    return preference + lambda_sft * supervised, stats


def build_optimizer(model, lr, weight_decay):
    decay, no_decay = [], []
    for _, param in model.named_parameters():
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


class RunningAverages:
    """Per-key means; counts are per key because grad_norm only exists on step boundaries."""

    def __init__(self):
        self.totals = {}
        self.counts = {}

    def add(self, stats):
        for key, value in stats.items():
            self.totals[key] = self.totals.get(key, 0.0) + value
            self.counts[key] = self.counts.get(key, 0) + 1

    def mean(self):
        return {key: self.totals[key] / self.counts[key] for key in self.totals}


@torch.no_grad()
def evaluate(policy, reference, pairs, args, stoi, device):
    was_training = policy.training
    policy.eval()
    averages = RunningAverages()
    for neg_ids, neg_mask, pos_ids, pos_mask in get_batches(
        pairs, args.batch_size, stoi, args.max_length, args.loss_scope, device
    ):
        pol_pos = sequence_logprob(policy, pos_ids, pos_mask, args.length_norm)
        pol_neg = sequence_logprob(policy, neg_ids, neg_mask, args.length_norm)
        ref_pos, ref_neg = reference_logprobs(reference, pos_ids, pos_mask,
                                              neg_ids, neg_mask, args)
        _, stats = dpo_loss(pol_pos, pol_neg, ref_pos, ref_neg, args.beta, args.lambda_sft)
        averages.add(stats)
    if was_training:
        policy.train()
    return averages.mean()


def reference_logprobs(reference, pos_ids, pos_mask, neg_ids, neg_mask, args):
    if reference is None:
        zero = torch.zeros(pos_ids.size(0), device=pos_ids.device)
        return zero, zero
    with torch.no_grad():
        return (sequence_logprob(reference, pos_ids, pos_mask, args.length_norm),
                sequence_logprob(reference, neg_ids, neg_mask, args.length_norm))


def summarise_failure_modes(wrong):
    """Separate a wrong number from an unparsable response or one that ran out of budget."""
    return dict(wrong_answer=sum(r["parsed_answer"] is not None for r in wrong),
                unparsed=sum(r["parsed_answer"] is None for r in wrong),
                truncated=sum(r["truncated"] for r in wrong))


def print_generation_metrics(label, metrics, modes):
    print(f"  {label}: generated answer accuracy "
          f"{metrics['answer_correct']}/{metrics['count']} ({metrics['answer_accuracy']:.1%}), "
          f"fully verified {metrics['fully_verified_accuracy']:.1%} | failures: "
          f"{modes['wrong_answer']} wrong math, {modes['unparsed']} unparsed, "
          f"{modes['truncated']} truncated")


def run_generation_eval(policy, questions, stoi, itos, args, label):
    """Task 3 scoring, imported unchanged from math_evaluation."""
    from math_evaluation import evaluate_generated_math
    report = evaluate_generated_math(policy, questions, stoi, itos, device=args.device,
                                     batch_size=args.gen_batch_size,
                                     max_new_tokens=args.gen_max_new_tokens)
    metrics = report["summary"]["overall"]
    wrong = [r for r in report["results"] if not r["answer_correct"]]
    modes = summarise_failure_modes(wrong)
    print_generation_metrics(label, metrics, modes)
    cases = [{k: r[k] for k in ("question", "expected_answer", "category", "response",
                                "parsed_answer", "stop_reason", "truncated",
                                "explanation_status")} for r in wrong]
    return report, metrics, dict(stage=label, failure_modes=modes, cases=cases)


def assert_no_dev_leakage(pairs, dev_pairs):
    def canonical(record):
        return "".join(record["positive"].split("?", 1)[0].split())

    train_questions = {canonical(p) for p in pairs}
    overlap = sum(canonical(p) in train_questions for p in dev_pairs)
    assert overlap == 0, f"{overlap} dev questions leaked into training"


def build_reference(GPT, GPTConfig, policy, args, vocab_size):
    if args.objective != "dpo" or args.reference != "sft":
        return None
    reference, _ = load_gpt(GPT, GPTConfig, args.checkpoint, args.device,
                            args.smoke, vocab_size, args.max_length)
    if args.smoke:  # the ablation needs identical weights, not a second init
        reference.load_state_dict(policy.state_dict())
    reference.eval()
    reference.requires_grad_(False)
    return reference


def prepare_output_path(args):
    out_path = Path(args.out)
    if out_path.exists() and not (args.overwrite or args.smoke):
        raise SystemExit(f"{out_path} already exists. Pass --overwrite to replace it, "
                         "or choose another --out so existing checkpoints are preserved.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def select_generation_questions(args, dev_pairs):
    """Chosen ONCE from development pairs; the identical list is reused every evaluation."""
    if not (args.gen_eval and dev_pairs):
        return []
    from math_evaluation import select_development_questions
    questions = select_development_questions(dev_pairs, args.gen_per_category, args.gen_seed)
    print(f"generated-answer evaluation: {len(questions)} development questions "
          f"({args.gen_per_category}/category, seed {args.gen_seed})")
    return questions


def wrap_with_progress(batcher, total, epoch, epochs):
    try:
        from tqdm import tqdm
    except ImportError:
        return batcher
    return tqdm(batcher, total=total, desc=f"epoch {epoch + 1}/{epochs}")


def compute_batch_loss(policy, reference, batch, args):
    neg_ids, neg_mask, pos_ids, pos_mask = batch
    use_bf16 = args.device == "cuda" and torch.cuda.is_bf16_supported()
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16
                else torch.autocast("cpu", enabled=False))
    with autocast:
        pol_pos = sequence_logprob(policy, pos_ids, pos_mask, args.length_norm)
        if args.objective == "dpo":
            pol_neg = sequence_logprob(policy, neg_ids, neg_mask, args.length_norm)
    if args.objective != "dpo":
        # Supervised training on the positives alone, response-only loss.
        # The negatives are not used and no reference model is needed.
        loss = -pol_pos.mean()
        return loss, dict(sft_loss=loss.item())
    ref_pos, ref_neg = reference_logprobs(reference, pos_ids, pos_mask, neg_ids, neg_mask, args)
    return dpo_loss(pol_pos, pol_neg, ref_pos, ref_neg, args.beta, args.lambda_sft)


def apply_optimizer_step(policy, optimizer, scheduler, max_grad_norm):
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return float(grad_norm)


def train_one_epoch(policy, reference, pairs, args, stoi, epoch, optimizer, scheduler, history):
    rng = random.Random(args.seed + epoch)
    total_batches = count_training_batches(pairs, args.batch_size)
    batcher = get_batches(pairs, args.batch_size, stoi, args.max_length,
                          args.loss_scope, args.device, rng)
    batcher = wrap_with_progress(batcher, total_batches, epoch, args.epochs)
    epoch_averages, window = RunningAverages(), RunningAverages()
    seen, step, started = 0, 0, time.time()
    optimizer.zero_grad(set_to_none=True)
    for micro, batch in enumerate(batcher):
        loss, stats = compute_batch_loss(policy, reference, batch, args)
        (loss / args.grad_accum).backward()
        last_micro = micro + 1 == total_batches
        if (micro + 1) % args.grad_accum == 0 or last_micro:
            stats["grad_norm"] = apply_optimizer_step(policy, optimizer, scheduler,
                                                      args.max_grad_norm)
            step += 1
        stats["loss"] = loss.item()
        seen += 1
        epoch_averages.add(stats)
        window.add(stats)
        if seen % args.log_every == 0:
            averaged = window.mean()
            window = RunningAverages()
            history["train"].append(dict(epoch=epoch + 1, step=step,
                                         lr=scheduler.get_last_lr()[0], **averaged))
            if hasattr(batcher, "set_postfix"):
                batcher.set_postfix(**{k: f"{averaged[k]:.4f}"
                                       for k in PROGRESS_KEYS if k in averaged})
    epoch_avg = epoch_averages.mean()
    shown = " ".join(f"{k}={epoch_avg[k]:.4f}" for k in EPOCH_SUMMARY_KEYS if k in epoch_avg)
    print(f"epoch {epoch + 1}: {shown} ({time.time() - started:.0f}s)")
    return epoch_avg


def evaluate_preferences(policy, reference, dev_pairs, args, stoi, epoch, history):
    if not (dev_pairs and args.objective == "dpo"):
        return {}
    dev_stats = evaluate(policy, reference, dev_pairs, args, stoi, args.device)
    history["dev"].append(dict(epoch=epoch + 1, **dev_stats))
    print(f"  preference dev: acc={dev_stats['accuracy']:.3f} "
          f"margin={dev_stats['margin']:.4f} dpo={dev_stats['dpo_loss']:.4f}")
    return dev_stats


def build_checkpoint_payload(policy, model_args, args, epoch, epoch_avg, dev_stats, gen_metrics):
    return {
        "model_state_dict": policy.state_dict(),  # key expected by Step 8
        "model_args": model_args,
        "epoch": epoch + 1,
        "config": vars(args),
        "train_metrics": epoch_avg,
        "dev_metrics": dev_stats,
        "generated_dev_metrics": gen_metrics,
    }


def save_best_by_generated_accuracy(payload, gen_metrics, best_generated, path):
    """Select on what Task 3 measures; preference accuracy saturates inside one epoch."""
    if gen_metrics["answer_accuracy"] <= best_generated:
        return best_generated
    torch.save(payload, path)
    print(f"  new best generated accuracy ({gen_metrics['answer_accuracy']:.1%}) -> {path}")
    return gen_metrics["answer_accuracy"]


def save_best_by_preference(payload, dev_stats, epoch_avg, best, out_path):
    """Only reached when generated evaluation is switched off."""
    score = (dev_stats.get("accuracy", epoch_avg.get("accuracy", 0.0)),
             dev_stats.get("margin", epoch_avg.get("margin", 0.0)))
    if score <= best:
        return best
    best_path = out_path.with_name(out_path.stem + "_best.pt")
    torch.save(payload, best_path)
    print(f"  new best preference (acc={score[0]:.3f}, margin={score[1]:.4f}) -> {best_path}")
    return score


def write_run_artifacts(history, failures_log, out_path):
    if failures_log:
        failures_path = out_path.with_name("task2_generation_failures.json")
        failures_path.write_text(json.dumps(failures_log, indent=2))
        print(f"Wrote {failures_path}")
    history_path = out_path.with_name("task2_history.json")
    history_path.write_text(json.dumps(history, indent=2, default=str))
    print(f"Wrote {history_path}")
    plot_history(history, out_path.with_name("task2_training_curves.png"))
    plot_generated_accuracy(history, out_path.with_name("task2_generated_accuracy.png"))


def print_generated_summary(history, best_generated):
    rows = history["generated_dev"]
    if not rows:
        return
    first, last = rows[0], rows[-1]
    print(f"generated answer accuracy: {first['answer_accuracy']:.1%} (epoch "
          f"{first['epoch']}) -> {last['answer_accuracy']:.1%} (epoch {last['epoch']}); "
          f"best {best_generated:.1%}")


def describe_run(args, pairs, dev_pairs, steps_per_epoch, total_steps):
    use_bf16 = args.device == "cuda" and torch.cuda.is_bf16_supported()
    print(f"device={args.device} pairs={len(pairs)} dev={len(dev_pairs)} "
          f"steps/epoch={steps_per_epoch} total_steps={total_steps} bf16={use_bf16}")
    print(f"beta={args.beta} lambda_sft={args.lambda_sft} reference={args.reference} "
          f"loss_scope={args.loss_scope} length_norm={args.length_norm}")


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_datasets(args):
    pairs = load_pairs(args.pairs, args.limit)
    dev_pairs = load_pairs(args.dev_pairs) if Path(args.dev_pairs).exists() else []
    assert_no_dev_leakage(pairs, dev_pairs)
    return pairs, dev_pairs


def build_schedule(policy, args, pairs):
    steps_per_epoch = math.ceil(count_training_batches(pairs, args.batch_size) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    optimizer = build_optimizer(policy, args.lr, args.weight_decay)
    scheduler = build_scheduler(optimizer, total_steps, args.warmup_frac, args.min_lr_ratio)
    return optimizer, scheduler, steps_per_epoch, total_steps


def record_baseline_generation(policy, gen_questions, stoi, itos, args, history, failures_log):
    if not gen_questions:
        return
    _, metrics, failures = run_generation_eval(policy, gen_questions, stoi, itos,
                                               args, "before training")
    history["generated_dev"].append(dict(epoch=0, **metrics))
    failures_log.append(dict(epoch=0, **failures))


def record_epoch_generation(policy, gen_questions, stoi, itos, args, epoch,
                            history, failures_log):
    if not gen_questions:
        return {}
    _, metrics, failures = run_generation_eval(policy, gen_questions, stoi, itos,
                                               args, f"epoch {epoch + 1}")
    history["generated_dev"].append(dict(epoch=epoch + 1, **metrics))
    failures_log.append(dict(epoch=epoch + 1, **failures))
    return metrics


def run_epochs(policy, reference, model_args, pairs, dev_pairs, gen_questions, args,
               stoi, itos, optimizer, scheduler, history, failures_log, out_path):
    best = (-float("inf"), -float("inf"))
    best_generated = -1.0
    generated_best_path = out_path.with_name(out_path.stem + "_generated_best.pt")
    for epoch in range(args.epochs):
        epoch_avg = train_one_epoch(policy, reference, pairs, args, stoi, epoch,
                                    optimizer, scheduler, history)
        dev_stats = evaluate_preferences(policy, reference, dev_pairs, args, stoi, epoch, history)
        gen_metrics = record_epoch_generation(policy, gen_questions, stoi, itos, args, epoch,
                                              history, failures_log)
        payload = build_checkpoint_payload(policy, model_args, args, epoch, epoch_avg,
                                           dev_stats, gen_metrics)
        torch.save(payload, out_path)
        print(f"Saved checkpoint to {out_path}")
        if not args.save_best:
            continue
        if gen_metrics:
            best_generated = save_best_by_generated_accuracy(payload, gen_metrics,
                                                             best_generated, generated_best_path)
        else:
            best = save_best_by_preference(payload, dev_stats, epoch_avg, best, out_path)
    return best_generated


def main(argv=None):
    args = parse_args(argv)
    args.device = pick_device(args.device)
    seed_everything(args.seed)

    sys.path.insert(0, str(ROOT))
    from model import GPT, GPTConfig  # supplied, unchanged

    stoi, itos, vocab_size = load_tokeniser()
    pairs, dev_pairs = load_datasets(args)
    policy, model_args = load_gpt(GPT, GPTConfig, args.checkpoint, args.device,
                                  args.smoke, vocab_size, args.max_length)
    policy.train()
    reference = build_reference(GPT, GPTConfig, policy, args, vocab_size)
    optimizer, scheduler, steps_per_epoch, total_steps = build_schedule(policy, args, pairs)
    describe_run(args, pairs, dev_pairs, steps_per_epoch, total_steps)

    history = {"config": vars(args), "model_args": model_args, "train": [], "dev": [],
               "generated_dev": []}
    failures_log = []
    out_path = prepare_output_path(args)
    gen_questions = select_generation_questions(args, dev_pairs)
    record_baseline_generation(policy, gen_questions, stoi, itos, args, history, failures_log)

    best_generated = run_epochs(policy, reference, model_args, pairs, dev_pairs, gen_questions,
                                args, stoi, itos, optimizer, scheduler, history, failures_log,
                                out_path)
    write_run_artifacts(history, failures_log, out_path)
    print_generated_summary(history, best_generated)
    return history


def load_pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def plot_generated_accuracy(history, path):
    rows = history.get("generated_dev") or []
    if len(rows) < 2:
        return
    plt = load_pyplot()
    if plt is None:
        return
    epochs = [r["epoch"] for r in rows]
    plt.figure()
    plt.plot(epochs, [100 * r["answer_accuracy"] for r in rows], "o-", label="answer accuracy")
    plt.plot(epochs, [100 * r["fully_verified_accuracy"] for r in rows], "s--",
             label="fully verified")
    plt.xlabel("Epoch (0 = before training)")
    plt.ylabel("Generated development accuracy (%)")
    plt.ylim(0, 100)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Wrote {path}")


def plot_loss_panel(axis, train, has_key):
    steps = list(range(len(train)))
    axis.plot(steps, [r["loss"] for r in train], label="total")
    for key, label in (("dpo_loss", "dpo"), ("sft_loss", "positive NLL")):
        if has_key(key):
            axis.plot(steps, [r[key] for r in train], label=label)
    axis.set_title("loss")
    axis.legend()


def plot_accuracy_panel(axis, train, dev, has_key):
    steps = list(range(len(train)))
    if has_key("accuracy"):
        axis.plot(steps, [r["accuracy"] for r in train], label="train")
        if dev:
            stride = max(1, len(steps) // max(1, len(dev)))
            axis.plot([min(len(steps) - 1, (i + 1) * stride - 1) for i in range(len(dev))],
                      [r["accuracy"] for r in dev], "o-", label="dev")
        axis.legend()
    axis.set_title("preference accuracy")


def plot_history(history, path):
    plt = load_pyplot()
    if plt is None:
        print("matplotlib unavailable; skipped the curve")
        return
    train = history["train"]
    if not train:
        return
    # The sft objective logs no preference metrics, so draw only what exists.
    def has_key(key):
        return all(key in record for record in train)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    plot_loss_panel(axes[0], train, has_key)
    plot_accuracy_panel(axes[1], train, history["dev"], has_key)
    if has_key("margin"):
        axes[2].plot(list(range(len(train))), [r["margin"] for r in train])
    axes[2].set_title("implicit reward margin")
    for axis in axes:
        axis.set_xlabel(f"logged step (every {history['config']['log_every']} batches)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
