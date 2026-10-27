# Task 3 generation/scoring handoff for Task 2

This code measures the model's actual generated math answers, rather than its ability to rank supplied positive/negative responses. `math_evaluation.py` is safe to import: it does not load a model, train, or run evaluation at import time. It restores the policy's previous train/eval mode after evaluation, uses no gradients, and never modifies parameters or checkpoint files.

## Files

- `math_evaluation.py`: reusable exact arithmetic/equation oracle, conservative response scorer, fixed balanced development selection, and batched greedy generation.
- `evaluate_task3.py`: standalone runner. By default it evaluates `dpo/dpo.pt` on 100 DEVELOPMENT questions (20 per category). Use its `--checkpoint` argument to compare another actual checkpoint.

Required existing project assets: root `model.py`, `sft/meta.pkl`, actual trained `dpo/dpo.pt`, and `dpo/task1_dev_pairs.json`. The full SFT checkpoint is not needed for this evaluation. Install PyTorch and NumPy in the active environment if needed.

## Standalone development check

From the repository root:

```bash
python dpo/evaluate_task3.py
```

This writes `dpo/task3_generation_results.json` and prints every answer plus summary metrics. Its default score is DEVELOPMENT accuracy, not final test accuracy. To evaluate an existing frozen question set:

```bash
python dpo/evaluate_task3.py --checkpoint dpo/dpo.pt --questions path/to/questions.json --out dpo/my_results.json
```

A frozen question file is a JSON list of records with `question`, integer `expected_answer`, and `category` fields. The expected answers are checked independently; only question strings enter generation. Categories should be addition, subtraction, multiplication, division, or algebra. Both `model_state_dict` and `model` checkpoint formats and `_orig_mod.` prefixes are supported.

## Add generated development accuracy to Task 2

In the training notebook, running from `dpo/`, import the module and select the questions ONCE before the epoch loop:

```python
from pathlib import Path
from math_evaluation import select_development_questions, evaluate_generated_math

# dev_lines is the EXISTING held-out development-pair list, never training lines.
generation_dev_questions = select_development_questions(
    dev_lines, per_category=20, seed=3003
)
best_generated_accuracy = -1.0
generated_dev_history = []
```

In a training script at `dpo/train_dpo.py`, the same import works. From a project-root notebook use `from dpo.math_evaluation import ...` instead.

After each epoch, run the following. In the notebook the trained policy is named `gpt`; in the CLI trainer use `policy` instead. `epoch` is zero-based in the existing loop.

```python
report = evaluate_generated_math(
    gpt, generation_dev_questions, stoi, itos,
    device=device, batch_size=16, max_new_tokens=64
)
metrics = report['summary']['overall']
accuracy = metrics['answer_accuracy']
generated_dev_history.append({'epoch': epoch + 1, **metrics})
print(f"Generated development accuracy: {metrics['answer_correct']}/{metrics['count']} "
      f"({accuracy:.1%})")

# Separate filename avoids overwriting the final dpo.pt checkpoint.
if accuracy > best_generated_accuracy:
    best_generated_accuracy = accuracy
    torch.save({
        'model_state_dict': gpt.state_dict(),
        'model_args': ckpt['model_args'],
        'epoch': epoch + 1,
        'generated_dev_metrics': metrics,
    }, 'dpo_generated_best.pt')  # Notebook working directory is dpo/.
```

For the CLI trainer, replace `gpt` with `policy`, `ckpt['model_args']` with `model_args`, `device` with `args.device`, and the output path with `out_path.with_name('dpo_generated_best.pt')`. Load `itos` from `load_tokeniser()` as well as `stoi` (the current CLI discards it). Use the same question list and decoding settings across epochs. Do not choose questions based on whether the model answers them correctly.

Continue saving `dpo.pt` as the final-epoch checkpoint. Save generated-development history with the training history, for example:

```python
history['generated_dev'] = generated_dev_history
```

Plot after training:

```python
plt.figure()
plt.plot([r['epoch'] for r in generated_dev_history],
         [100 * r['answer_accuracy'] for r in generated_dev_history], 'o-')
plt.xlabel('Epoch')
plt.ylabel('Generated development answer accuracy (%)')
plt.ylim(0, 100)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('task2_generated_accuracy.png', dpi=150)
plt.show()
```

Keep the existing preference plots as separate diagnostics. Evaluation measures progress; it does not itself improve the model. The next proposed controlled training experiment is `lambda_sft=1.0` instead of `0.1`, keeping other settings fixed initially. That experiment has NOT been run as part of this handoff and no improved score is claimed.

## Scoring and limitations

Generation appends one space to each question, matching Task 2 preprocessing. It uses deterministic argmax decoding and equal-length batches. The supplied `generate()` is single-item only and returns `(tokens, hidden_state)`; this helper uses the same forward interface and period/newline stopping. Requests exceeding the model context fail rather than silently cropping.

`answer_accuracy` measures a standard `The answer is N` integer prefix. `fully_verified_accuracy` additionally verifies the explanation's arithmetic equality and its relevance to the original operation/equation. Wrong answers and unparsed answers count as incorrect. Unrecognized explanation wording is conservatively reported as unparsed, not proof of mathematical incorrectness. Truncated responses are flagged separately. The oracle supports the supplied one-operation arithmetic and simple integer algebra forms, not arbitrary prose or multi-step equations.

Development is used to select checkpoints. Do not train on those records or use the final test for selection. The previously examined 100-question benchmark is now diagnostic because both checkpoints were compared on it; after tuning, use a NEW untouched final test. Development selection should represent the intended test difficulty. Comparing many checkpoints on development is expected, but must not be described as an independent final test.

## Checks actually run before this push

The module was run on the real epoch-5 `dpo.pt`. All 100 greedy predictions and summary metrics exactly matched the earlier Task 3 evaluator (12/100 correct). It restored training mode and created no parameter gradients. Development selection was checked for deterministic, balanced, unique questions. The standalone CLI also ran successfully on a five-question development smoke test and reopened its saved JSON output. This handoff does not claim a better trained model or a 60% score. Integration into the epoch loop and further training remain Task 2 work.

Sources: supplied SC3000/CZ3005 assignment; supplied `model.py`, vocabulary, trained checkpoint, and project data. Base project: https://github.com/zhouyuan888888/NanoGPT-Math . AI assistance: OpenAI Codex assisted with implementation, explanations and execution checks; review and acknowledge that assistance under the professor's stated permission.
