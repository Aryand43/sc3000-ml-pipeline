# Generation/scoring and final-submission status

The complete single-notebook submission is now `dpo/dpo.ipynb`, with Tasks 1–3 code, recorded training plots and fresh evaluation outputs. The former unfinished Step 8 has been replaced. Team contributions are listed at its beginning.

## Results and checkpoint

Use the provided final `dpo.pt`, SHA256 `7ee2ada8e5824a39cae2b48a93a59527e9e39ff7619c0586f69c2cadd66dd194`. The old `archive/checkpoints/dpo_best.pt` is historical and not used by the notebook. The archived reports are not final-test evidence. It obtains 79/100 fresh frozen questions, 85/100 on the old diagnostic benchmark, 80/100 development and 4/5 supplied examples. Recorded warm-up/DPO histories are explicitly distinguished from the small training smoke tests and current model inference.

The 100 fresh questions exclude normalized training, development and old-benchmark questions. Do not tune on their results while calling them untouched final evaluation. Development is used for checkpoint selection; the previous 100-question benchmark is diagnostic after inspection. See README for category results and limitations.

## Reuse in Task 2

`math_evaluation.py` is import-safe, restores the model's previous train/eval mode, uses no gradients and sees only question text during generation. `train_dpo.py` already selects the same development questions before training and evaluates them before training and after each epoch.

```python
from math_evaluation import select_development_questions, evaluate_generated_math
questions = select_development_questions(dev_lines, per_category=20, seed=3003)
# Inside each epoch, after optimization:
report = evaluate_generated_math(policy, questions, stoi, itos, device=device)
accuracy = report['summary']['overall']['answer_accuracy']
```

`train_dpo.py` saves `*_generated_best.pt` by this score and plots it. The provided final `dpo.pt` is the selected delivered final-epoch checkpoint; notebook inference verifies its hash. The full notebook embeds the same helper/trainer implementation, so no helper import is needed for grading.

## Standalone checks

From project root:

```bash
python dpo/evaluate_task3.py --checkpoint dpo/dpo.pt --device cpu
python dpo/evaluate_task3.py --checkpoint dpo/dpo.pt --development --device cpu
```

Default is the frozen fresh benchmark, not development. `--questions custom.json` accepts explicit question/expected-answer/category records. The default CLI output is `dpo/task3_cli_results.json`, so notebook result files are preserved. Generation/scoring implementation is unchanged from the verified evaluator; checkpoint loading supports `model_state_dict` and `model`, prefix cleanup and explicit device selection.

Exact integer answer accuracy requires a `The answer is N` prefix. Fully verified accuracy additionally checks a relevant arithmetic/inverse-operation equality. Unrecognized explanation wording is reported separately. Context-budget overflow is refused; token exhaustion is flagged. This is a conservative checker for the supplied problem forms, not a general mathematical proof checker.

The default notebook runs smoke checks and inference without retraining the submitted model. Reproduce 20-epoch supervised warm-up then 5 DPO epochs by enabling `RUN_FULL_TRAINING`; outputs go into a new directory. See README before running, and do not overwrite or rename checkpoint files to make scores appear better.

Source references remain in the notebook and README.
