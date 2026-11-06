# SC3000 / CZ3005 — Teaching NanoGPT to Do Math

> **Start here:** [dpo/dpo.ipynb](dpo/dpo.ipynb)
>
> **Final model:** [dpo/dpo.pt](dpo/dpo.pt)
>
> **Fresh held-out result:** **79/100** (greedy decoding; 20 questions per category)

The notebook is the complete submission entry point: Tasks 1–3 implementation, explanations, saved outputs and failure analysis. You do not need to read the optional Python scripts to assess the project. The supplied `model.py` and `configurator.py` are unchanged.

## Examiner's guide

1. Open **`dpo/dpo.ipynb`** to inspect the saved dataset checks, recorded training curves and all generated test answers without running anything.
2. To verify the result, follow **Run the notebook** below and run all cells. Default execution does not repeat the long training run or overwrite the final model.
3. Use **`dpo/task3_final_summary.json`** for compact scores and **`dpo/task3_final_results.json`** for every prediction. Development accuracy is reported separately from the fresh test.
4. Ignore **`archive/`** when grading final results. It preserves an old checkpoint and superseded diagnostics, excluded from the submission ZIP.

## Which files matter?

| Group | Files | Purpose |
|---|---|---|
| **Main deliverables** | `dpo/dpo.ipynb`, `dpo/dpo.pt`, `dpo/pos_neg_pairs.json` | Complete notebook, final weights and 100,000 training pairs |
| **Required notebook assets** | `model.py`, `configurator.py`, `sft/meta.pkl`, actual `sft/gpt.pt` | Supplied architecture, vocabulary and pretrained weights; a Git LFS pointer is insufficient |
| **Required notebook data** | `dpo/task1_dev_pairs.json`, `dpo/task2_history.json`, `dpo/task2_sft_history.json`, `dpo/task3_fresh_test_questions.json`, `dpo/task3_previous_test_questions.json` | Separate development data, recorded real training histories and frozen evaluation questions |
| **Final evidence** | `dpo/task1_validation.json`, `dpo/task3_final_results.json`, `dpo/task3_final_summary.json`, `dpo/task2_*.png` | Validation, predictions, scores and training plots |
| **Setup and packaging** | `requirements.txt`, `VALIDATION_ENVIRONMENT.json`, `package_submission.py` | Tested environment, execution checks and focused ZIP creation |
| **Optional support** | `dpo/train_dpo.py`, `dpo/evaluate_task3.py`, `dpo/math_evaluation.py`, `dpo/TASK3_HANDOFF.md` | Command-line alternatives and integration notes; implementation is already embedded in the notebook |
| **Regenerable metadata** | `dpo/task1_metadata.json` | Per-pair provenance created by the notebook; ignored in Git and not a required input |
| **Archived — not final results** | `archive/` | Preserved historical checkpoint and diagnostic reports |

## Team contributions

| Team member | Contribution |
|---|---|
| Elisetti Atul Dev | Task 1 — preference dataset |
| Aryan Dutt | Task 2 — training and monitoring |
| Harsh Kamdar | Task 3 — evaluation and analysis |

## Run the notebook

Use Python 3.12 and install `requirements.txt` into the environment used by the notebook kernel. Exact versions used for validation are pinned; `VALIDATION_ENVIRONMENT.json` records them.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name sc3000-math --display-name "SC3000 Math"
```

Open `dpo/dpo.ipynb` in your Jupyter application, select that kernel, and run all cells. Either the project root or `dpo/` is a valid working directory.

Default controls:

- `RUN_FULL_TRAINING=False`: regenerate/validate the same dataset, run small pipeline smoke tests, display recorded long-run training histories, and freshly evaluate the provided final checkpoint. It does not overwrite `dpo/dpo.pt`.
- `RUN_TRAINING_SMOKE=True`: real one-epoch tiny-model SFT and DPO checks on 512 pairs. Their scores are not the project's final performance.
- `RUN_OPTIONAL_NEGATIVE_SMOKE=True`: exercise the actual supplied NanoGPT generation interface on two questions. The submitted preference dataset remains entirely synthetic.
- `EVAL_DEVICE='cpu'`: reference decoding environment; CUDA/MPS can be selected explicitly.

The normal run is designed for quick review. The notebook clearly labels recorded training figures instead of claiming the 20+5-epoch job was rerun during submission checks.

### Checkpoint files and Git LFS

A GitHub ZIP download may contain only a small LFS pointer for `sft/gpt.pt`. A pointer is not a checkpoint. From a Git clone, install Git LFS and run `git lfs pull`; alternatively obtain the supplied actual weights. The notebook and packager reject pointers with a clear message. The final submission ZIP built by `package_submission.py` includes real weights, so its extracted copy needs no LFS access.

## Training method and reproduction

Task 1 contains 100,000 pairs, 20,000 per category, plus 500 separate development pairs. Positives use exact arithmetic and short worked equalities. Synthetic negatives and the optional, conservatively filtered model-negative path are identified separately. Schema, character vocabulary, complete sequence length, label correctness and duplicates are validated.

The successful run used positive-response supervised learning for **20 epochs at lr 0.001**, followed by **5 DPO epochs at lr 0.0001 and lambda_sft 1.0**. Policy and frozen reference start from the selected math-SFT checkpoint. Generated development accuracy, rather than preference accuracy alone, selects checkpoints. Token-mean response-only log probabilities, true-length masks, AdamW, warm-up/cosine scheduling and gradient clipping are documented in the notebook.

To reproduce the long run, set `RUN_FULL_TRAINING=True` in the notebook. It writes into a new `runs/notebook_reproduction/` directory. Existing experimental outputs are protected from overwrite. The submitted final checkpoint remains unchanged.

Equivalent optional CLI commands, from the root:

```bash
python dpo/train_dpo.py --objective sft --checkpoint sft/gpt.pt --lr 0.001 --epochs 20 --out runs/reproduction/math_sft/sft.pt
python dpo/train_dpo.py --objective dpo --checkpoint runs/reproduction/math_sft/sft_generated_best.pt --lambda-sft 1.0 --lr 0.0001 --epochs 5 --out runs/reproduction/sft_then_dpo/dpo.pt
```

Historical absolute paths in the recorded JSON logs describe the original training machine; they are not used by these reproduction commands. The old alternative is preserved at `archive/checkpoints/dpo_best.pt`; it is not selected by the final notebook or included in the submission ZIP.

## Evaluated results

Reference run: actual final `dpo/dpo.pt`, greedy decoding, CPU, PyTorch 2.14.1.

| Set | Correct | Interpretation |
|---|---:|---|
| Fresh frozen benchmark | **79/100** | 20/category, seed 20261106; excluded normalized training, development and old-benchmark questions |
| Previous benchmark | 85/100 | Diagnostic; the old model scored 12/100 on these same questions |
| Development subset | 80/100 | Used during training/selection; not an independent final score |
| Supplied unique examples | 4/5 | Reported separately, with exposure labels |

Fresh categories: addition 20/20, subtraction 19/20, multiplication 8/20, exact division 15/20, algebra 17/20. Fully verified explanation accuracy is also 79/100. No responses were unparsed or truncated. Frozen question JSON, all predictions, checkpoint/data hashes and summaries are supplied in `dpo/task3_final_results.json` and `dpo/task3_final_summary.json`.

All questions/settings were fixed before first inference. No failures were removed or regenerated. Equivalent calculations can still occur across distinct question forms. Multiplication remains weak; a carry-related explanation is a hypothesis, not a proven mechanism. These results do not guarantee the TA's private-test score. SFT's best development score was 81% and DPO's was 80%; this experiment does not demonstrate an accuracy advantage from the DPO stage alone.

Optional standalone evaluation defaults to the frozen benchmark:

```bash
python dpo/evaluate_task3.py --checkpoint dpo/dpo.pt --device cpu
python dpo/evaluate_task3.py --checkpoint dpo/dpo.pt --development --device cpu
```

The CLI writes `dpo/task3_cli_results.json`, separate from notebook results. See `dpo/TASK3_HANDOFF.md` for reuse during training. The earlier development-only report is preserved at `archive/reports/task3_generation_results.json`. It is superseded by the consolidated final reports and is not the fresh-test score.

## Build the submission

After running and saving the notebook:

```bash
python package_submission.py --out YOUR_TEAM_NAME.zip
```

The packager checks complete notebook execution, errors, required files, actual checkpoint bytes, preference count/schema and result/checkpoint consistency. It writes a SHA256 manifest and verifies every archived payload. By default, the ZIP contains the notebook, its required assets/data, environment instructions and final evidence. It excludes `archive/`, experiments, caches, the optional CLI scripts/handoff and bulky regenerable per-pair metadata. All training/scoring code remains embedded in the notebook. Use `--include-support` if you also want the optional CLI scripts, handoff and regenerated per-pair metadata. Use `--check-only` to validate inputs and inspect the file list **without creating a ZIP**. Rename to the actual team name. The assignment's stated deadline is 11:59 pm Singapore time on 6 November 2026.

## Sources

Sources: supplied SC3000/CZ3005 assignment; [NanoGPT-Math](https://github.com/zhouyuan888888/NanoGPT-Math); [NanoGPT](https://github.com/karpathy/nanoGPT); [Rafailov et al., DPO](https://arxiv.org/abs/2305.18290). PyTorch, NumPy, Matplotlib and Python Fraction arithmetic support the implementation. Team members should review and understand the submitted work.
