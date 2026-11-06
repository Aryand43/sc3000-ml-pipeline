# SC3000 / CZ3005 — Teaching NanoGPT to Do Math

**Start here:** [dpo/dpo.ipynb](dpo/dpo.ipynb) · **Final model:** [dpo/dpo.pt](dpo/dpo.pt)

We trained the supplied NanoGPT on worked math solutions, then applied DPO. The final model answered **79/100 fresh test questions correctly** using greedy decoding, with 20 questions per category.

The notebook contains the dataset generation, training, evaluation and discussion. The supplied `model.py` and `configurator.py` are unchanged.

## Team contributions

| Team member | Contribution |
|---|---|
| Elisetti Atul Dev | Task 1 — preference dataset |
| Aryan Dutt | Task 2 — training and monitoring |
| Harsh Kamdar | Task 3 — evaluation and analysis |

## Files

| Group | Files |
|---|---|
| Main submission | `dpo/dpo.ipynb`, `dpo/dpo.pt`, `dpo/pos_neg_pairs.json` |
| Required assets and data | `model.py`, `configurator.py`, `sft/`, development pairs, both training-history JSON files and both frozen test-question JSON files in `dpo/` |
| Results | `dpo/task1_validation.json`, `dpo/task3_final_summary.json`, `dpo/task3_final_results.json`, training plots |
| Optional tools | `dpo/train_dpo.py`, `dpo/evaluate_task3.py`, `dpo/math_evaluation.py`, `package_submission.py` |

`archive/` contains superseded checkpoints and reports, excluded from the submission package.

## Run the notebook

Use Python 3.12. Pinned dependencies are in `requirements.txt`; the tested environment is recorded in `VALIDATION_ENVIRONMENT.json`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user --name sc3000-math --display-name "SC3000 Math"
```

Open `dpo/dpo.ipynb`, select that kernel and run all cells. The working directory can be the project root or `dpo/`.

**Default execution** evaluates the supplied final model, regenerates and validates the dataset, runs small training checks and displays recorded training curves. It does not repeat the full training run or overwrite the final model. CPU is the default evaluation device.

**Full reproduction:** set `RUN_FULL_TRAINING=True` to run 20 supervised warm-up epochs followed by 5 DPO epochs. New checkpoints are written to `runs/notebook_reproduction/`. Hyperparameters and training stages are explained in the notebook.

The pretrained checkpoint uses Git LFS. After cloning, run `git lfs pull`. GitHub ZIP downloads may contain a pointer instead of the weights; the notebook requires the actual checkpoint.

## Dataset and results

The training dataset has 100,000 pairs, balanced across addition, subtraction, multiplication, exact division and algebra, plus 500 separate development pairs. It contains **100,000 distinct question strings and 71,547 whitespace-normalised questions**. Spacing variants account for the difference; these are not additional mathematical problems. Positives use exact arithmetic; submitted negatives are synthetic.

| Evaluation | Correct |
|---|---:|
| Fresh test, separate from training and development | **79/100** |
| Development subset used during training | **80/100** |

Multiplication was weakest at 8/20. The combined pipeline works, but best SFT development accuracy was 81% versus 80% after DPO: this experiment does not demonstrate an additional accuracy gain from DPO. Full predictions, category scores and failure analysis are in the notebook and results JSON files. The fresh-test score does not guarantee the examiner's private-test result.

## Package the submission

After running and saving the notebook:

```bash
python package_submission.py --out YOUR_TEAM_NAME.zip
```

The packager checks execution, required files, checkpoints and result consistency. Use `--check-only` to validate without creating a ZIP, or `--include-support` to include optional tools and regenerated metadata.

## References

Supplied SC3000/CZ3005 assignment; [NanoGPT-Math](https://github.com/zhouyuan888888/NanoGPT-Math); [NanoGPT](https://github.com/karpathy/nanoGPT); [Rafailov et al., DPO](https://arxiv.org/abs/2305.18290).
