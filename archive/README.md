# Archive — not used for final results

These files are preserved for history only. They are not loaded by the final notebook, selected by default evaluation, or included in the submission ZIP. Start at [`../dpo/dpo.ipynb`](../dpo/dpo.ipynb); the final checkpoint is [`../dpo/dpo.pt`](../dpo/dpo.pt).

| File | Why archived | Current source of truth |
|---|---|---|
| `checkpoints/dpo_best.pt` | Old epoch-1 preference-selected checkpoint, predating the final retraining. Its filename does not make it the best current model. | `dpo/dpo.pt` |
| `reports/task3_generation_results.json` | Earlier development-only evaluation of the retrained model. Its 80% score is not an independent fresh-test score. | `dpo/task3_final_summary.json` and `dpo/task3_final_results.json` distinguish all evaluation sets |
| `reports/task2_generation_failures.json` | Detailed per-epoch development failure log; useful diagnostics but not needed to run the notebook. | Final predictions in `dpo/task3_final_results.json`; training curves use the retained `dpo/task2_history.json` |

Paths in the last column are relative to the repository root. Only file locations changed: archived contents and the final model's weights remain unchanged. New experimental runs may produce similarly named logs inside their own output directories; those are separate from these historical copies.
