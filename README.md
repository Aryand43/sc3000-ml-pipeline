# sc3000-ml-pipeline

DPO fine-tuning of a character-level NanoGPT to answer arithmetic questions.
NTU SC3000 / CZ3005 group assignment.

## Layout

| path | what |
| --- | --- |
| `sft/gpt.pt`, `sft/meta.pkl` | supplied SFT checkpoint and char tokenizer (vocab 74) |
| `dpo/pos_neg_pairs.json` | Task 1 preference pairs |
| `dpo/task1_dev_pairs.json` | held-out pairs the dev questions are drawn from |
| `dpo/train_dpo.py` | Task 2 trainer (DPO and SFT objectives) |
| `dpo/dpo.ipynb` | Task 2 notebook |
| `dpo/math_evaluation.py` | Task 3 oracle, answer parser, dev question sampler |
| `dpo/evaluate_task3.py` | standalone generation scorer |
| `dpo/dpo.pt` | current best checkpoint |

## Setup

```bash
uv venv .venv && uv pip install --python .venv/bin/python "torch==2.14.1" numpy tqdm matplotlib
```

torch 2.15.0 aborts on any MPS `linear`, so the pin matters on Apple silicon.

## Train

Two stages. Supervised fit on the positive responses first, then DPO from that
checkpoint with both policy and frozen reference initialised from it.

```bash
python dpo/train_dpo.py --objective sft --lr 1e-3 --out runs/math_sft/sft.pt
python dpo/train_dpo.py --checkpoint runs/math_sft/sft_generated_best.pt --lambda-sft 1.0 --out runs/sft_then_dpo/dpo.pt
```

Each run scores generated answers before training and after every epoch, saves
the best checkpoint by that score as `*_generated_best.pt`, and writes
`task2_history.json`, the curves, and `task2_generation_failures.json` with the
prompt, expected answer and raw output of every miss.

## Evaluate

```bash
python dpo/evaluate_task3.py --checkpoint dpo/dpo.pt
```

100 held-out questions, 20 per category, greedy decoding, seed 3003.

## Results

Dev answer accuracy, same questions and decoding for every arm.

| arm | best | mean over 5 epochs |
| --- | --- | --- |
| original `sft/gpt.pt` | 0% | n/a |
| direct DPO | 17% | n/a |
| direct DPO, `--lambda-sft 1.0` | 24% | 14.6% |
| math-SFT only, lr 1e-3 | 70% | 33.8% |
| math-SFT then DPO | **73%** | 67.6% |

The 73% run sits in a 63 to 73 band across epochs, so treat 67.6% as its level.
By category at 73%: addition 20/20, subtraction 16/20, division 15/20,
algebra 13/20, multiplication 9/20. All 27 misses are wrong arithmetic, with
nothing unparsed or truncated.

Learning rate was the lever, not the DPO objective. The same supervised fit at
lr 1e-4 reaches only 17%. `--lambda-sft` has little effect because Adam
normalises per parameter, so the NLL term already dominates once the DPO
gradient saturates.

Preference accuracy saturates at 1.000 in epoch 1 while generation is still
bad, which is why checkpoints are selected on answer accuracy instead.
