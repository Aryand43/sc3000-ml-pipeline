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
python dpo/train_dpo.py --objective sft --lr 1e-3 --epochs 20 --out runs/math_sft/sft.pt
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

| arm | best |
| --- | --- |
| original `sft/gpt.pt` | 0% |
| direct DPO | 17% |
| direct DPO, `--lambda-sft 1.0` | 24% |
| math-SFT, 5 epochs | 70% |
| math-SFT, 20 epochs | **81%** |
| math-SFT 20 epochs then DPO | 80% |

`dpo/dpo.pt` is the last of these. By category: addition 20/20,
subtraction 19/20, division 17/20, algebra 15/20, multiplication 9/20.
Every miss is wrong arithmetic, nothing unparsed or truncated.

Training length was the lever. At 5 epochs the model scored 71% on questions it
had trained on and 73% on unseen ones, so it was underfitting rather than
memorising. Twenty epochs moves that to 91% train and 80% dev, which closes the
gap and exhausts the lever; more epochs now overfit.

Multiplication did not move at all between the two, at 9/20 in both. Every
other category improved. The model emits the answer most significant digit
first while carries propagate the other way, so it has to commit to the leading
digit before doing the carry work. 22 of the 27 misses at 73% were off by an
exact multiple of ten. Fixing this needs a scratchpad or reversed digits in the
response, which the scoring format does not currently allow.

DPO is roughly neutral here. It takes 81% to 80%, within noise on 100
questions. Its real contribution is earlier: the original SFT checkpoint
refuses every question, and any math fine-tuning removes that.

Preference accuracy saturates at 1.000 in epoch 1 while generation is still
bad, which is why checkpoints are selected on answer accuracy instead.
