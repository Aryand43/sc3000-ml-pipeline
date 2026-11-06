"""Run generation accuracy from the project root: python dpo/evaluate_task3.py.

Defaults to the frozen held-out benchmark evaluated on 6 November 2026.
Use --development for checkpoint-selection questions; --questions accepts another frozen set.
"""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import torch

from math_evaluation import evaluate_generated_math, select_development_questions

ROOT = Path(__file__).resolve().parent.parent
FROZEN_TEST_QUESTIONS = ROOT / 'dpo/task3_fresh_test_questions.json'
LFS_POINTER_PREFIX = b'version https://git-lfs.github.com/spec/'
MAX_SCORING_THREADS = 4


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'dpo/dpo.pt')
    parser.add_argument('--dev-pairs', type=Path, default=ROOT / 'dpo/task1_dev_pairs.json')
    parser.add_argument('--questions', type=Path,
                        help='JSON list of frozen question/expected_answer/category records')
    parser.add_argument('--development', action='store_true',
                        help='Evaluate the development subset used during training, '
                             'not the frozen test')
    parser.add_argument('--out', type=Path, default=ROOT / 'dpo/task3_cli_results.json')
    parser.add_argument('--per-category', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--seed', type=int, default=3003)
    parser.add_argument('--device', default=None)
    return parser


def pick_device(requested):
    if requested:
        return requested
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def load_tokeniser():
    with (ROOT / 'sft/meta.pkl').open('rb') as f:
        return pickle.load(f)


def reject_lfs_pointer(path):
    with path.open('rb') as f:
        if f.read(80).startswith(LFS_POINTER_PREFIX):
            raise RuntimeError('Checkpoint is an LFS pointer; retrieve its actual weights')


def load_checkpoint_model(checkpoint_path, vocab_size, device):
    from model import GPT, GPTConfig
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = GPT(GPTConfig(**checkpoint['model_args']))
    state = checkpoint.get('model_state_dict', checkpoint.get('model'))
    if state is None:
        raise ValueError('Missing model_state_dict/model in checkpoint')
    model.load_state_dict({k.removeprefix('_orig_mod.'): v for k, v in state.items()}, strict=True)
    assert model.config.vocab_size == vocab_size
    return model.to(device).eval(), checkpoint


def resolve_questions(args, parser):
    """Return the evaluation questions and a human-readable description of their origin."""
    if args.questions and args.development:
        parser.error('--questions and --development are mutually exclusive')
    if args.questions:
        return json.loads(args.questions.read_text()), str(args.questions)
    if args.development:
        pairs = json.loads(args.dev_pairs.read_text())
        questions = select_development_questions(pairs, args.per_category, args.seed)
        return questions, str(args.dev_pairs)
    return json.loads(FROZEN_TEST_QUESTIONS.read_text()), str(FROZEN_TEST_QUESTIONS)


def describe_run(args, checkpoint, source, device):
    return dict(checkpoint=str(args.checkpoint), epoch=checkpoint.get('epoch'),
                checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                question_source=source, development_evaluation=args.development, seed=args.seed,
                decoding='greedy', device=device, torch_version=torch.__version__,
                batch_size=args.batch_size, max_new_tokens=args.max_new_tokens)


def save_results(output, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    temporary.write_text(json.dumps(output, indent=2) + '\n')
    temporary.replace(destination)
    saved = json.loads(destination.read_text())
    assert saved['summary'] == output['summary']


def print_report(output, destination):
    print(json.dumps(output['summary'], indent=2))
    for record in output['results']:
        print(f"{'PASS' if record['answer_correct'] else 'FAIL'} {record['question']} "
              f"{record['response']} | expected={record['expected_answer']}")
    print('Saved', destination)


def main():
    parser = build_parser()
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    meta = load_tokeniser()
    reject_lfs_pointer(args.checkpoint)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(MAX_SCORING_THREADS, torch.get_num_threads()))
    device = pick_device(args.device)
    model, checkpoint = load_checkpoint_model(args.checkpoint, len(meta['stoi']), device)
    questions, source = resolve_questions(args, parser)
    print('Question source:', source)
    print('Development evaluation:', args.development)
    output = evaluate_generated_math(model, questions, meta['stoi'], meta['itos'], device,
                                     args.batch_size, args.max_new_tokens, progress=True)
    output['provenance'] = describe_run(args, checkpoint, source, device)
    save_results(output, args.out)
    print_report(output, args.out)


if __name__ == '__main__':
    main()
