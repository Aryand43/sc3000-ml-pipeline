"""Run generation accuracy from the project root: python dpo/evaluate_task3.py.

Defaults to a DEVELOPMENT evaluation of dpo.pt. Frozen diagnostic questions can
be passed with --questions; use a genuinely untouched set for final reporting.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
import torch
from math_evaluation import evaluate_generated_math, select_development_questions


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=root/'dpo/dpo.pt')
    parser.add_argument('--dev-pairs',type=Path,default=root/'dpo/task1_dev_pairs.json')
    parser.add_argument('--questions',type=Path,help='JSON list of frozen question/expected_answer/category records')
    parser.add_argument('--out',type=Path,default=root/'dpo/task3_generation_results.json')
    parser.add_argument('--per-category',type=int,default=20)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--max-new-tokens',type=int,default=64)
    parser.add_argument('--seed',type=int,default=3003)
    parser.add_argument('--device',default=None)
    args = parser.parse_args()
    sys.path.insert(0,str(root))
    from model import GPT,GPTConfig
    with (root/'sft/meta.pkl').open('rb') as f:
        meta = pickle.load(f)
    with args.checkpoint.open('rb') as f:
        if f.read(80).startswith(b'version https://git-lfs.github.com/spec/'):
            raise RuntimeError('Checkpoint is an LFS pointer; retrieve its actual weights')
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    device = args.device or ('cuda' if torch.cuda.is_available() else
        'mps' if getattr(torch.backends,'mps',None) and torch.backends.mps.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model = GPT(GPTConfig(**checkpoint['model_args']))
    state = checkpoint.get('model_state_dict',checkpoint.get('model'))
    if state is None:
        raise ValueError('Missing model_state_dict/model in checkpoint')
    model.load_state_dict({k.removeprefix('_orig_mod.'):v for k,v in state.items()},strict=True)
    assert model.config.vocab_size == len(meta['stoi'])
    model.to(device).eval()
    if args.questions:
        questions = json.loads(args.questions.read_text())
        source = str(args.questions)
    else:
        pairs = json.loads(args.dev_pairs.read_text())
        questions = select_development_questions(pairs,args.per_category,args.seed)
        source = str(args.dev_pairs)
    output = evaluate_generated_math(model,questions,meta['stoi'],meta['itos'],device,
        args.batch_size,args.max_new_tokens,progress=True)
    output['provenance'] = dict(checkpoint=str(args.checkpoint),epoch=checkpoint.get('epoch'),
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        question_source=source,seed=args.seed,decoding='greedy',device=device,
        torch_version=torch.__version__,batch_size=args.batch_size,max_new_tokens=args.max_new_tokens)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix+'.tmp')
    temporary.write_text(json.dumps(output,indent=2)+'\n')
    temporary.replace(args.out)
    saved = json.loads(args.out.read_text())
    assert saved['summary']==output['summary']
    print(json.dumps(output['summary'],indent=2))
    for record in output['results']:
        print(f"{'PASS' if record['answer_correct'] else 'FAIL'} {record['question']} "
            f"{record['response']} | expected={record['expected_answer']}")
    print('Saved',args.out)


if __name__=='__main__':
    main()
