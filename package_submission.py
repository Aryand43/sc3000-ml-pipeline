"""Create a checked submission ZIP from this project root, including real weights."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent
FILES = [
    'README.md','requirements.txt','VALIDATION_ENVIRONMENT.json','package_submission.py','model.py','configurator.py',
    'sft/gpt.pt','sft/meta.pkl','dpo/dpo.ipynb','dpo/dpo.pt',
    'dpo/pos_neg_pairs.json','dpo/task1_dev_pairs.json','dpo/task1_metadata.json','dpo/task1_validation.json',
    'dpo/train_dpo.py','dpo/math_evaluation.py','dpo/evaluate_task3.py','dpo/TASK3_HANDOFF.md',
    'dpo/task2_history.json','dpo/task2_sft_history.json',
    'dpo/task2_training_curves.png','dpo/task2_generated_accuracy.png','dpo/task2_sft_generated_accuracy.png',
    'dpo/task3_fresh_test_questions.json','dpo/task3_previous_test_questions.json',
    'dpo/task3_final_results.json','dpo/task3_final_summary.json',
]


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'SC3000_Math_Submission.zip')
    args=parser.parse_args()
    for name in FILES:
        if not (ROOT/name).is_file():raise SystemExit(f'Missing {name}; execute dpo/dpo.ipynb first.')
    for name in ['sft/gpt.pt','dpo/dpo.pt']:
        with (ROOT/name).open('rb') as f:header=f.read(80)
        if header.startswith(b'version https://git-lfs.github.com/spec/'):
            raise SystemExit(f'{name} is only an LFS pointer. Download actual weights before packaging.')
        if not header.startswith(b'PK'):raise SystemExit(f'Unexpected checkpoint format: {name}')
    notebook=json.loads((ROOT/'dpo/dpo.ipynb').read_text())
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            if cell.get('execution_count') is None:raise SystemExit('Notebook has unexecuted cells; run and save it.')
            if any(o.get('output_type')=='error' for o in cell.get('outputs',[])):
                raise SystemExit('Notebook contains execution errors.')
    pairs=json.loads((ROOT/'dpo/pos_neg_pairs.json').read_text())
    if len(pairs)!=100000 or any(set(p)!={'positive','negative'} for p in pairs):
        raise SystemExit('Preference count/schema mismatch.')
    final=json.loads((ROOT/'dpo/task3_final_summary.json').read_text())
    if sha(ROOT/'dpo/dpo.pt')!=final['provenance']['checkpoint_sha256']:
        raise SystemExit('Saved Task 3 results do not belong to dpo/dpo.pt. Evaluate the submitted checkpoint first.')
    manifest=dict(files={name:dict(sha256=sha(ROOT/name),bytes=(ROOT/name).stat().st_size) for name in FILES},
        notes='Recorded long-run training histories; fresh Task 1 and Task 3 notebook execution. '
              'No experimental runs, caches, alternate checkpoints or LFS pointers are included.')
    temporary=args.out.with_suffix(args.out.suffix+'.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as z:
        for name in FILES:z.write(ROOT/name,name)
        z.writestr('SUBMISSION_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(temporary) as z:
        if z.testzip() is not None:raise SystemExit('ZIP CRC validation failed.')
        stored=json.loads(z.read('SUBMISSION_MANIFEST.json'))
        for name,item in stored['files'].items():
            if hashlib.sha256(z.read(name)).hexdigest()!=item['sha256']:
                raise SystemExit(f'ZIP checksum mismatch: {name}')
    temporary.replace(args.out)
    print(f'Created and verified {args.out} ({args.out.stat().st_size:,} bytes).')
    print('Rename the ZIP to your actual team name before submitting.')


if __name__=='__main__':main()
