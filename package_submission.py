"""Build a focused submission ZIP; --check-only validates without creating one."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REQUIRED_FILES = [
    'README.md', 'requirements.txt', 'VALIDATION_ENVIRONMENT.json', 'package_submission.py',
    'model.py', 'configurator.py', 'sft/gpt.pt', 'sft/meta.pkl',
    'dpo/dpo.ipynb', 'dpo/dpo.pt', 'dpo/pos_neg_pairs.json', 'dpo/task1_dev_pairs.json',
    'dpo/task2_history.json', 'dpo/task2_sft_history.json',
    'dpo/task3_fresh_test_questions.json', 'dpo/task3_previous_test_questions.json',
]
EVIDENCE_FILES = [
    'dpo/task1_validation.json', 'dpo/task2_training_curves.png',
    'dpo/task2_generated_accuracy.png', 'dpo/task2_sft_generated_accuracy.png',
    'dpo/task3_final_results.json', 'dpo/task3_final_summary.json',
]
SUPPORT_FILES = [
    'dpo/train_dpo.py', 'dpo/math_evaluation.py', 'dpo/evaluate_task3.py',
    'dpo/task1_metadata.json',
]

CHECKPOINT_FILES = ['sft/gpt.pt', 'dpo/dpo.pt']
EXPECTED_PREFERENCE_COUNT = 100000
LFS_POINTER_PREFIX = b'version https://git-lfs.github.com/spec/'
ZIP_MAGIC = b'PK'
MANIFEST_NAME = 'SUBMISSION_MANIFEST.json'
MANIFEST_NOTES = ('Recorded long-run training histories; executed notebook with saved outputs. '
                  'No archive/, experiments, caches, alternate checkpoints or LFS pointers '
                  'included.')


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def require_files_exist(root, files):
    for name in files:
        if not (root / name).is_file():
            raise SystemExit(f'Missing {name}; execute dpo/dpo.ipynb first. '
                             'Per-pair metadata is only needed with --include-support.')


def require_real_checkpoints(root):
    for name in CHECKPOINT_FILES:
        with (root / name).open('rb') as f:
            header = f.read(80)
        if header.startswith(LFS_POINTER_PREFIX):
            raise SystemExit(f'{name} is only an LFS pointer. '
                             'Obtain actual weights before packaging.')
        if not header.startswith(ZIP_MAGIC):
            raise SystemExit(f'Unexpected checkpoint format: {name}')


def require_executed_notebook(root):
    notebook = read_json(root / 'dpo/dpo.ipynb')
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        if cell.get('execution_count') is None:
            raise SystemExit('Notebook has unexecuted cells; run and save it.')
        if any(output.get('output_type') == 'error' for output in cell.get('outputs', [])):
            raise SystemExit('Notebook contains execution errors.')


def require_preference_dataset(root):
    pairs = read_json(root / 'dpo/pos_neg_pairs.json')
    wrong_count = len(pairs) != EXPECTED_PREFERENCE_COUNT
    wrong_schema = any(set(pair) != {'positive', 'negative'} for pair in pairs)
    if wrong_count or wrong_schema:
        raise SystemExit('Preference count/schema mismatch.')


def require_results_match_checkpoint(root):
    final = read_json(root / 'dpo/task3_final_summary.json')
    if file_sha256(root / 'dpo/dpo.pt') != final['provenance']['checkpoint_sha256']:
        raise SystemExit('Saved results do not belong to dpo/dpo.pt. '
                         'Evaluate the submitted checkpoint first.')


def validate_inputs(root, files):
    require_files_exist(root, files)
    require_real_checkpoints(root)
    require_executed_notebook(root)
    require_preference_dataset(root)
    require_results_match_checkpoint(root)


def select_files(include_support):
    return REQUIRED_FILES + EVIDENCE_FILES + (SUPPORT_FILES if include_support else [])


def build_manifest(root, files, include_support):
    return dict(
        files={name: dict(sha256=file_sha256(root / name), bytes=(root / name).stat().st_size)
               for name in files},
        notes=MANIFEST_NOTES,
        optional_support_included=include_support,
    )


def write_archive(destination, root, files, manifest):
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            archive.write(root / name, name)
        archive.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2) + '\n')


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise SystemExit('ZIP CRC validation failed.')
        stored = json.loads(archive.read(MANIFEST_NAME))
        for name, item in stored['files'].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != item['sha256']:
                raise SystemExit(f'ZIP checksum mismatch: {name}')


def report_selection(files):
    print('Validation passed. Selected submission files:')
    for name in files:
        print(name)
    print(f'{len(files)} files; archive/ and experimental runs excluded. No ZIP created.')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'SC3000_Math_Submission.zip')
    parser.add_argument('--include-support', action='store_true',
                        help='Also include optional CLI scripts, handoff and '
                             'regenerated per-pair metadata')
    parser.add_argument('--check-only', action='store_true',
                        help='Validate required inputs and list payload files; '
                             'do not create a ZIP')
    return parser


def main():
    args = build_parser().parse_args()
    files = select_files(args.include_support)
    validate_inputs(ROOT, files)
    if args.check_only:
        report_selection(files)
        return
    manifest = build_manifest(ROOT, files, args.include_support)
    temporary = args.out.with_suffix(args.out.suffix + '.tmp')
    write_archive(temporary, ROOT, files, manifest)
    verify_archive(temporary)
    temporary.replace(args.out)
    print(f'Created and verified {args.out} ({args.out.stat().st_size:,} bytes).')
    print('Rename the ZIP to your actual team name before submitting.')


if __name__ == '__main__':
    main()
