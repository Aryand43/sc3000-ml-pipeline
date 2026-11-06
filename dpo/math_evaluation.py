"""Reusable greedy math generation and exact scoring for Tasks 2 and 3.

Importing this module does not load checkpoints, generate answers or train models.
The arithmetic oracle is used only after generation, never to supply model tokens.
"""
import random
import re
from collections import defaultdict
from fractions import Fraction

ANSWER_PATTERN = r'^The answer is (-?\d+)(?= because |\.$|$)'
EXPLANATION_PATTERN = r'The answer is (-?\d+) because (.+)=(-?\d+)\.'
POSITIVE_ANSWER_PATTERN = r'The answer is (-?\d+) because '
BINARY_EXPRESSION_PATTERN = r'(-?\d+)([+*/-])(-?\d+)'
UNKNOWN_ON_LEFT_PATTERN = r'x([+*/-])(\d+)'
UNKNOWN_ON_RIGHT_PATTERN = r'(\d+)([-*])x'

INVERSE_OPERATOR = {'+': '-', '-': '+', '*': '/', '/': '*'}
OPERATOR_CATEGORY = {'+': 'addition', '-': 'subtraction',
                     '*': 'multiplication', '/': 'division'}
CATEGORY_ORDER = ['addition', 'subtraction', 'multiplication', 'division', 'algebra']


def canonical_question(question):
    return re.sub(r'\s+', '', question)


def arithmetic_value(expression):
    match = re.fullmatch(BINARY_EXPRESSION_PATTERN, expression)
    if match is None:
        raise ValueError(f'Unsupported arithmetic: {expression!r}')
    left, operator, right = Fraction(int(match[1])), match[2], Fraction(int(match[3]))
    if operator == '+':
        return left + right
    if operator == '-':
        return left - right
    if operator == '*':
        return left * right
    if not right:
        raise ValueError('Division by zero')
    return left / right


def solve_equation(lhs, rhs):
    match = re.fullmatch(UNKNOWN_ON_LEFT_PATTERN, lhs)
    if match:
        operator, operand = match[1], int(match[2])
        if operator == '+':
            return rhs - operand
        if operator == '-':
            return rhs + operand
        if operator == '*':
            return rhs / operand
        return rhs * operand
    match = re.fullmatch(UNKNOWN_ON_RIGHT_PATTERN, lhs)
    if not match:
        raise ValueError(f'Unsupported equation: {lhs!r}')
    operand = int(match[1])
    return operand - rhs if match[2] == '-' else rhs / operand


def solve_question(question):
    canonical = canonical_question(question)
    if canonical.endswith(',x=?'):
        lhs, rhs = canonical[:-4].split('=')
        return solve_equation(lhs, Fraction(int(rhs)))
    if not canonical.endswith('=?'):
        raise ValueError(canonical)
    return arithmetic_value(canonical[:-2])


def parse_answer(response):
    """Anchored parse so an explanation's divisor or operand is never read as the answer."""
    match = re.match(ANSWER_PATTERN, response)
    return int(match[1]) if match else None


def expected_explanation_expression(question):
    """The one equation that counts as showing this question's own work."""
    canonical = canonical_question(question)
    if not canonical.endswith(',x=?'):
        return canonical[:-2]
    lhs, rhs = canonical[:-4].split('=')
    match = re.fullmatch(UNKNOWN_ON_LEFT_PATTERN, lhs)
    if match:
        return f'{rhs}{INVERSE_OPERATOR[match[1]]}{match[2]}'
    match = re.fullmatch(UNKNOWN_ON_RIGHT_PATTERN, lhs)
    return f'{match[1]}-{rhs}' if match[2] == '-' else f'{rhs}/{match[1]}'


def classify_explanation(question, expected_answer, response):
    """Conservative: the equality must perform this question's operation, not merely be true."""
    match = re.fullmatch(EXPLANATION_PATTERN, response)
    if not match:
        return 'unparsed'
    try:
        equation_value = arithmetic_value(match[2])
        relevant_expression = expected_explanation_expression(question)
        verified = (int(match[1]) == expected_answer == equation_value == int(match[3])
                    and match[2].replace(' ', '') == relevant_expression)
    except (ValueError, ZeroDivisionError, TypeError):
        return 'incorrect_or_irrelevant'
    return 'verified' if verified else 'incorrect_or_irrelevant'


def score_response(record):
    response = record['response'].strip()
    parsed = parse_answer(response)
    correct = parsed == record['expected_answer'] if parsed is not None else False
    explanation_status = classify_explanation(record['question'], record['expected_answer'],
                                              response)
    return dict(record, parsed_answer=parsed, answer_correct=correct,
                explanation_status=explanation_status,
                fully_verified=(correct and explanation_status == 'verified'
                                and not record['truncated']))


def category_metrics(rows):
    count = len(rows)
    return dict(count=count,
                answer_correct=sum(r['answer_correct'] for r in rows),
                answer_accuracy=sum(r['answer_correct'] for r in rows) / count,
                fully_verified=sum(r['fully_verified'] for r in rows),
                fully_verified_accuracy=sum(r['fully_verified'] for r in rows) / count,
                answer_unparsed=sum(r['parsed_answer'] is None for r in rows),
                explanation_unparsed=sum(r['explanation_status'] == 'unparsed' for r in rows),
                truncated=sum(r['truncated'] for r in rows))


def summarize_results(results):
    categories = sorted({r['category'] for r in results})
    return dict(overall=category_metrics(results),
                by_category={category: category_metrics(
                    [r for r in results if r['category'] == category])
                    for category in categories})


def question_category(question):
    canonical = canonical_question(question)
    if 'x' in canonical:
        return 'algebra'
    match = re.fullmatch(r'-?\d+([+*/-])-?\d+=\?', canonical)
    if not match:
        raise ValueError(f'Unsupported question: {question!r}')
    return OPERATOR_CATEGORY[match[1]]


def question_from_pair(pair):
    if set(pair) != {'positive', 'negative'}:
        raise ValueError('Expected positive/negative pair schema')
    question = pair['positive'].split('?', 1)[0] + '?'
    if not pair['negative'].startswith(question + ' '):
        raise ValueError('Pair does not share a question')
    return question


def development_record(pair):
    """Build one scored development record, rejecting any pair the oracle disagrees with."""
    question = question_from_pair(pair)
    answer = solve_question(question)
    if answer.denominator != 1:
        raise ValueError('Expected integer math solutions')
    stated = re.match(POSITIVE_ANSWER_PATTERN, pair['positive'][len(question) + 1:])
    if not stated or int(stated[1]) != answer:
        raise ValueError('Positive answer disagrees with exact oracle')
    return dict(question=question, expected_answer=int(answer),
                category=question_category(question), split='development')


def select_development_questions(pairs, per_category=20, seed=3003):
    """Select a fixed balanced subset from DEVELOPMENT pairs, never training pairs.

    Call once before training, not with a different seed every epoch. Reject
    malformed pairs or positive labels inconsistent with the exact oracle.
    """
    if per_category < 1:
        raise ValueError('per_category must be positive')
    groups = defaultdict(list)
    seen = set()
    for pair in pairs:
        record = development_record(pair)
        key = canonical_question(record['question'])
        if key in seen:
            continue
        seen.add(key)
        groups[record['category']].append(record)
    rng = random.Random(seed)
    selected = []
    for category in CATEGORY_ORDER:
        choices = sorted(groups[category], key=lambda r: r['question'])
        if len(choices) < per_category:
            raise ValueError(f'Need {per_category} development questions for {category}; '
                             f'found {len(choices)}')
        selected.extend(rng.sample(choices, per_category))
    rng.shuffle(selected)
    return selected


def prepare_generation_records(questions, stoi, max_new_tokens, block_size):
    """Validate every question against the oracle and group their indices by prompt length."""
    records = [dict(q) for q in questions]
    groups = defaultdict(list)
    for index, record in enumerate(records):
        expected = solve_question(record['question'])
        if expected.denominator != 1 or int(expected) != record['expected_answer']:
            raise ValueError('Question/expected answer mismatch')
        prompt = record['question'] + ' '
        unsupported = set(prompt) - set(stoi)
        if unsupported:
            raise ValueError(f'Unsupported prompt characters: {unsupported}')
        if len(prompt) + max_new_tokens > block_size:
            raise ValueError('Generation budget exceeds context; refusing to crop')
        groups[len(prompt)].append(index)
    if len({canonical_question(r['question']) for r in records}) != len(records):
        raise ValueError('Duplicate evaluation questions')
    return records, groups


def generate_responses(model, prompts, stoi, itos, device, max_new_tokens):
    """Greedy decode one equal-length batch, stopping each row on newline or period."""
    import torch
    ids = torch.tensor([[stoi[c] for c in p] for p in prompts], dtype=torch.long, device=device)
    active = [True] * len(prompts)
    continuations = [''] * len(prompts)
    reasons = ['max_new_tokens'] * len(prompts)
    for _ in range(max_new_tokens):
        logits, _ = model(ids)
        next_ids = logits[:, -1, :].argmax(dim=-1)
        for row, token in enumerate(next_ids.tolist()):
            if not active[row]:
                continue
            char = itos[token]
            if char == '\n':
                active[row], reasons[row] = False, 'newline'
            else:
                continuations[row] += char
                if char == '.':
                    active[row], reasons[row] = False, 'period'
        if not any(active):
            break
        ids = torch.cat((ids, next_ids[:, None]), dim=1)
    return continuations, reasons


def evaluate_generated_math(model, questions, stoi, itos, device=None,
                            batch_size=16, max_new_tokens=64, progress=False):
    """Generate greedy responses and return {'summary': ..., 'results': ...}.

    questions: records with question, expected_answer, category. Generation sees
    ONLY question strings. Equal-length batching needs no padding. Matches the
    supplied model's period/newline stopping. Preserves the model's train/eval
    mode and does not change gradients, parameters, checkpoints, or RNG seeds.
    Unparsed and wrong answers count as incorrect; truncation is reported.
    """
    import torch
    if not questions or batch_size < 1 or max_new_tokens < 1:
        raise ValueError('Nonempty questions and positive generation settings required')
    device = device or next(model.parameters()).device
    records, groups = prepare_generation_records(questions, stoi, max_new_tokens,
                                                 model.config.block_size)
    results = [None] * len(records)
    was_training = model.training
    model.eval()
    done = 0
    try:
        with torch.inference_mode():
            for _, group in sorted(groups.items()):
                for start in range(0, len(group), batch_size):
                    chunk = group[start:start + batch_size]
                    prompts = [records[i]['question'] + ' ' for i in chunk]
                    continuations, reasons = generate_responses(model, prompts, stoi, itos,
                                                                device, max_new_tokens)
                    for index, text, reason in zip(chunk, continuations, reasons):
                        raw = dict(records[index], response=text, stop_reason=reason,
                                   truncated=reason == 'max_new_tokens')
                        results[index] = score_response(raw)
                    done += len(chunk)
                    if progress:
                        print(f'Generated {done}/{len(records)} answers', flush=True)
    finally:
        model.train(was_training)
    return dict(summary=summarize_results(results), results=results)
