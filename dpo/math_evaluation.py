"""Reusable greedy math generation and exact scoring for Tasks 2 and 3.

Importing this module does not load checkpoints, generate answers or train models.
The arithmetic oracle is used only after generation, never to supply model tokens.
"""
from collections import Counter, defaultdict
from fractions import Fraction
import random
import re

def canonical_question(question):
    return re.sub(r'\s+', '', question)

def arithmetic_value(expr):
    match = re.fullmatch(r'(-?\d+)([+*/-])(-?\d+)', expr)
    if match is None: raise ValueError(f'Unsupported arithmetic: {expr!r}')
    a, op, b = Fraction(int(match[1])), match[2], Fraction(int(match[3]))
    if op == '+': return a+b
    if op == '-': return a-b
    if op == '*': return a*b
    if not b: raise ValueError('Division by zero')
    return a/b

def solve_question(question):
    q = canonical_question(question)
    if q.endswith(',x=?'):
        lhs, rhs = q[:-4].split('=')
        rhs = Fraction(int(rhs))
        m = re.fullmatch(r'x([+*/-])(\d+)', lhs)
        if m:
            op, a = m[1], int(m[2])
            if op == '+': return rhs-a
            if op == '-': return rhs+a
            if op == '*': return rhs/a
            return rhs*a
        m = re.fullmatch(r'(\d+)([-*])x', lhs)
        if not m: raise ValueError(f'Unsupported equation: {q!r}')
        a = int(m[1])
        return a-rhs if m[2]=='-' else rhs/a
    if not q.endswith('=?'): raise ValueError(q)
    return arithmetic_value(q[:-2])

def score_response(record):
    # Strict, anchored answer parser: never grab the last number from an
    # explanation (it might be a divisor, operand, or a contradictory answer).
    response = record['response'].strip()
    match = re.match(r'^The answer is (-?\d+)(?= because |\.$|$)', response)
    parsed = int(match[1]) if match else None
    correct = parsed == record['expected_answer'] if parsed is not None else False
    # A deliberately conservative explanation check: equality must perform the
    # actual operation/inverse operation for this question, not just any true sum.
    explanation = re.fullmatch(r'The answer is (-?\d+) because (.+)=(-?\d+)\.', response)
    explanation_status = 'unparsed'
    if explanation:
        try:
            equation_answer = arithmetic_value(explanation[2])
            q = canonical_question(record['question'])
            if q.endswith(',x=?'):
                lhs,rhs = q[:-4].split('=')
                m = re.fullmatch(r'x([+*/-])(\d+)', lhs)
                if m:
                    op,a = m[1],m[2]
                    inverse = {'+':'-', '-':'+', '*':'/', '/':'*'}[op]
                    relevant_expr = f'{rhs}{inverse}{a}'
                else:
                    m = re.fullmatch(r'(\d+)([-*])x',lhs)
                    relevant_expr = f'{m[1]}-{rhs}' if m[2]=='-' else f'{rhs}/{m[1]}'
            else: relevant_expr = q[:-2]
            valid = (int(explanation[1]) == record['expected_answer'] ==
                equation_answer == int(explanation[3]) and
                explanation[2].replace(' ','') == relevant_expr)
            explanation_status = 'verified' if valid else 'incorrect_or_irrelevant'
        except (ValueError,ZeroDivisionError,TypeError):
            explanation_status = 'incorrect_or_irrelevant'
    return dict(record, parsed_answer=parsed, answer_correct=correct,
        explanation_status=explanation_status,
        fully_verified=correct and explanation_status=='verified' and not record['truncated'])

def summarize_results(results):
    def metrics(rows):
        n=len(rows)
        return dict(count=n, answer_correct=sum(r['answer_correct'] for r in rows),
            answer_accuracy=sum(r['answer_correct'] for r in rows)/n,
            fully_verified=sum(r['fully_verified'] for r in rows),
            fully_verified_accuracy=sum(r['fully_verified'] for r in rows)/n,
            answer_unparsed=sum(r['parsed_answer'] is None for r in rows),
            explanation_unparsed=sum(r['explanation_status']=='unparsed' for r in rows),
            truncated=sum(r['truncated'] for r in rows))
    return dict(overall=metrics(results), by_category={c:metrics([r for r in results if r['category']==c])
        for c in sorted({r['category'] for r in results})})


def question_category(question):
    q = canonical_question(question)
    if 'x' in q:
        return 'algebra'
    m = re.fullmatch(r'-?\d+([+*/-])-?\d+=\?', q)
    if not m:
        raise ValueError(f'Unsupported question: {question!r}')
    return {'+':'addition','-':'subtraction','*':'multiplication','/':'division'}[m[1]]


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
        if set(pair) != {'positive', 'negative'}:
            raise ValueError('Expected positive/negative pair schema')
        question = pair['positive'].split('?',1)[0]+'?'
        if not pair['negative'].startswith(question+' '):
            raise ValueError('Pair does not share a question')
        key = canonical_question(question)
        answer = solve_question(question)
        if answer.denominator != 1:
            raise ValueError('Expected integer math solutions')
        positive_answer = re.match(r'The answer is (-?\d+) because ',pair['positive'][len(question)+1:])
        if not positive_answer or int(positive_answer[1]) != answer:
            raise ValueError('Positive answer disagrees with exact oracle')
        if key in seen:
            continue
        seen.add(key)
        category = question_category(question)
        groups[category].append(dict(question=question,expected_answer=int(answer),
            category=category,split='development'))
    rng = random.Random(seed)
    selected = []
    for category in ['addition','subtraction','multiplication','division','algebra']:
        choices = sorted(groups[category],key=lambda r:r['question'])
        if len(choices) < per_category:
            raise ValueError(f'Need {per_category} development questions for {category}; found {len(choices)}')
        selected.extend(rng.sample(choices,per_category))
    rng.shuffle(selected)
    return selected


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
    groups = defaultdict(list)
    records = [dict(q) for q in questions]
    for i,record in enumerate(records):
        expected = solve_question(record['question'])
        if expected.denominator != 1 or int(expected) != record['expected_answer']:
            raise ValueError('Question/expected answer mismatch')
        prompt = record['question']+' '
        unsupported = set(prompt)-set(stoi)
        if unsupported:
            raise ValueError(f'Unsupported prompt characters: {unsupported}')
        if len(prompt)+max_new_tokens > model.config.block_size:
            raise ValueError('Generation budget exceeds context; refusing to crop')
        groups[len(prompt)].append(i)
    if len({canonical_question(q['question']) for q in records}) != len(records):
        raise ValueError('Duplicate evaluation questions')
    results = [None]*len(records)
    was_training = model.training
    model.eval()
    done = 0
    try:
        with torch.inference_mode():
            for _,group in sorted(groups.items()):
                for start in range(0,len(group),batch_size):
                    chunk = group[start:start+batch_size]
                    prompts = [records[i]['question']+' ' for i in chunk]
                    ids = torch.tensor([[stoi[c] for c in p] for p in prompts],dtype=torch.long,device=device)
                    active = [True]*len(chunk)
                    continuations = ['']*len(chunk)
                    reasons = ['max_new_tokens']*len(chunk)
                    for _ in range(max_new_tokens):
                        logits,_ = model(ids)
                        next_ids = logits[:,-1,:].argmax(dim=-1)
                        for j,t in enumerate(next_ids.tolist()):
                            if not active[j]:
                                continue
                            char = itos[t]
                            if char=='\n':
                                active[j],reasons[j] = False,'newline'
                            else:
                                continuations[j] += char
                                if char=='.':
                                    active[j],reasons[j] = False,'period'
                        if not any(active):
                            break
                        ids = torch.cat((ids,next_ids[:,None]),dim=1)
                    for i,text,reason in zip(chunk,continuations,reasons):
                        raw = dict(records[i],response=text,stop_reason=reason,
                                   truncated=reason=='max_new_tokens')
                        results[i] = score_response(raw)
                    done += len(chunk)
                    if progress:
                        print(f'Generated {done}/{len(records)} answers',flush=True)
    finally:
        model.train(was_training)
    return dict(summary=summarize_results(results),results=results)
