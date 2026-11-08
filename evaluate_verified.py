"""Fresh challenge-set evaluation for quote-verified local RAG.

Run --validate-only first (no models). Requires local vLLM on port 8000
for a full run. This is an internal synthetic-document evaluation, not a
statistically independent external benchmark. Do not tune thresholds on it.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES_PATH = ROOT / 'data' / 'verified_challenge_cases.json'
OUTPUT = ROOT / 'eval_results'


def load_validate() -> list[dict]:
    from evaluate import load_chunks, TESTS
    from pypdf import PdfReader
    from evidence_guard import informative_terms

    payload = json.loads(CASES_PATH.read_text(encoding='utf-8'))
    cases = payload['cases']
    chunks = load_chunks()
    pages = [' '.join((p.extract_text() or '').split()) for p in PdfReader(str(ROOT / 'data' / 'enterprise_knowledge_base_demo.pdf')).pages]
    old_questions = {q['question'].strip().casefold() for q in TESTS}
    from evaluate_refusal import NEGATIVES
    old_questions |= {q.strip().casefold() for _, q in NEGATIVES}
    if len(cases) != 24 or len({q['id'] for q in cases}) != 24:
        raise ValueError('Expected exactly 24 uniquely labelled challenge cases.')
    if sum(q['expected'] == 'answer' for q in cases) != 12 or sum(q['expected'] == 'abstain' for q in cases) != 12:
        raise ValueError('Expected 12 positive and 12 negative examples.')
    for case in cases:
        if case['question'].strip().casefold() in old_questions:
            raise ValueError(f"Previously seen question duplicated: {case['id']}")
        if case['expected'] == 'answer':
            page = case['page']
            anchor = case['anchor']
            if not 1 <= page <= len(pages) or anchor.casefold() not in pages[page - 1].casefold():
                raise ValueError(f"Gold phrase not found on PDF page for {case['id']}: {anchor}")
            if not any(chunk['page'] == page and anchor.casefold() in chunk['text'].casefold() for chunk in chunks):
                raise ValueError(f"Gold phrase split across retrieval chunks for {case['id']}")
        elif case['expected'] == 'abstain':
            if not case.get('label_note'):
                raise ValueError(f"Human review note required for negative case {case['id']}")
        else:
            raise ValueError(f"Unknown label {case['expected']}")
    print(f'Validated {len(cases)} fresh challenge questions (12 answerable + 12 unanswerable) against {len(chunks)} chunks.')
    print('IMPORTANT: Negative gold labels require human review; this is not independent external data.')
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate-only', action='store_true', help='Check gold anchors; do not load models')
    parser.add_argument('--limit', type=int, default=0, help='Small smoke test only; not final metrics')
    parser.add_argument('--smoke', action='store_true', help='Run 2 positive + 2 negative quick checks')
    args = parser.parse_args()
    cases = load_validate()
    if args.validate_only:
        return
    if args.smoke:
        cases = [next(c for c in cases if c['id'] == x) for x in ('P101', 'P103', 'N111', 'N112')]
        print('SMOKE TEST ONLY (2 positive + 2 negative): not a final accuracy score.')
    elif args.limit:
        if args.limit < 1:
            raise ValueError('Limit must be >= 1')
        cases = cases[:args.limit]
        print('SMOKE TEST ONLY: partial questions; do not report as final accuracy.')

    import requests
    try:
        response = requests.get('http://127.0.0.1:8000/v1/models', timeout=5)
        response.raise_for_status()
        names = [m['id'] for m in response.json().get('data', [])]
        if 'mlx-community/Qwen2.5-3B-Instruct-4bit' not in names:
            raise RuntimeError(f'Unexpected model on port 8000: {names}')
    except (requests.RequestException, ValueError, KeyError, RuntimeError) as exc:
        print(f'ERROR: vLLM Qwen2.5-3B is not ready on port 8000: {exc}', file=sys.stderr)
        sys.exit(2)

    from verified_rag import VerifiedRAG
    engine = VerifiedRAG()
    rows = []
    for i, case in enumerate(cases, start=1):
        result = engine.answer(case['question'])
        decision = result['answerability_decision']
        expected = case['expected']
        page_ok = bool(decision == 'answer' and expected == 'answer' and result.get('verified_page') == case.get('page'))
        row = {
            'id': case['id'], 'question': case['question'], 'expected': expected,
            'predicted': decision, 'correct_decision': decision == expected,
            'expected_page': case.get('page'), 'verified_page': result.get('verified_page'),
            'page_match': page_ok,
            'reason': result.get('refusal_reason') or result.get('support_reason', ''),
            'answer': result.get('answer'), 'verified_quote': result.get('verified_quote'),
            'page_corrected': result.get('page_corrected'),
            'rerank_applied': result.get('rerank_applied'),
            'retrieval_latency_ms': result.get('retrieval_latency_ms'),
            'verification_latency_ms': result.get('verification_latency_ms'),
        }
        rows.append(row)
        print(f"[{i:02d}/{len(cases)}] {case['id']}: expected={expected}, got={decision}, page={row['verified_page']}, reason={row['reason']}", flush=True)

    valid = all(row['predicted'] != 'error' for row in rows)
    positives = [row for row in rows if row['expected'] == 'answer']
    negatives = [row for row in rows if row['expected'] == 'abstain']
    fraction = lambda items, key: round(sum(bool(x[key]) for x in items) / len(items), 4) if items else None
    if valid:
        tp_rate = fraction(positives, 'correct_decision')
        tn_rate = fraction(negatives, 'correct_decision')
        balanced = round((tp_rate + tn_rate) / 2, 4) if positives and negatives else None
    else:
        tp_rate = tn_rate = balanced = None
    summary = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'dataset': '24 new questions on SAME fictional 10-page PDF (not external validation)',
        'label_status': 'positive quote/page manually anchored; negative unanswerability manually asserted',
        'previously_seen_cases_used_to_tune': False,
        'n': len(rows), 'valid_run': valid,
        'known_answer_acceptance': tp_rate,
        'unknown_answer_rejection': tn_rate,
        'balanced_decision_accuracy': balanced,
        'positive_verified_page_rate': fraction(positives, 'page_match'),
        'model_errors': sum(row['predicted'] == 'error' for row in rows),
        'warning': 'Exact-quote match does not prove factual entailment. Manually review each accepted quote and answer. No thresholds were tuned here.',
        'cases': rows,
    }
    OUTPUT.mkdir(exist_ok=True)
    prefix = 'verified_challenge_smoke' if args.limit or args.smoke else 'verified_challenge'
    (OUTPUT / f'{prefix}.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    with (OUTPUT / f'{prefix}.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print('\n===== NEW CHALLENGE SET RESULTS =====')
    print('Valid model run:', valid)
    print('Known-answer acceptance:', tp_rate)
    print('Unknown-answer rejection:', tn_rate)
    print('Balanced decision accuracy:', balanced)
    print('Correct PDF page among answerable questions:', summary['positive_verified_page_rate'])
    print('Saved CSV and JSON to eval_results/')
    print('IMPORTANT: Check quote entailment manually; this is not production accuracy.')
    if not valid:
        sys.exit(2)

if __name__ == '__main__':
    main()
