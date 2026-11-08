"""Evidence-verified adaptive RAG pipeline for a *fictional* 10-page PDF.

Shared by app_verified.py and evaluate_verified.py. The verifier is a fallible
LLM: an exact quote proves provenance, not that the answer logically follows.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

from adaptive_rerank import Gate, gate_features, skip_rerank
from answer_support import verify_answer_support
from evidence_guard import EvidenceThresholds, evidence_features, has_enough_evidence
from hybrid_retrieval import BM25Index, rrf_fuse

ROOT = Path(__file__).resolve().parent
REFUSAL = "I couldn't find enough verified evidence in the available handbook to answer that question."
TOP_K = 8
EVIDENCE_K = 3
RERANK_GATE = Gate(bm25_margin_min=0.05, cosine_margin_min=0.0)  # Old DEV-tuned config; frozen


def load_evidence_thresholds(path: Path | None = None) -> EvidenceThresholds:
    """Load thresholds from the previous DEV-only calibration, not the new tests."""
    config = path or ROOT / 'eval_results' / 'evidence_gate_config.json'
    if not config.is_file():
        raise FileNotFoundError('Missing eval_results/evidence_gate_config.json. Run existing evaluate_refusal.py first.')
    raw = json.loads(config.read_text(encoding='utf-8'))
    return EvidenceThresholds(min_cosine=float(raw['min_cosine']), min_coverage=float(raw['min_coverage']))


class VerifiedRAG:
    def __init__(self) -> None:
        # Heavy dependencies are intentionally lazy, so label validation requires
        # no ML models and no API connection.
        import faiss
        from sentence_transformers import SentenceTransformer, CrossEncoder
        from evaluate import load_chunks

        self.chunks = load_chunks()
        self.texts = [item['text'] for item in self.chunks]
        self.lexical = BM25Index(self.texts)
        self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
        vectors = self.embedder.encode(self.texts, normalize_embeddings=True, show_progress_bar=False).astype('float32')
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2', device='cpu')
        self.evidence_thresholds = load_evidence_thresholds()
        print(f'Verified RAG ready: {len(self.chunks)} chunks, adaptive route, evidence check.', flush=True)

    def answer(self, question: str) -> dict[str, Any]:
        question = question.strip()
        if len(question) < 3:
            raise ValueError('Question too short')
        retrieval_t0 = perf_counter()
        qvec = self.embedder.encode([question], normalize_embeddings=True, show_progress_bar=False).astype('float32')
        dense_scores, dense_pos = self.index.search(qvec, min(TOP_K, len(self.chunks)))
        dense_ids = [int(x) for x in dense_pos[0]]
        dense_values = [float(s) for s in dense_scores[0]]
        lexical_scores = self.lexical.score(question)
        lexical_ids = sorted(range(len(self.chunks)), key=lambda i: (-lexical_scores[i], i))[:TOP_K]
        lexical_values = [float(lexical_scores[i]) for i in lexical_ids]
        candidate_ids = rrf_fuse(dense_ids, lexical_ids, top_k=TOP_K)
        features = gate_features(dense_ids, dense_values, lexical_ids, lexical_values)
        skip = skip_rerank(features, RERANK_GATE)
        quick_features = evidence_features(question, dense_values[0], [self.texts[i] for i in candidate_ids[:EVIDENCE_K]])
        enough = has_enough_evidence(quick_features, self.evidence_thresholds)

        base = {
            'question': question,
            'retrieval_method': 'Hybrid-RRF (rerank skipped)' if skip else 'Hybrid + Cross-Encoder Rerank',
            'rerank_applied': False,
            'gate_features': features,
            'evidence_features': quick_features,
            'verified_quote': None,
            'verified_page': None,
            'page_corrected': False,
            'sources': [],
        }
        if not enough:
            return {**base, 'answer': REFUSAL, 'answerability_decision': 'abstain',
                    'refusal_reason': 'low_evidence_screening_score',
                    'retrieval_latency_ms': round((perf_counter()-retrieval_t0)*1000, 2),
                    'verification_latency_ms': 0.0}

        if skip:
            ranked = [(i, None) for i in candidate_ids[:EVIDENCE_K]]
        else:
            scores = self.reranker.predict([[question, self.texts[i]] for i in candidate_ids])
            ranked_all = sorted(zip(candidate_ids, [float(s) for s in scores]), key=lambda pair: -pair[1])
            ranked = ranked_all[:EVIDENCE_K]
        base['rerank_applied'] = not skip
        retrieval_ms = (perf_counter() - retrieval_t0) * 1000
        evidence = [self.chunks[i] for i, _ in ranked]
        t0 = perf_counter()
        verification = verify_answer_support(question, evidence)
        verification_ms = (perf_counter()-t0)*1000
        common = {**base, 'retrieval_latency_ms': round(retrieval_ms, 2),
                  'verification_latency_ms': round(verification_ms, 2)}
        if verification['decision'] == 'error':
            return {**common, 'answer': None, 'answerability_decision': 'error',
                    'refusal_reason': verification['reason']}
        if verification['decision'] != 'answer':
            return {**common, 'answer': REFUSAL, 'answerability_decision': 'abstain',
                    'refusal_reason': verification['reason']}

        # Resolve quote source from returned, original evidence; never trust a
        # model invented page number, and never search documents outside retrieval.
        quote_norm = ' '.join(verification['quote'].split()).casefold()
        supporting = [
            (idx, score) for idx, score in ranked
            if quote_norm in ' '.join(self.chunks[idx]['text'].split()).casefold()
            and self.chunks[idx]['page'] == verification['page']
        ]
        if not supporting:
            return {**common, 'answer': REFUSAL, 'answerability_decision': 'abstain',
                    'refusal_reason': 'source_resolution_failed'}
        idx, score = supporting[0]
        source = {'page': self.chunks[idx]['page'], 'excerpt': self.chunks[idx]['text'],
                  'rerank_score': round(score, 3) if score is not None else None,
                  'verified_quote': verification['quote'], 'source_role': 'verbatim_quote_match'}
        return {**common,
                'answer': verification['answer'] + f" [Page {verification['page']}]",
                'answerability_decision': 'answer', 'refusal_reason': None,
                'support_reason': verification['reason'],
                'verified_quote': verification['quote'], 'verified_page': verification['page'],
                'page_corrected': verification.get('page_corrected', False),
                'sources': [source]}
