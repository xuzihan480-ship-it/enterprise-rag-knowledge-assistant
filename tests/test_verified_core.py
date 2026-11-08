"""Offline regression checks; require no model, network or GPU.
Run: python -m unittest discover -s tests -p 'test_verified_core.py'
"""
import json
import unittest
from unittest.mock import patch

from answer_support import parse_verification
from evidence_guard import EvidenceThresholds
from verified_rag import VerifiedRAG


class _Embedding:
    def encode(self, *args, **kwargs):
        class _Encoded:
            def astype(self, _):
                return self
        return _Encoded()

class _Index:
    def search(self, vector, k):
        return [[0.87, 0.6, 0.1][:k]], [[0, 1, 2][:k]]

class _BM:
    def score(self, question):
        return [8, 3, 1]


class VerifiedTest(unittest.TestCase):
    def setUp(self):
        self.eng = object.__new__(VerifiedRAG)
        self.eng.chunks = [
            {'page': 2, 'text': 'Standard access grants expire after 90 calendar days.'},
            {'page': 3, 'text': 'Data quality dimensions are completeness, uniqueness, and consistency.'},
            {'page': 7, 'text': 'Reranking can increase latency in retrieval systems.'},
        ]
        self.eng.texts = [c['text'] for c in self.eng.chunks]
        self.eng.embedder = _Embedding()
        self.eng.index = _Index()
        self.eng.lexical = _BM()
        self.eng.evidence_thresholds = EvidenceThresholds(min_cosine=0.0, min_coverage=0.0)

    def test_page_repair_only_from_provided_passages(self):
        quote = 'Standard access grants expire after 90 calendar days.'
        src = [{'page': 2, 'text': quote}]
        result = parse_verification(json.dumps({'supported': True, 'quote': quote, 'page': 3, 'answer': '90 calendar days'}), src)
        self.assertEqual(result['page'], 2)
        self.assertTrue(result['page_corrected'])
        self.assertEqual(parse_verification(json.dumps({'supported': True, 'quote': quote, 'page': 3, 'answer': '90 calendar days'}), [{'page': 3, 'text': 'other info'}])['decision'], 'abstain')

    @patch('verified_rag.has_enough_evidence', return_value=True)
    @patch('verified_rag.verify_answer_support')
    def test_success_uses_verified_page(self, verify, enough):
        quote = self.eng.chunks[0]['text']
        verify.return_value = {'decision': 'answer', 'answer': '90 calendar days', 'quote': quote, 'page': 2, 'page_corrected': True, 'reason': 'quote_page_corrected'}
        result = self.eng.answer('How long do standard access grants last?')
        self.assertEqual(result['answerability_decision'], 'answer')
        self.assertEqual(result['sources'][0]['page'], 2)
        self.assertIn('[Page 2]', result['answer'])

    @patch('verified_rag.has_enough_evidence', return_value=True)
    @patch('verified_rag.verify_answer_support')
    def test_verifier_error_is_not_counted_as_abstention(self, verify, enough):
        verify.return_value = {'decision': 'error', 'reason': 'model_service_error: connection refused'}
        result = self.eng.answer('What is the NDS access policy?')
        self.assertEqual(result['answerability_decision'], 'error')
        self.assertIsNone(result['answer'])

    @patch('verified_rag.has_enough_evidence', return_value=True)
    @patch('verified_rag.verify_answer_support')
    def test_no_verified_quote_cannot_return_answer(self, verify, enough):
        verify.return_value = {'decision': 'answer', 'quote': 'No such phrase appears in the passages.', 'answer': 'invented', 'page': 2, 'reason': 'verbatim_quote_verified'}
        result = self.eng.answer('Who runs the company?')
        self.assertEqual(result['answerability_decision'], 'abstain')

if __name__ == '__main__':
    unittest.main()
