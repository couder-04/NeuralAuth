"""
test_intent_engine.py
======================
Unit tests for IntentEngine. Uses the injected `text_generator` seam
(see IntentEngine.__init__) so these tests need NO GPU, NO model
weights, and NO transformers/torch installation — they exercise the
full validate -> retry -> normalize -> classify pipeline against a
scripted fake runtime.

Run with:
    python3 -m unittest test_intent_engine.py -v
"""

from __future__ import annotations

import json
import unittest
from typing import Dict, List, Optional, Tuple

from engines.intent_engine import IntentEngine, Transaction, InferenceMetadata, ParsedResult
from config.intent.config import load_config


def _json_response(**fields) -> str:
    base = {
        "intent": "UNKNOWN",
        "amount": 0,
        "currency": "INR",
        "beneficiary": "",
        "transaction_category": "UNKNOWN",
        "purpose": "UNKNOWN",
        "confidence": 0.5,
    }
    base.update(fields)
    return json.dumps(base)


class ScriptedGenerator:
    """
    A fake text-generation backend matching the `text_generator`
    protocol IntentEngine expects: given a batch of chat-message-lists,
    return a list of (text_or_None, input_tokens, output_tokens).

    `script` maps call index (0-based, i.e. which *round* this is) to
    either:
      - a single string/None applied to every item in that round's batch, or
      - a list of per-item strings/None (must match batch length)
    If more rounds are requested than the script has entries, the last
    entry is reused.
    Set `raise_on_call` to an exception instance/class to simulate a
    hard runtime failure (e.g. CUDA OOM) on a specific call index.
    """

    def __init__(
        self,
        script: List,
        raise_on_call: Optional[Dict[int, Exception]] = None,
    ):
        self.script = script
        self.raise_on_call = raise_on_call or {}
        self.calls: List[List[List[Dict[str, str]]]] = []

    def __call__(
        self, batch_messages: List[List[Dict[str, str]]]
    ) -> List[Tuple[Optional[str], int, int]]:
        call_idx = len(self.calls)
        self.calls.append(batch_messages)

        if call_idx in self.raise_on_call:
            raise self.raise_on_call[call_idx]

        entry = self.script[min(call_idx, len(self.script) - 1)]
        if not isinstance(entry, list):
            entry = [entry] * len(batch_messages)
        assert len(entry) == len(batch_messages), (
            f"script entry length {len(entry)} != batch length {len(batch_messages)}"
        )

        results = []
        for text in entry:
            if text is None:
                results.append((None, 0, 0))
            else:
                results.append((text, 42, 17))  # arbitrary fixed token counts
        return results


def make_engine(script, saved_beneficiaries=None, raise_on_call=None, max_retries=1):
    config = load_config()
    config = config.__class__(**{**config.__dict__, "max_retries": max_retries})
    generator = ScriptedGenerator(script, raise_on_call=raise_on_call)
    engine = IntentEngine(
        config=config,
        saved_beneficiaries=saved_beneficiaries or set(),
        text_generator=generator,
    )
    return engine, generator


class TestValidTransactionExtraction(unittest.TestCase):
    def test_valid_money_transfer_saved_beneficiary(self):
        engine, _ = make_engine(
            [_json_response(
                intent="MONEY_TRANSFER", amount=20000, currency="inr",
                beneficiary="Rahul", transaction_category="p2p_transfer",
                purpose="personal_transfer", confidence=0.92,
            )],
            saved_beneficiaries={"Rahul"},
        )
        result = engine.parse("Transfer twenty thousand rupees to Rahul")
        txn = result.transaction

        self.assertEqual(txn.intent, "MONEY_TRANSFER")
        self.assertEqual(txn.amount, 20000.0)
        self.assertEqual(txn.currency, "INR")
        self.assertEqual(txn.beneficiary, "Rahul")
        self.assertEqual(txn.beneficiary_type, "SAVED")
        self.assertEqual(txn.transaction_category, "P2P_TRANSFER")
        self.assertEqual(result.metadata.attempts_used, 1)
        self.assertTrue(result.metadata.succeeded)

    def test_valid_transfer_new_beneficiary(self):
        engine, _ = make_engine(
            [_json_response(
                intent="MONEY_TRANSFER", amount=50, currency="USD",
                beneficiary="John", transaction_category="P2P_TRANSFER",
                purpose="PERSONAL_TRANSFER", confidence=0.8,
            )],
            saved_beneficiaries={"Rahul"},
        )
        result = engine.parse("Send 50 dollars to John")
        self.assertEqual(result.transaction.beneficiary_type, "NEW")

    def test_balance_inquiry_no_beneficiary_required(self):
        engine, _ = make_engine(
            [_json_response(intent="BALANCE_INQUIRY", confidence=0.99)]
        )
        result = engine.parse("What's my account balance?")
        self.assertEqual(result.transaction.intent, "BALANCE_INQUIRY")
        self.assertEqual(result.transaction.beneficiary, "")
        self.assertEqual(result.transaction.beneficiary_type, "UNKNOWN")
        self.assertTrue(result.metadata.succeeded)

    def test_beneficiary_casing_preserved_not_titlecased(self):
        # McArthur must NOT become "Mcarthur" via .title()
        engine, _ = make_engine(
            [_json_response(
                intent="MONEY_TRANSFER", amount=10, currency="USD",
                beneficiary="McArthur", transaction_category="P2P_TRANSFER",
                purpose="PERSONAL_TRANSFER", confidence=0.7,
            )]
        )
        result = engine.parse("Send 10 dollars to McArthur")
        self.assertEqual(result.transaction.beneficiary, "McArthur")

    def test_saved_beneficiary_match_is_case_insensitive(self):
        engine, _ = make_engine(
            [_json_response(
                intent="MONEY_TRANSFER", amount=10, currency="USD",
                beneficiary="rahul", transaction_category="P2P_TRANSFER",
                purpose="PERSONAL_TRANSFER", confidence=0.7,
            )],
            saved_beneficiaries={"Rahul"},
        )
        result = engine.parse("Send 10 dollars to rahul")
        # Display name keeps model's casing ("rahul"), but classification
        # still recognizes it as SAVED via casefolded comparison.
        self.assertEqual(result.transaction.beneficiary, "rahul")
        self.assertEqual(result.transaction.beneficiary_type, "SAVED")


class TestMalformedJSON(unittest.TestCase):
    def test_non_json_output_falls_back_to_unknown_after_retries(self):
        engine, gen = make_engine(["not json at all", "still not json"])
        result = engine.parse("garbled input")
        self.assertEqual(result.transaction.intent, "UNKNOWN")
        self.assertFalse(result.metadata.succeeded)
        self.assertEqual(len(gen.calls), 2)  # initial + 1 retry

    def test_truncated_json_falls_back(self):
        engine, _ = make_engine(['{"intent": "MONEY_TRANSFER", "amount": 10'])
        result = engine.parse("truncated")
        self.assertEqual(result.transaction.intent, "UNKNOWN")

    def test_json_wrapped_in_markdown_fences_is_extracted(self):
        payload = _json_response(intent="BALANCE_INQUIRY", confidence=0.5)
        engine, _ = make_engine([f"```json\n{payload}\n```"])
        result = engine.parse("balance check")
        self.assertEqual(result.transaction.intent, "BALANCE_INQUIRY")
        self.assertTrue(result.metadata.succeeded)


class TestInvalidCurrency(unittest.TestCase):
    def test_unsupported_currency_rejected(self):
        engine, gen = make_engine([
            _json_response(intent="MONEY_TRANSFER", amount=10, currency="JPY",
                            beneficiary="Sam", transaction_category="P2P_TRANSFER",
                            purpose="PERSONAL_TRANSFER", confidence=0.5),
            _json_response(intent="MONEY_TRANSFER", amount=10, currency="JPY",
                            beneficiary="Sam", transaction_category="P2P_TRANSFER",
                            purpose="PERSONAL_TRANSFER", confidence=0.5),
        ])
        result = engine.parse("send 10 yen to Sam")
        self.assertEqual(result.transaction.intent, "UNKNOWN")
        self.assertFalse(result.metadata.succeeded)
        # confirm the retry prompt actually included a correction turn
        second_call_messages = gen.calls[1][0]
        self.assertTrue(
            any("previous response was invalid" in m["content"] for m in second_call_messages)
        )


class TestMissingBeneficiary(unittest.TestCase):
    def test_money_transfer_without_beneficiary_rejected(self):
        engine, _ = make_engine([
            _json_response(intent="MONEY_TRANSFER", amount=100, currency="INR",
                            beneficiary="", transaction_category="P2P_TRANSFER",
                            purpose="PERSONAL_TRANSFER", confidence=0.6),
            _json_response(intent="MONEY_TRANSFER", amount=100, currency="INR",
                            beneficiary="", transaction_category="P2P_TRANSFER",
                            purpose="PERSONAL_TRANSFER", confidence=0.6),
        ])
        result = engine.parse("send money")
        self.assertEqual(result.transaction.intent, "UNKNOWN")

    def test_bill_payment_without_beneficiary_rejected(self):
        engine, _ = make_engine([
            _json_response(intent="BILL_PAYMENT", amount=500, currency="INR",
                            beneficiary="", transaction_category="BILL_PAYMENT",
                            purpose="UTILITY", confidence=0.6),
        ] * 2)
        result = engine.parse("pay the bill")
        self.assertEqual(result.transaction.intent, "UNKNOWN")

    def test_recovers_when_retry_supplies_beneficiary(self):
        engine, gen = make_engine([
            _json_response(intent="BILL_PAYMENT", amount=500, currency="INR",
                            beneficiary="", transaction_category="BILL_PAYMENT",
                            purpose="UTILITY", confidence=0.6),
            _json_response(intent="BILL_PAYMENT", amount=500, currency="INR",
                            beneficiary="Electricity Board", transaction_category="BILL_PAYMENT",
                            purpose="UTILITY", confidence=0.8),
        ])
        result = engine.parse("pay five hundred rupees electricity bill")
        self.assertTrue(result.metadata.succeeded)
        self.assertEqual(result.transaction.beneficiary, "Electricity Board")
        self.assertEqual(result.metadata.attempts_used, 2)


class TestExcessiveAmount(unittest.TestCase):
    def test_amount_over_ceiling_rejected(self):
        config = load_config()
        over_ceiling = config.max_transaction_amount + 1
        engine, _ = make_engine([
            _json_response(intent="MONEY_TRANSFER", amount=over_ceiling, currency="INR",
                            beneficiary="Sam", transaction_category="P2P_TRANSFER",
                            purpose="PERSONAL_TRANSFER", confidence=0.5),
        ] * 2)
        result = engine.parse("send an absurd amount to Sam")
        self.assertEqual(result.transaction.intent, "UNKNOWN")

    def test_negative_amount_rejected(self):
        engine, _ = make_engine([
            _json_response(intent="MONEY_TRANSFER", amount=-5, currency="INR",
                            beneficiary="Sam", transaction_category="P2P_TRANSFER",
                            purpose="PERSONAL_TRANSFER", confidence=0.5),
        ] * 2)
        result = engine.parse("send negative five to Sam")
        self.assertEqual(result.transaction.intent, "UNKNOWN")

    def test_nan_amount_rejected(self):
        engine, _ = make_engine([
            '{"intent": "MONEY_TRANSFER", "amount": NaN, "currency": "INR", '
            '"beneficiary": "Sam", "transaction_category": "P2P_TRANSFER", '
            '"purpose": "PERSONAL_TRANSFER", "confidence": 0.5}'
        ] * 2)
        result = engine.parse("send NaN to Sam")
        self.assertEqual(result.transaction.intent, "UNKNOWN")


class TestRetryLogic(unittest.TestCase):
    def test_success_on_first_attempt_does_not_trigger_second_call(self):
        engine, gen = make_engine([_json_response(intent="BALANCE_INQUIRY", confidence=0.9)])
        engine.parse("balance please")
        self.assertEqual(len(gen.calls), 1)

    def test_correction_message_reflects_actual_validation_error(self):
        engine, gen = make_engine([
            _json_response(intent="MONEY_TRANSFER", amount=10, currency="ZZZ",
                            beneficiary="Sam", transaction_category="P2P_TRANSFER",
                            purpose="PERSONAL_TRANSFER", confidence=0.5),
            _json_response(intent="MONEY_TRANSFER", amount=10, currency="USD",
                            beneficiary="Sam", transaction_category="P2P_TRANSFER",
                            purpose="PERSONAL_TRANSFER", confidence=0.5),
        ])
        result = engine.parse("send 10 ZZZ to Sam")
        self.assertTrue(result.metadata.succeeded)
        retry_messages = gen.calls[1][0]
        correction_turn = [m for m in retry_messages if "previous response was invalid" in m["content"]]
        self.assertEqual(len(correction_turn), 1)
        self.assertIn("currency", correction_turn[0]["content"].lower())

    def test_respects_configured_max_retries(self):
        engine, gen = make_engine(["garbage"] * 5, max_retries=3)
        engine.parse("nonsense")
        self.assertEqual(len(gen.calls), 4)  # initial + 3 retries


class TestGenerationFailures(unittest.TestCase):
    def test_generator_returning_none_falls_back_to_unknown(self):
        engine, gen = make_engine([None, None])
        result = engine.parse("anything")
        self.assertEqual(result.transaction.intent, "UNKNOWN")
        self.assertFalse(result.metadata.succeeded)
        self.assertEqual(result.metadata.failure_reason,
                          "all generation/validation attempts exhausted")

    def test_generator_raising_exception_falls_back_to_unknown(self):
        # max_retries=0: isolates "generation raised" -> UNKNOWN behavior
        # from retry-recovery behavior (covered separately below).
        engine, _ = make_engine(
            [_json_response()],
            raise_on_call={0: RuntimeError("simulated backend crash")},
            max_retries=0,
        )
        result = engine.parse("anything")
        self.assertEqual(result.transaction.intent, "UNKNOWN")
        self.assertFalse(result.metadata.succeeded)

    def test_simulated_cuda_oom_degrades_gracefully(self):
        class FakeCudaOOM(RuntimeError):
            pass

        engine, _ = make_engine(
            [_json_response()],
            raise_on_call={0: FakeCudaOOM("CUDA out of memory")},
            max_retries=0,
        )
        result = engine.parse("anything")
        self.assertEqual(result.transaction.intent, "UNKNOWN")
        self.assertFalse(result.metadata.succeeded)

    def test_transient_generation_failure_recovers_on_retry(self):
        # A generation failure (e.g. transient OOM) IS retried, and if
        # the retry succeeds, the item resolves normally rather than
        # being permanently marked UNKNOWN.
        engine, gen = make_engine(
            [None, _json_response(intent="BALANCE_INQUIRY", confidence=0.6)],
            max_retries=1,
        )
        result = engine.parse("anything")
        self.assertTrue(result.metadata.succeeded)
        self.assertEqual(result.metadata.attempts_used, 2)
        self.assertEqual(len(gen.calls), 2)

    def test_engine_never_raises_out_of_parse(self):
        engine, _ = make_engine(
            [_json_response()],
            raise_on_call={0: Exception("totally unexpected failure")},
        )
        try:
            result = engine.parse("anything")
        except Exception as exc:  # pragma: no cover
            self.fail(f"parse() must never raise, but raised: {exc}")
        self.assertIsInstance(result, ParsedResult)


class TestBatchedParsing(unittest.TestCase):
    def test_parse_many_generates_one_call_per_round_not_per_item(self):
        engine, gen = make_engine([
            [
                _json_response(intent="BALANCE_INQUIRY", confidence=0.9),
                'not json',
                _json_response(intent="BALANCE_INQUIRY", confidence=0.8),
            ],
            [_json_response(intent="BALANCE_INQUIRY", confidence=0.7)],  # retry round: only item 2
        ])
        results = engine.parse_many(["a", "b", "c"])
        self.assertEqual(len(gen.calls), 2)  # 1 batched call per round, not 3+3
        self.assertEqual(len(gen.calls[0]), 3)  # round 1: all 3 items batched together
        self.assertEqual(len(gen.calls[1]), 1)  # round 2: only the failed item retried
        self.assertTrue(all(r.metadata.succeeded for r in results))

    def test_parse_many_preserves_order(self):
        engine, _ = make_engine([
            [
                _json_response(intent="BALANCE_INQUIRY", confidence=0.1),
                _json_response(intent="TRANSACTION_HISTORY", confidence=0.2),
            ]
        ])
        results = engine.parse_many(["first", "second"])
        self.assertEqual(results[0].transaction.intent, "BALANCE_INQUIRY")
        self.assertEqual(results[1].transaction.intent, "TRANSACTION_HISTORY")

    def test_request_ids_propagate_into_metadata(self):
        engine, _ = make_engine([_json_response(intent="BALANCE_INQUIRY", confidence=0.5)])
        result = engine.parse("balance?", request_id="req-123", session_id="sess-1", user_id="user-9")
        self.assertEqual(result.metadata.request_id, "req-123")
        self.assertEqual(result.metadata.session_id, "sess-1")
        self.assertEqual(result.metadata.user_id, "user-9")

    def test_auto_generated_request_id_when_not_supplied(self):
        engine, _ = make_engine([_json_response(intent="BALANCE_INQUIRY", confidence=0.5)])
        result = engine.parse("balance?")
        self.assertTrue(result.metadata.request_id)  # non-empty auto uuid


class TestTransactionMetadataSeparation(unittest.TestCase):
    def test_transaction_has_no_inference_fields(self):
        txn_fields = set(Transaction.__dataclass_fields__.keys())
        self.assertNotIn("latency_ms", txn_fields)
        self.assertNotIn("model_name", txn_fields)
        self.assertNotIn("attempts_used", txn_fields)

    def test_metadata_has_no_business_fields(self):
        meta_fields = set(InferenceMetadata.__dataclass_fields__.keys())
        self.assertNotIn("amount", meta_fields)
        self.assertNotIn("beneficiary", meta_fields)
        self.assertNotIn("intent", meta_fields)


if __name__ == "__main__":
    unittest.main(verbosity=2)