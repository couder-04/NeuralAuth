"""
tests/test_intent.py

Unit tests for IntentEngine.
"""

import pytest

from engines.intent_engine import (
    IntentEngine,
    Transaction,
)


# ---------------------------------------------------------
# Fake Generator
# ---------------------------------------------------------

def generator_factory(response):

    def generate(messages):

        return [

            (
                response,
                100,
                25,
            )

        ]

    return generate


# ---------------------------------------------------------
# Engine
# ---------------------------------------------------------

def create_engine(response):

    return IntentEngine(

        text_generator=generator_factory(response),

        saved_beneficiaries={"Rahul", "Amit"},

    )


# ---------------------------------------------------------
# Money Transfer
# ---------------------------------------------------------

def test_money_transfer():

    engine = create_engine(
        """
        {
            "intent":"MONEY_TRANSFER",
            "amount":2500,
            "currency":"INR",
            "beneficiary":"Rahul",
            "transaction_category":"P2P_TRANSFER",
            "purpose":"PERSONAL_TRANSFER",
            "confidence":0.97
        }
        """
    )

    result = engine.parse(
        "Transfer ₹2500 to Rahul"
    )

    tx = result.transaction

    assert tx.intent == "MONEY_TRANSFER"

    assert tx.amount == 2500

    assert tx.currency == "INR"

    assert tx.beneficiary == "Rahul"

    assert tx.beneficiary_type == "SAVED"


# ---------------------------------------------------------
# New Beneficiary
# ---------------------------------------------------------

def test_new_beneficiary():

    engine = create_engine(
        """
        {
            "intent":"MONEY_TRANSFER",
            "amount":500,
            "currency":"INR",
            "beneficiary":"Rohan",
            "transaction_category":"P2P_TRANSFER",
            "purpose":"PERSONAL_TRANSFER",
            "confidence":0.95
        }
        """
    )

    tx = engine.parse("Send money").transaction

    assert tx.beneficiary_type == "NEW"


# ---------------------------------------------------------
# Balance Inquiry
# ---------------------------------------------------------

def test_balance():

    engine = create_engine(
        """
        {
            "intent":"BALANCE_INQUIRY",
            "amount":0,
            "currency":"INR",
            "beneficiary":"",
            "transaction_category":"UNKNOWN",
            "purpose":"UNKNOWN",
            "confidence":0.99
        }
        """
    )

    tx = engine.parse(
        "Check balance"
    ).transaction

    assert tx.intent == "BALANCE_INQUIRY"


# ---------------------------------------------------------
# Invalid JSON
# ---------------------------------------------------------

def test_invalid_json():

    engine = create_engine(

        "INVALID"

    )

    tx = engine.parse(

        "Hello"

    ).transaction

    assert tx.intent == "UNKNOWN"


# ---------------------------------------------------------
# Missing Beneficiary
# ---------------------------------------------------------

def test_missing_beneficiary():

    engine = create_engine(
        """
        {
            "intent":"MONEY_TRANSFER",
            "amount":100,
            "currency":"INR",
            "beneficiary":"",
            "transaction_category":"P2P_TRANSFER",
            "purpose":"PERSONAL_TRANSFER",
            "confidence":0.9
        }
        """
    )

    tx = engine.parse(

        "Transfer"

    ).transaction

    assert tx.intent == "UNKNOWN"


# ---------------------------------------------------------
# Currency Validation
# ---------------------------------------------------------

def test_invalid_currency():

    engine = create_engine(
        """
        {
            "intent":"BALANCE_INQUIRY",
            "amount":0,
            "currency":"XYZ",
            "beneficiary":"",
            "transaction_category":"UNKNOWN",
            "purpose":"UNKNOWN",
            "confidence":0.9
        }
        """
    )

    tx = engine.parse(

        "Balance"

    ).transaction

    assert tx.intent == "UNKNOWN"


# ---------------------------------------------------------
# Confidence Range
# ---------------------------------------------------------

def test_invalid_confidence():

    engine = create_engine(
        """
        {
            "intent":"BALANCE_INQUIRY",
            "amount":0,
            "currency":"INR",
            "beneficiary":"",
            "transaction_category":"UNKNOWN",
            "purpose":"UNKNOWN",
            "confidence":2.0
        }
        """
    )

    tx = engine.parse("Balance").transaction

    assert tx.intent == "UNKNOWN"


# ---------------------------------------------------------
# Batch Parsing
# ---------------------------------------------------------

def test_parse_many():

    response = """
    {
        "intent":"BALANCE_INQUIRY",
        "amount":0,
        "currency":"INR",
        "beneficiary":"",
        "transaction_category":"UNKNOWN",
        "purpose":"UNKNOWN",
        "confidence":0.95
    }
    """

    engine = IntentEngine(

        text_generator=lambda messages: [

            (response,100,20)

            for _ in messages

        ]

    )

    results = engine.parse_many(

        [

            "Balance",

            "Balance",

            "Balance",

        ]

    )

    assert len(results) == 3

    for r in results:

        assert r.transaction.intent == "BALANCE_INQUIRY"


# ---------------------------------------------------------
# Unknown Transaction
# ---------------------------------------------------------

def test_unknown_transaction():

    tx = Transaction.unknown()

    assert tx.intent == "UNKNOWN"

    assert tx.amount == 0

    assert tx.confidence == 0


# ---------------------------------------------------------
# Metadata
# ---------------------------------------------------------

def test_metadata():

    engine = create_engine(
        """
        {
            "intent":"BALANCE_INQUIRY",
            "amount":0,
            "currency":"INR",
            "beneficiary":"",
            "transaction_category":"UNKNOWN",
            "purpose":"UNKNOWN",
            "confidence":0.95
        }
        """
    )

    result = engine.parse(

        "Balance"

    )

    meta = result.metadata

    assert meta.succeeded is True

    assert meta.attempts_used == 1

    assert meta.input_tokens > 0

    assert meta.output_tokens > 0