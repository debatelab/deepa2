"""tests the datastructures """

import dataclasses

from deepa2 import (
    DeepA2Item,
    QuotedStatement,
)


def test_from_batch():
    """test from batch"""
    source_text = "source_text-123"
    reasons = [
        QuotedStatement(text="reasons-123", ref_reco=1),
        QuotedStatement(text="reasons-123", ref_reco=2),
    ]
    conjectures = [QuotedStatement(text="conjectures-123", ref_reco=3)]
    da2_item = DeepA2Item(
        source_text=source_text, reasons=reasons, conjectures=conjectures
    )
    print(dataclasses.fields(da2_item))

    da2_batched = {k: [v] for k, v in dataclasses.asdict(da2_item).items()}

    print(da2_batched)

    da2_item2 = DeepA2Item.from_batch(da2_batched)

    print(da2_item2)

    assert da2_item2 == da2_item
