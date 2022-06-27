"""tests the ArgQ"""
from __future__ import annotations

from typing import List

from deepa2 import (
    DeepA2Item,
    QuotedStatement,
)

from deepa2.testing import BaseBuilderTest, RawExample

from deepa2.builder.arg_q_builder import (
    ArgQBuilder,
    AddSourceText,
    PreprocessedArgQExample,
)


class TestArgQ(BaseBuilderTest):
    """test ArgQ"""

    builder = ArgQBuilder()

    skip_features = [
        "source_text",
        "conjectures",
        "metadata",
    ]

    RAW_EXAMPLES: List[RawExample] = []

    PREPROCESSED_EXAMPLES = [
        PreprocessedArgQExample(
            topic="topic1",
            stance=1,
            argument_stance_conf="argumentpro1",
            argument_stance_nonconf="argumentcon1.",
        ),
        PreprocessedArgQExample(
            topic="topic2",
            stance=-1,
            argument_stance_conf="argumentcon2",
            argument_stance_nonconf="argumentpro2.",
        ),
    ]

    DEEP_A2_ITEMS = [
        DeepA2Item(
            source_text="topic1? Yes! argumentpro1. argumentcon1.",
            reasons=[
                QuotedStatement(text="argumentpro1", starts_at=-1, ref_reco=1),
            ],
            conjectures=[
                QuotedStatement(text="topic1? Yes!", starts_at=-1, ref_reco=3),
            ],
        ),
        DeepA2Item(
            source_text="topic2? No! argumentpro2. argumentcon2.",
            reasons=[
                QuotedStatement(text="argumentcon2", starts_at=-1, ref_reco=1),
            ],
            conjectures=[
                QuotedStatement(text="topic2? No!", starts_at=-1, ref_reco=3),
            ],
        ),
    ]

    def test_arg_q(self, processed_examples):
        """test da2features that have been constructed non-deterministically"""
        for i, processed_example in enumerate(processed_examples):
            # source_text starts with topic?
            assert processed_example["source_text"].startswith(
                self.PREPROCESSED_EXAMPLES[i].topic
            )
            # get conjecture
            ctext = processed_example["conjectures"][0]["text"]
            # conjecture in source_text?
            assert ctext in processed_example["source_text"]
            # conjecture corresponding to stance?
            if self.PREPROCESSED_EXAMPLES[i].stance == 1:
                assert any((pro_expr in ctext) for pro_expr in AddSourceText.PRO_EXPRS)
            else:
                assert any((con_expr in ctext) for con_expr in AddSourceText.CON_EXPRS)
