"""tests the ArgKP Builder"""
from __future__ import annotations

from typing import List

from deepa2 import (
    DeepA2Item,
    ArgdownStatement,
    Formalization,
)

from deepa2.testing import BaseBuilderTest, RawExample

from deepa2.builder.arg_kp_builder import (
    ArgKPBuilder,
    AddContext,
    PreprocessedArgKPExample,
)


class TestArgKP(BaseBuilderTest):
    """test ArgKP"""

    builder = ArgKPBuilder()

    skip_features = [
        "context",
        "metadata",
    ]

    RAW_EXAMPLES: List[RawExample] = []

    PREPROCESSED_EXAMPLES = [
        PreprocessedArgKPExample(
            topic="topic1",
            stance=1,
            argument="argumentpro1.",
            key_point="keypoint1.",
        ),
        PreprocessedArgKPExample(
            topic="topic2",
            stance=-1,
            argument="argumentcon2.",
            key_point="keypoint2.",
        ),
    ]

    DEEP_A2_ITEMS = [
        DeepA2Item(
            source_text="topic1? argumentpro1.",
            context="topic1? Yes!",
            gist="keypoint1.",
            argdown_reconstruction="(1) keypoint1. "
            "(2) if keypoint1 then topic1. "
            "-- with modus ponens from (1) (2) -- "
            "(3) topic1.",
            premises=[
                ArgdownStatement(text="keypoint1.", ref_reco=1),
                ArgdownStatement(text="if keypoint1 then topic1.", ref_reco=2),
            ],
            conclusion=[
                ArgdownStatement(text="topic1.", ref_reco=3),
            ],
            premises_formalized=[
                Formalization(form="p", ref_reco=1),
                Formalization(form="p -> q", ref_reco=2),
            ],
            conclusion_formalized=[Formalization(form="q", ref_reco=3)],
            misc_placeholders=["p", "q"],
            plchd_substitutions=[("p", "keypoint1"), ("q", "topic1")],
        ),
        DeepA2Item(
            source_text="topic2? argumentcon2.",
            context="topic2? No!",
            gist="keypoint2.",
            argdown_reconstruction="(1) keypoint2. "
            "(2) if keypoint2 then it is not the case that topic2. "
            "-- with modus ponens from (1) (2) -- "
            "(3) it is not the case that topic2.",
            premises=[
                ArgdownStatement(text="keypoint2.", ref_reco=1),
                ArgdownStatement(
                    text="if keypoint2 then it is not the case that topic2.", ref_reco=2
                ),
            ],
            conclusion=[
                ArgdownStatement(text="it is not the case that topic2.", ref_reco=3),
            ],
            premises_formalized=[
                Formalization(form="p", ref_reco=1),
                Formalization(form="p -> not q", ref_reco=2),
            ],
            conclusion_formalized=[Formalization(form="not q", ref_reco=3)],
            misc_placeholders=["p", "q"],
            plchd_substitutions=[("p", "keypoint2"), ("q", "topic2")],
        ),
    ]

    def test_arg_kp(self, processed_examples):
        """test da2features that have been constructed non-deterministically"""
        for i, processed_example in enumerate(processed_examples):
            # context starts with topic?
            ctext = processed_example["context"]
            assert ctext.startswith(self.PREPROCESSED_EXAMPLES[i].topic)
            # context corresponding to stance?
            if self.PREPROCESSED_EXAMPLES[i].stance == 1:
                assert any((pro_expr in ctext) for pro_expr in AddContext.PRO_EXPRS)
            else:
                assert any((con_expr in ctext) for con_expr in AddContext.CON_EXPRS)
