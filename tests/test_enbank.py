"""tests the EnBankBuilder"""
from __future__ import annotations

from deepa2 import (
    ArgdownStatement,
    DeepA2Item,
    Formalization,
    QuotedStatement,
)

from deepa2.testing import BaseBuilderTest

from deepa2.builder.entailmentbank_builder import (
    EnBankBuilder,
    RawEnBankExample,
    PreprocessedEnBankExample,
)


class TestEnBank(BaseBuilderTest):
    """test enbank"""

    builder = EnBankBuilder()

    skip_features = [
        "source_text",
        "metadata",
        "source_paraphrase",
        "reasons",
        "conjectures",
        "premises",
    ]

    RAW_EXAMPLES = [
        RawEnBankExample(
            id="AKDE&ED_2012_4_22",
            context="context",
            question="question",
            answer="answer",
            hypothesis="hypothesis",
            proof="sent1 & sent2 -> int1: int_conc_1; int1 & sent3 -> hypothesis; ",
            full_text_proof="full text proof",
            depth_of_proof="2",
            length_of_proof="2",
            meta={
                "question_text": "question_text",
                "answer_text": "answer_text.",
                "hypothesis_id": "int4",
                "triples": {
                    "sent1": "sent_1",
                    "sent2": "sent_2",
                    "sent3": "sent_3",
                    "sent4": "distractor",
                },
                "distractors": ["sent4"],
                "distractors_relevance": [0.74396926],
                "intermediate_conclusions": {"int1": "int_conc_1"},
                "core_concepts": ["core_concept"],
                "step_proof": "sent1 & sent2 -> int1: int_conc_1; int1 & sent3 -> hypothesis; ",
            },
        ),
    ]

    PREPROCESSED_EXAMPLES = [
        PreprocessedEnBankExample(
            id="AKDE&ED_2012_4_22",
            step_proof="sent1 & sent2 -> int1: int_conc_1; int1 & sent3 -> hypothesis; ",
            triples={
                "sent1": "sent_1",
                "sent2": "sent_2",
                "sent3": "sent_3",
                "sent4": "distractor",
            },
            intermediate_conclusions={"int1": "int_conc_1"},
            question_text="question_text",
            answer_text="answer_text.",
            hypothesis="hypothesis",
            core_concepts=["core_concept"],
            distractors=["sent4"],
        )
    ]

    DEEP_A2_ITEMS = [
        DeepA2Item(
            source_text="answer_text. that is because distractor. sent_1. sent_2. sent_3.",
            title="core_concept",
            gist="hypothesis",
            source_paraphrase="sent_1 sent_2 sent_3 Therefore: answer_text..",
            context="question_text",
            argdown_reconstruction=(
                "(1) sent_1\n"
                "(2) sent_2\n"
                "--\n"
                "with ?? from (1) (2)\n"
                "--\n"
                "(3) int_conc_1\n"
                "(4) sent_3\n"
                "--\n"
                "with ?? from (3) (4)\n"
                "--\n"
                "(5) int_conc_1"
            ),
            erroneous_argdown="",
            reasons=[
                QuotedStatement(text="sent_1", starts_at=-1, ref_reco=1),
                QuotedStatement(text="sent_2", starts_at=-1, ref_reco=2),
                QuotedStatement(text="sent_3", starts_at=-1, ref_reco=4),
            ],
            conjectures=[
                QuotedStatement(
                    text="question_text answer_text.", starts_at=-1, ref_reco=5
                )
            ],
            premises=[
                ArgdownStatement(text="sent_1", explicit="", ref_reco=1),
                ArgdownStatement(text="sent_2", explicit="", ref_reco=2),
                ArgdownStatement(text="sent_3", explicit="", ref_reco=4),
            ],
            intermediary_conclusions=[ArgdownStatement()],
            conclusion=[ArgdownStatement(text="int_conc_1.", explicit="", ref_reco=5)],
            premises_formalized=[Formalization()],
            intermediary_conclusions_formalized=[Formalization()],
            conclusion_formalized=[Formalization()],
            predicate_placeholders=[],
            entity_placeholders=[],
            misc_placeholders=[],
            plchd_substitutions=[],
            metadata=[
                ("labels", "{'sent1': 1, 'sent2': 2, 'int1': 5, 'sent3': 4}"),
                ("reason_order", "['sent1', 'sent2', 'sent3']"),
            ],
        )
    ]

    def test_enbank(self, processed_examples):
        """test da2features that have been constructed non-deterministically"""

        for i, processed_example in enumerate(processed_examples):

            for k in [
                "reasons",
                "conjectures",
                "premises",
            ]:
                texts_gt = {r.text for r in getattr(self.DEEP_A2_ITEMS[i], k)}
                texts_proc = {r["text"] for r in processed_example[k]}
                assert texts_gt == texts_proc

            assert all(
                (v in processed_example["source_text"])
                for _, v in self.PREPROCESSED_EXAMPLES[i].triples.items()
            )

            assert processed_example["source_text"].startswith("answer_text")
