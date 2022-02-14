"""tests the ESNLIBuilder"""
from __future__ import annotations
import dataclasses

import pytest
import datasets

from deepa2.builder.entailmentbank_builder import (
    EnBankBuilder,
    RawEnBankExample,
    PreprocessedEnBankExample,
)


RAW_EXAMPLES_1 = [
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

PREPROCESSED_EXAMPLES_1 = [
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
    {
        "source_text": "answer_text. that is because distractor. sent_1. sent_2. sent_3.",
        "title": "core_concept",
        "gist": "hypothesis",
        "source_paraphrase": "sent_1 sent_2 sent_3 Therefore: answer_text..",
        "context": "question_text",
        "argdown_reconstruction": (
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
        "erroneous_argdown": "",
        "reasons": [
            {"text": "sent_1", "starts_at": -1, "ref_reco": 1},
            {"text": "sent_2", "starts_at": -1, "ref_reco": 2},
            {"text": "sent_3", "starts_at": -1, "ref_reco": 4},
        ],
        "conjectures": [
            {"text": "question_text answer_text.", "starts_at": -1, "ref_reco": 5}
        ],
        "premises": [
            {"text": "sent_1", "explicit": "", "ref_reco": 1},
            {"text": "sent_2", "explicit": "", "ref_reco": 2},
            {"text": "sent_3", "explicit": "", "ref_reco": 4},
        ],
        "intermediary_conclusions": [{"text": "", "explicit": "", "ref_reco": -1}],
        "conclusion": [{"text": "int_conc_1.", "explicit": "", "ref_reco": 5}],
        "premises_formalized": [{"form": "", "ref_reco": -1}],
        "intermediary_conclusions_formalized": [{"form": "", "ref_reco": -1}],
        "conclusion_formalized": [{"form": "", "ref_reco": -1}],
        "predicate_placeholders": [],
        "entity_placeholders": [],
        "misc_placeholders": [],
        "plchd_substitutions": [],
        "distractors": [],
        "metadata": [
            ("labels", "{'sent1': 1, 'sent2': 2, 'int1': 5, 'sent3': 4}"),
            ("reason_order", "['sent1', 'sent2', 'sent3']"),
        ],
    }
]


def test_enbank_preprocessor():
    """tests esnli preprocessor"""
    raw_data = {}
    for field in dataclasses.fields(RawEnBankExample):
        key = field.name
        raw_data[key] = [getattr(example, key) for example in RAW_EXAMPLES_1]
    enbank_dataset = datasets.Dataset.from_dict(raw_data)
    enbank_dataset = EnBankBuilder.preprocess(enbank_dataset)
    preprocessed_data = enbank_dataset.to_dict(
        batch_size=1, batched=True
    )  # return iterator
    preprocessed_enbank_examples = []
    for batch in preprocessed_data:
        # unbatch
        preprocessed_enbank_example = PreprocessedEnBankExample(
            **{k: v[0] for k, v in batch.items()}
        )
        preprocessed_enbank_examples.append(preprocessed_enbank_example)
    assert preprocessed_enbank_examples == PREPROCESSED_EXAMPLES_1


@pytest.fixture(name="processed_examples")
def fixture_processed_examples():
    """processes examples"""
    da2items = []
    for preprocessed_enbank_example in PREPROCESSED_EXAMPLES_1:
        batched_input = {
            k: [v] for k, v in dataclasses.asdict(preprocessed_enbank_example).items()
        }
        enbank_builder = EnBankBuilder()
        enbank_builder.set_input(batched_input)
        enbank_builder.configure_product()
        enbank_builder.produce_da2item()
        enbank_builder.postprocess_da2item()
        enbank_builder.add_metadata_da2item()
        da2items.extend(enbank_builder.product)  # product is a list of dicts
    return da2items


def test_enbank_item_1(processed_examples):
    """contrary hypothesis is never conclusion"""

    proc_example = processed_examples[0]
    for k, val in proc_example.items():
        if DEEP_A2_ITEMS[0][k] != val:
            if k not in ["source_text", "metadata"]:
                print(f"{k}: {val}")

    assert all(
        DEEP_A2_ITEMS[0][k] == v
        for k, v in proc_example.items()
        if k
        not in [
            "source_text",
            "metadata",
            "source_paraphrase",
            "reasons",
            "conjectures",
            "premises",
        ]
    )

    for k in [
        "reasons",
        "conjectures",
        "premises",
    ]:
        texts_gt = {r["text"] for r in DEEP_A2_ITEMS[0][k]}
        texts_proc = {r["text"] for r in proc_example[k]}
        assert texts_gt == texts_proc

    assert all(
        (v in proc_example["source_text"])
        for _, v in PREPROCESSED_EXAMPLES_1[0].triples.items()
    )

    assert proc_example["source_text"].startswith("answer_text")
