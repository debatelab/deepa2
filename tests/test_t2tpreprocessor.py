"""tests the t2tpreprocessor """

import dataclasses

from deepa2 import (
    DeepA2Item,
    QuotedStatement,
)

from deepa2.preptrain import T2TPreprocessor


config_1 = {
    "sources": [{"path": "."}],
    "export_path": ".",
    "export_format": "parquet",
    "generative_modes": [
        {
            "name": "not-used-to-infer-inputs-and-target",
            "target": "conjectures",
            "input": ["source_text", "reasons"],
            "weight": 1,
        }
    ],
    "input_column_name": "text",
    "target_column_name": "target",
}

config_2 = {  # same as config_1, mode inferred from name
    "sources": [{"path": "."}],
    "export_path": ".",
    "export_format": "parquet",
    "generative_modes": [
        {
            "name": "s+r => j",
        }
    ],
    "input_column_name": "text",
    "target_column_name": "target",
}


def test_1():
    """test T2TPreprocessor"""
    t2t_preprocessor = T2TPreprocessor(**config_1)
    source_text = "source_text-1234"
    reasons = [
        QuotedStatement(text="reasons-1234", ref_reco=1),
        QuotedStatement(text="reasons-1234", ref_reco=2),
    ]
    conjectures = [QuotedStatement(text="conjectures-1234", ref_reco=3)]
    da2_item = DeepA2Item(
        source_text=source_text, reasons=reasons, conjectures=conjectures
    )
    t2t_item = t2t_preprocessor.map_to_t2t(
        {k: [v] for k, v in dataclasses.asdict(da2_item).items()}
    )

    print(t2t_item)

    assert t2t_item["target"][0] == "conjectures-1234 (ref: (3))"
    assert (
        t2t_item["text"][0] == "conjectures: source_text: source_text-1234 "
        "reasons: reasons-1234 (ref: (1)) | reasons-1234 (ref: (2))"
    )


def test_infer_keys_from_name():
    """input / output correctly infered from keys"""
    modes_1 = T2TPreprocessor(  # pylint: disable=protected-access
        **config_1
    )._generative_modes
    modes_2 = T2TPreprocessor(  # pylint: disable=protected-access
        **config_2
    )._generative_modes

    assert modes_1[0].input == modes_2[0].input

    assert modes_1[0].target == modes_2[0].target
