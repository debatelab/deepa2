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
    "generative_modes": [
        {
            "name": "a+r => j",
            "target": "conjectures",
            "input": ["source_text", "reasons"],
            "weight": 1,
        }
    ],
    "input_column_name": "text",
    "target_column_name": "target",
}

# TODO: test config with keys only


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
