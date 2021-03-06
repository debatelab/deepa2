"""tests the ESNLIBuilder"""
from __future__ import annotations
import dataclasses

from deepa2 import DeepA2Item

from deepa2.testing import BaseBuilderTest

from deepa2.builder.aifdb_builder import (
    AIFDBBuilder,
    RawAIFDBExample,
    PreprocessedAIFDBExample,
)


class TestEnBank(BaseBuilderTest):
    """test enbank"""

    builder = AIFDBBuilder(name="test")

    # skip all:
    skip_features = [field.name for field in dataclasses.fields(DeepA2Item)]

    RAW_EXAMPLES = [
        RawAIFDBExample(
            nodeset={
                "nodes": [
                    {
                        "nodeID": "8837",
                        "text": "analyses",
                        "type": "L",
                    },
                    {
                        "nodeID": "28819",
                        "text": "It was a ghastly aberration",
                        "type": "I",
                    },
                    {
                        "nodeID": "28820",
                        "text": "Asserting",
                        "type": "YA",
                    },
                    {
                        "nodeID": "28821",
                        "text": "LA :  It was a ghastly aberration",
                        "type": "L",
                    },
                    {
                        "nodeID": "28822",
                        "text": "it was in fact typical",
                        "type": "I",
                    },
                    {
                        "nodeID": "28823",
                        "text": "Rhetorical Questioning",
                        "type": "YA",
                    },
                    {
                        "nodeID": "28824",
                        "text": "CL : Or was it in fact typical?",
                        "type": "L",
                    },
                    {
                        "nodeID": "28825",
                        "text": "TA",
                        "type": "TA",
                    },
                    {
                        "nodeID": "28826",
                        "text": "CA",
                        "type": "CA",
                    },
                    {
                        "nodeID": "28827",
                        "text": "Disagreeing",
                        "type": "YA",
                    },
                    {
                        "nodeID": "28828",
                        "text": "it was the product of a policy that was unsustainable "
                        "that could only be pursued by increasing repression",
                        "type": "I",
                    },
                    {
                        "nodeID": "28829",
                        "text": "Assertive Questioning",
                        "type": "YA",
                    },
                    {
                        "nodeID": "28830",
                        "text": "CL : Was it the product of a policy that was "
                        "unsustainable that could only be pursued by increasing repression?",
                        "type": "L",
                    },
                    {
                        "nodeID": "28831",
                        "text": "TA",
                        "type": "TA",
                    },
                    {
                        "nodeID": "28832",
                        "text": "RA",
                        "type": "RA",
                    },
                    {
                        "nodeID": "28833",
                        "text": "Arguing",
                        "type": "YA",
                    },
                ],
                "edges": [
                    {
                        "edgeID": "32170",
                        "fromID": "8837",
                        "toID": "8837",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33053",
                        "fromID": "8837",
                        "toID": "28819",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33054",
                        "fromID": "8837",
                        "toID": "28820",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33055",
                        "fromID": "8837",
                        "toID": "28821",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33056",
                        "fromID": "28820",
                        "toID": "28819",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33057",
                        "fromID": "28821",
                        "toID": "28820",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33058",
                        "fromID": "8837",
                        "toID": "28822",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33059",
                        "fromID": "8837",
                        "toID": "28823",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33060",
                        "fromID": "8837",
                        "toID": "28824",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33061",
                        "fromID": "28823",
                        "toID": "28822",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33062",
                        "fromID": "28824",
                        "toID": "28823",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33063",
                        "fromID": "28821",
                        "toID": "28825",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33064",
                        "fromID": "8837",
                        "toID": "28825",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33065",
                        "fromID": "28825",
                        "toID": "28824",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33066",
                        "fromID": "28822",
                        "toID": "28826",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33067",
                        "fromID": "8837",
                        "toID": "28826",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33068",
                        "fromID": "28826",
                        "toID": "28819",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33069",
                        "fromID": "28825",
                        "toID": "28827",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33070",
                        "fromID": "8837",
                        "toID": "28827",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33071",
                        "fromID": "28827",
                        "toID": "28826",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33072",
                        "fromID": "8837",
                        "toID": "28828",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33073",
                        "fromID": "8837",
                        "toID": "28829",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33074",
                        "fromID": "8837",
                        "toID": "28830",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33075",
                        "fromID": "28829",
                        "toID": "28828",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33076",
                        "fromID": "28830",
                        "toID": "28829",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33077",
                        "fromID": "28824",
                        "toID": "28831",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33078",
                        "fromID": "8837",
                        "toID": "28831",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33079",
                        "fromID": "28831",
                        "toID": "28830",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33080",
                        "fromID": "28828",
                        "toID": "28832",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33081",
                        "fromID": "8837",
                        "toID": "28832",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33082",
                        "fromID": "28832",
                        "toID": "28822",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33083",
                        "fromID": "28831",
                        "toID": "28833",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33084",
                        "fromID": "8837",
                        "toID": "28833",
                        "formEdgeID": None,
                    },
                    {
                        "edgeID": "33085",
                        "fromID": "28833",
                        "toID": "28832",
                        "formEdgeID": None,
                    },
                ],
            },
            text="""

LA:
It
was a ghastly aberration.



CL:
Or
was it in fact typical?

Was it the product of a policy that was
unsustainable that could only be pursued by

increasing repression?
""",
            corpus="test_corpus",
        ),
    ]

    PREPROCESSED_EXAMPLES = [
        PreprocessedAIFDBExample(
            text="\n\nLA:\nIt\nwas a ghastly aberration.\n\n\n\nCL:\nOr\nwas it in fact "
            "typical?\n\nWas it the product of a policy that was\nunsustainable that could "
            "only be pursued by\n\nincreasing repression?\n",
            corpus="test_corpus",
            type="CA",
            reasons=["CL : Or was it in fact typical?"],
            conjectures=["LA :  It was a ghastly aberration"],
            premises=["it was in fact typical"],
            conclusions=["It was a ghastly aberration"],
        ),
        PreprocessedAIFDBExample(
            text="\n\nLA:\nIt\nwas a ghastly aberration.\n\n\n\nCL:\nOr\nwas it "
            "in fact typical?\n\nWas it the product of a policy that was\nunsustainable "
            "that could only be pursued by\n\nincreasing repression?\n",
            corpus="test_corpus",
            type="RA",
            reasons=[
                "CL : Was it the product of a policy that was unsustainable that could "
                "only be pursued by increasing repression?"
            ],
            conjectures=["CL : Or was it in fact typical?"],
            premises=[
                "it was the product of a policy that was unsustainable that could only "
                "be pursued by increasing repression"
            ],
            conclusions=["it was in fact typical"],
        ),
    ]

    def test_aifdb_source(self, processed_examples):
        """test processed da2items"""
        for da2item in processed_examples:
            assert da2item["source_text"] == self.RAW_EXAMPLES[0].text

    @staticmethod
    def test_aifdb_reasons(processed_examples):
        """test processed da2items"""
        text = "CL : Or was it in fact typical?"
        assert processed_examples[0]["reasons"][0]["text"] == text

    @staticmethod
    def test_aifdb_conjectures(processed_examples):
        """test processed da2items"""
        text = "LA :  It was a ghastly aberration"
        assert processed_examples[0]["conjectures"][0]["text"] == text

    @staticmethod
    def test_aifdb_paraphrase(processed_examples):
        """test processed da2items"""
        text = (
            "it was the product of a policy that was unsustainable "
            "that could only be pursued by increasing repression In "
            "consequence: it was in fact typical"
        )
        assert processed_examples[1]["source_paraphrase"][:100] == text[:100]
        assert processed_examples[1]["source_paraphrase"][-10:] == text[-10:]
