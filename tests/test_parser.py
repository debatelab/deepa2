"""tests the layouter """

import pytest

from deepa2 import (
    DeepA2Parser,
)

from deepa2.parsers import (
    Argument,
    ArgumentStatement
)


@pytest.fixture(name="parser")
def fixture_parser() -> DeepA2Parser:
    """DeepA2Parser"""
    return DeepA2Parser()

@pytest.fixture(name="argdown_examples")
def fixture_argdown_examples():
    """argdown examples"""
    examples = [
        """(1) premise ---- (2) conclusion""",
    ]
    return examples

@pytest.fixture(name="parsed_arguments")
def fixture_parsed_arguments(argdown_examples):
    """parsed_arguments"""
    parser = DeepA2Parser()
    arguments = []
    for example in argdown_examples:
        argument = parser.parse_argdown(example)
        arguments.append(argument)

    return arguments





def test_title(layouter):
    """test layouter"""
    title = "title-123"
    da2item = layouter.format(DeepA2Item(title=title))
    assert da2item["title"] == title


def test_title_none(layouter):
    """test layouter"""
    title = None
    da2item = layouter.format(DeepA2Item(title=title))
    assert da2item["title"] == title


def test_title_none2(layouter):
    """test layouter"""
    title = "title-123"
    reasons = None
    da2item = layouter.format(DeepA2Item(title=title, reasons=reasons))
    assert da2item["title"] == title


def test_reasons(layouter):
    """test layouter"""
    t_1 = "text_1"
    r_1 = 2
    s_1 = 123
    t_2 = "text_2"
    r_2 = 4
    s_2 = -1
    reasons = [
        QuotedStatement(text=t_1, ref_reco=r_1, starts_at=s_1),
        QuotedStatement(text=t_2, ref_reco=r_2, starts_at=s_2),
    ]
    da2item = layouter.format(DeepA2Item(reasons=reasons))
    assert da2item["reasons"] == f"{t_1} (ref: ({r_1})) | {t_2} (ref: ({r_2}))"


def test_reasons_none(layouter):
    """test layouter"""
    reasons = None
    da2item = layouter.format(DeepA2Item(reasons=reasons))
    assert da2item["reasons"] == reasons


def test_reasons_none2(layouter):
    """test layouter"""
    reasons = []
    da2item = layouter.format(DeepA2Item(reasons=reasons))
    assert da2item["reasons"] == " "


def test_adst(layouter):
    """test layouter"""
    t_1 = "text_1"
    r_1 = 2
    premises = [ArgdownStatement(text=t_1, ref_reco=r_1, explicit=False)]
    da2item = layouter.format(DeepA2Item(premises=premises))
    assert da2item["premises"] == f"{t_1} (ref: ({r_1}))"


def test_form(layouter):
    """test layouter"""
    t_1 = "(x): not F(x)"
    r_1 = 2
    conclusion_formalized = [Formalization(form=t_1, ref_reco=r_1)]
    da2item = layouter.format(DeepA2Item(conclusion_formalized=conclusion_formalized))
    assert da2item["conclusion_formalized"] == f"{t_1} (ref: ({r_1}))"


def test_plcd(layouter):
    """test layouter"""
    misc_placeholders = ["p", "r", "and-me-too"]
    da2item = layouter.format(DeepA2Item(misc_placeholders=misc_placeholders))
    assert da2item["misc_placeholders"] == "p | r | and-me-too"


def test_subst(layouter):
    """test layouter"""
    plchd_substitutions = [("p", "sen_p"), ("F", "pred_F")]
    da2item = layouter.format(DeepA2Item(plchd_substitutions=plchd_substitutions))
    assert da2item["plchd_substitutions"] == "p : sen_p | F : pred_F"
