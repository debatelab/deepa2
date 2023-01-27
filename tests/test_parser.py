"""tests the layouter """

import pytest

from deepa2 import (
    DeepA2Parser,
)
from deepa2.datastructures import Formalization

from deepa2.parsers import Argument, ArgumentStatement, FOLParser, QuotedStatement


@pytest.fixture(name="argdown_examples")
def fixture_argdown_examples():
    """argdown examples"""
    examples = [
        """(1) premise \n---- (2) conclusion""",
        """(1) premise -- with mp from (1) -- (2) conclusion""",
        """
        ```argdown
        (1) premise 1
        -- with mp from (1) --
        (2) i-conclusion 1
        (3) premise 2
        -- from (1) --
        (4) i-conclusion 2
        (5) premise 3
        (6) premise 4
        ----
        (7) conclusion
        ```
        """,
        """
        -- with mp from (1) --
        (2) i-conclusion 1
        (3) premise 2
        -- from (1) --
        (4) i-conclusion 2
        (5) premise 3
        (6) premise 4
        ----
        (7) conclusion
        """,
        """
        (1) premise 1
        -- with mp from (1) --
        (2) i-conclusion 1
        (3) premise 2
        -- from (1) --
        (4) i-conclusion 2
        (5) premise 3
        (6) premise 4
        """,
        "(1) If someone is a sufferer of allergy to mango, "
        "then they are not a sufferer of allergy to sesame or "
        "a sufferer of allergy to carrot. (2) If someone is "
        "allergic to carrot, then they aren't allergic to mango. -- "
        "with generalized dilemma [negation variant, transposition] "
        "from (1) (2) -- (3) If someone is allergic to mango, then "
        "they aren't allergic to sesame. (4) If someone isn't "
        "allergic to sesame, then they are allergic to turkey. -- "
        "with hypothetical syllogism [negation variant] from (3) (4) "
        "-- (5) If someone is allergic to mango, then they are "
        "allergic to turkey.",
    ]
    return examples


@pytest.fixture(name="parsed_arguments")
def fixture_parsed_arguments(argdown_examples):
    """parsed_arguments"""
    arguments = []
    for example in argdown_examples:
        argument = DeepA2Parser.parse_argdown(example)
        arguments.append(argument)

    return arguments


def test_example_1(parsed_arguments):
    """test first argument"""
    argument: Argument = parsed_arguments[0]
    print(argument)
    assert len(argument.statements) == 2
    assert argument.statements[0].label == 1


def test_example_2(parsed_arguments):
    """test second argument"""
    argument: Argument = parsed_arguments[1]
    print(argument)
    assert len(argument.statements) == 2


def test_example_3(parsed_arguments):
    """test third argument"""
    argument: Argument = parsed_arguments[2]
    reference = Argument(
        statements=[
            ArgumentStatement(
                text="premise 1",
                is_conclusion=False,
                label=1,
                uses=None,
                inference_info=None,
                schemes=None,
                variants=None,
            ),
            ArgumentStatement(
                text="i-conclusion 1",
                is_conclusion=True,
                label=2,
                uses=[1],
                inference_info="with mp from (1)",
                schemes=["mp"],
                variants=None,
            ),
            ArgumentStatement(
                text="premise 2",
                is_conclusion=False,
                label=3,
                uses=None,
                inference_info=None,
                schemes=None,
                variants=None,
            ),
            ArgumentStatement(
                text="i-conclusion 2",
                is_conclusion=True,
                label=4,
                uses=[],
                inference_info="from (1)",
                schemes=None,
                variants=None,
            ),
            ArgumentStatement(
                text="premise 3",
                is_conclusion=False,
                label=5,
                uses=None,
                inference_info=None,
                schemes=None,
                variants=None,
            ),
            ArgumentStatement(
                text="premise 4",
                is_conclusion=False,
                label=6,
                uses=None,
                inference_info=None,
                schemes=None,
                variants=None,
            ),
            ArgumentStatement(
                text="conclusion",
                is_conclusion=True,
                label=7,
                uses=[],
                inference_info="",
                schemes=None,
                variants=None,
            ),
        ]
    )
    print(argument)
    assert argument == reference


def test_example_4(parsed_arguments):
    """test second argument"""
    argument: Argument = parsed_arguments[3]
    print(argument)
    assert argument is None


def test_example_5(parsed_arguments):
    """test second argument"""
    argument: Argument = parsed_arguments[4]
    print(argument)
    assert not argument.statements[-1].is_conclusion


def test_example_6(parsed_arguments):
    """test second argument"""
    argument: Argument = parsed_arguments[5]
    print(argument)
    assert len(argument.statements) == 5


def test_empty_label():
    """test empty label"""
    ad_raw = """() premise \n---- (2) conclusion"""
    argument = DeepA2Parser.parse_argdown(ad_raw)
    assert argument is None


def test_keys1():
    """test keys"""
    text = "p: premise | q: conclusion"
    quotes = DeepA2Parser.parse_keys(text)
    references = [
        ("p", "premise"),
        ("q", "conclusion"),
    ]
    assert quotes == references


def test_keys2():
    """test keys"""
    text = "p : sen_p | F : pred_F"
    quotes = DeepA2Parser.parse_keys(text)
    references = [
        ("p", "sen_p"),
        ("F", "pred_F"),
    ]
    assert quotes == references


def test_quotations1():
    """test quotations"""
    text = "quote-1 (ref: (2)) | quote 2 (ref: (1))"
    quotes = DeepA2Parser.parse_quotes(text)
    references = [
        QuotedStatement(text="quote-1", ref_reco=2),
        QuotedStatement(text="quote 2", ref_reco=1),
    ]
    assert quotes == references


def test_formalizations1():
    """test formalization"""
    text = "(x): F x -> G x (ref: (2)) | (x)(y): F x -> (R x y v G y) (ref: (1))"
    formalizations = DeepA2Parser.parse_formalization(text)
    references = [
        Formalization(form="(x): F x -> G x", ref_reco=2),
        Formalization(form="(x)(y): F x -> (R x y v G y)", ref_reco=1),
    ]
    assert formalizations == references


def test_formalizations2():
    """test formalization"""
    text = "(x): F x -> G x (REF: (2)) | (x)(y): F x -> (Rxy v G y) (ref: (1))"
    formalizations = DeepA2Parser.parse_formalization(text)
    references = [None, None]
    assert formalizations == references


def test_fol_parser1():
    """test FOL Parser"""
    formalizations = [
        Formalization(form="(x): F x -> G x", ref_reco=2),
        Formalization(form="(x)(y): F x -> (R x y v G y)", ref_reco=1),
        Formalization(form="(x)(y): F x -> (Ez):(R x y v G y)", ref_reco=1),
        Formalization(form="(x): ((y): F x) -> (R x y v G y)", ref_reco=1),
        Formalization(form="(x)(y): F x -> not (R x y & not G y)", ref_reco=1),
    ]
    formulae = FOLParser.parse(formalizations)
    print(formulae)
    assert all(bool(f) for f in formulae)


# TODO: add more tests for FOL parser
