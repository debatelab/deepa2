"""tests the layouter """

import pytest

from deepa2 import (
    DeepA2Parser,
)

from deepa2.parsers import Argument, ArgumentStatement


@pytest.fixture(name="argdown_examples")
def fixture_argdown_examples():
    """argdown examples"""
    examples = [
        """(1) premise ---- (2) conclusion""",
        """(1) premise -- with mp from (1) -- (2) conclusion""",
        """
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
