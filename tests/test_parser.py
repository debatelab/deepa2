"""tests the layouter """

import pytest

from deepa2 import (
    DeepA2Parser,
)

from deepa2.parsers import Argument


@pytest.fixture(name="argdown_examples")
def fixture_argdown_examples():
    """argdown examples"""
    examples = [
        """(1) premise ---- (2) conclusion""",
        """(1) premise -- with mp from (1) -- (2) conclusion""",
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
    """test first argument"""
    argument: Argument = parsed_arguments[1]
    print(argument)
    assert len(argument.statements) == 2
