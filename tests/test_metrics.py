"""tests the metrics """

import pytest

from deepa2.metrics import (
    DA2PredictionEvaluator,
)


@pytest.fixture(name="examples")
def fixture_examples():
    """prediction -- reference pairs"""

    base_reference_1 = """
        (1) Premise 1
        -- with mp from (1) --
        (2) i-conclusion 1
        (3) Premise 2
        -- from (1) --
        (4) i-conclusion 2
        (5) Premise 3
        (6) Premise 4
        ----
        (7) conclusion
        """
    base_predictions_1 = [
        """
        -- with mp from (1) --
        (2) i-conclusion 1
        (3) Premise 2
        -- from (1) --
        (4) i-conclusion 2
        (5) Premise 3
        (6) Premise 4
        ----
        (7) conclusion
        """,
        """
        (1) Premise 1
        -- with mp from (2) --
        (2) i-conclusion 1
        (3) Premise 2
        -- from (1) --
        (4) i-conclusion 2
        (5) Premise 1
        (6) Premise 4
        """,
        """
        (1) Premise 1
        -- with mp from (2) --
        (2) i-conclusion 1
        (3) Premise 2
        -- from (1) --
        (4) i-conclusion 2
        (5) Premise 3
        (6) Premise 4
        ----
        (7) conclusion
        """,
        """
        (1) Premise 1
        -- with mp from (1) --
        (2) i-conclusion 1
        (3) Premise 2
        -- from (1) --
        (4) Premise 1
        (5) Premise 3
        (6) Premise 4
        ----
        (7) conclusion
        """,
    ]
    base_predictions_1 += [base_reference_1]
    predictions = base_predictions_1
    references = [base_reference_1] * len(base_predictions_1)

    return predictions, references


def test_evaluator(examples):
    """test global metrics"""

    scorer = DA2PredictionEvaluator()
    predictions, references = examples

    metrics = scorer.compute_metrics(predictions, references)

    print(metrics)

    assert "valid_argdown" in metrics
    assert metrics.get("valid_argdown") > 0
    assert metrics.get("valid_argdown") < 1
    assert metrics.get("no_redundancy") < 1
    assert metrics.get("no_redundancy") > 0
    assert metrics.get("no_petitio") < 1
    assert metrics.get("no_petitio") > 0
    assert metrics.get("inferential_similarity") < 1

    assert scorer.scores[0]["no_redundancy"] is None
    assert scorer.scores[0]["valid_argdown"] == 0

    assert scorer.scores[-1]["inferential_similarity"] == 1


def test_evaluator_no_arg():
    """test global metrics"""

    scorer = DA2PredictionEvaluator()
    predictions = [
        "is a reason (ref: (1))",
        "is another reason (ref: (1))",
    ]
    references = [
        "is a reason (ref: (1))",
        "is another reason (ref: (2))",
    ]

    metrics = scorer.compute_metrics(predictions, references)

    print(metrics)
    assert "bleu-score" in metrics
    assert round(metrics["bleu-score"]) == 86


def test_evaluator_mixed():
    """test global metrics"""

    scorer = DA2PredictionEvaluator()
    predictions = [
        "is a reason (ref: (1))",
        "(1) Premise 1 -- with mp from (1) -- (2) i-conclusion 1",
        "four items too many",
        "four items too many",
        None,
    ]
    references = [
        "is a reason (ref: (1))",
        "(1) Premise 1 -- with mp from (1) -- (2) i-conclusion 1",
        None,
        None,
        "four items too few",
    ]

    metrics = scorer.compute_metrics(predictions, references)

    assert "bleu-score" in metrics
    assert round(metrics["bleu-score"]) == 64
    assert metrics.get("valid_argdown") == 1
