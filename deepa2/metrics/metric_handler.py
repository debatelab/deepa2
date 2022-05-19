"""metric handlers and basic class for calculating metrics"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional, List, Dict

import pandas

from deepa2 import DeepA2Parser


class DA2MetricHandler(ABC):
    """
    The Handler interface declares a method for building the chain of handlers.
    It also declares a method for executing a request.
    """

    @abstractmethod
    def set_next(self, handler: DA2MetricHandler) -> DA2MetricHandler:
        """set next handler"""

    @abstractmethod
    def handle(self, prediction: str, reference: str) -> Optional[Dict]:
        """handle request"""


class AbstractDA2MetricHandler(DA2MetricHandler):
    """
    The default chaining behavior can be implemented inside a base handler
    class.
    """

    _next_handler: Optional[DA2MetricHandler] = None
    _da2_parser: DeepA2Parser

    def __init__(self) -> None:
        self._da2_parser = DeepA2Parser()

    def set_next(self, handler: DA2MetricHandler) -> DA2MetricHandler:
        self._next_handler = handler
        # Returning a handler from here will let us link handlers in a
        # convenient way like this:
        # monkey.set_next(squirrel).set_next(dog)
        return handler

    @abstractmethod
    def handle(self, prediction: str, reference: str) -> Optional[Dict]:
        if self._next_handler:
            return self._next_handler.handle(prediction, reference)

        return None


# All Concrete DA2 Metric Handlers either handle a request or pass it
# to the next handler in the chain.


class ArgdownHandler(AbstractDA2MetricHandler):
    """handles argument reconstructions"""

    def handle(self, prediction: str, reference: str) -> Optional[Dict]:
        ref_as_argdown = self._da2_parser.parse_argdown(reference)
        if ref_as_argdown:
            # reference is argdown
            score: Dict[str, Any] = {}
            return score
        return super().handle(prediction, reference)


class StatementHandler(AbstractDA2MetricHandler):
    """handles statement list predictions"""

    def handle(self, prediction: str, reference: str) -> Optional[Dict]:
        is_statement_list = False
        if is_statement_list:
            score: Dict[str, Any] = {}
            return score
        return super().handle(prediction, reference)


class FormalizationHandler(AbstractDA2MetricHandler):
    """handles formalization predictions"""

    def handle(self, prediction: str, reference: str) -> Optional[Dict]:
        is_formalization_list = False
        if is_formalization_list:
            score: Dict[str, Any] = {}
            return score
        return super().handle(prediction, reference)


class DA2PredictionEvaluator:  # pylint: disable=too-few-public-methods
    """evaluates a list of predictions and references"""

    def __init__(self) -> None:
        self.argdown_evaluator = ArgdownHandler()
        self.statement_evaluator = StatementHandler()
        self.formalization_evaluator = FormalizationHandler()

        self.argdown_evaluator.set_next(self.statement_evaluator).set_next(
            self.formalization_evaluator
        )

    def compute_metrics(self, predictions: List[str], references: List[str]):
        """
        compute da2 metrics of predictions given references

        Args:
        predictions: list of predictions to score.
        references: list of reference for each prediction.
        """

        scores = []
        for pred, ref in zip(predictions, references):
            score = self.argdown_evaluator.handle(pred, ref)
            scores.append(score)

        # aggregate scores
        df_scores = pandas.DataFrame.from_records(scores)

        return df_scores.mean(axis=0).to_dict()
