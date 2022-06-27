"""metric handlers and basic class for calculating metrics"""

from __future__ import annotations
from abc import ABC, abstractmethod
import re
from typing import Any, Optional, List, Dict, Sequence

import editdistance  # type: ignore
import numpy as np
import pandas
import sacrebleu as scb

from deepa2 import DeepA2Parser, Formalization
from deepa2.parsers import Argument


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
        ref_as_argdown = DeepA2Parser.parse_argdown(reference)
        if ref_as_argdown:
            # reference is argdown
            pred_as_argdown = DeepA2Parser.parse_argdown(prediction)
            score = self.score(pred_as_argdown, ref_as_argdown)
            return score
        return super().handle(prediction, reference)

    def score(
        self, parsed_pred: Optional[Argument], parsed_ref: Optional[Argument]
    ) -> Dict[str, Any]:
        """scores a reconstructed argument relative to a reference reconstruction"""

        score = {
            "valid_argdown": self.valid_argdown(parsed_pred),
            "pc_structure": self.pc_structure(parsed_pred),
            "consistent_usage": self.consistent_usage(parsed_pred),
            "no_petitio": self.no_petitio(parsed_pred),
            "no_redundancy": self.no_redundancy(parsed_pred),
            "inferential_similarity": self.inferential_similarity(
                parsed_pred, parsed_ref
            ),
        }

        score.update(self.compute_aggregates(score))

        return score

    @staticmethod
    def compute_aggregates(score: Dict[str, Any]) -> Dict[str, Any]:
        """calculates aggregates"""

        agg_scores = {}

        agg_ad_1 = 0
        if score.get("valid_argdown", 0):
            agg_ad_1 = (
                score["pc_structure"]
                and score["consistent_usage"]
                and score["no_redundancy"]
            )
        agg_scores["agg_ad_1"] = agg_ad_1

        return agg_scores

    @staticmethod
    def valid_argdown(parsed_pred: Optional[Argument]) -> int:
        """checks if a reconstruction is valid argdown"""

        return 1 if parsed_pred else 0

    @staticmethod
    def pc_structure(parsed_pred: Optional[Argument]) -> Optional[int]:
        """checks if a reconstruction has premises and conclusion"""
        if parsed_pred is None:
            return None

        has_pc_structure = (
            not parsed_pred.statements[0].is_conclusion
        ) and parsed_pred.statements[-1].is_conclusion

        return int(has_pc_structure)

    @staticmethod
    def consistent_usage(parsed_pred: Optional[Argument]) -> Optional[int]:
        """checks if info about used statements is consistent"""

        if parsed_pred is None:
            return None

        used_exist = True  # does every statement referred to in inference exist?
        used_statements = []
        for statement in parsed_pred.statements:
            if statement.uses and statement.label:
                if any(u >= statement.label for u in statement.uses):
                    used_exist = False
                    break
                used_statements.extend(statement.uses)
        # is every statement (except final one) explicitly referred to in some inference?
        evryth_used = len(set(used_statements)) == (len(parsed_pred.statements) - 1)
        has_consistent_usage = used_exist and evryth_used

        return int(has_consistent_usage)

    @staticmethod
    def no_petitio(parsed_pred: Optional[Argument]) -> Optional[int]:
        """checks if a reconstruction is no petitio
        i.e. no conclusion is a premise,
        petitio is a special case of redundancy"""

        if parsed_pred is None:
            return None

        no_petitio = True
        visited_texts = []
        for statement in parsed_pred.statements:
            if statement.text:
                if statement.is_conclusion:
                    # check if conclusion has been introduced as premise before
                    if statement.text.strip() in visited_texts:
                        no_petitio = False
                        break
                else:
                    visited_texts.append(statement.text.strip())

        return int(no_petitio)

    @staticmethod
    def no_redundancy(parsed_pred: Optional[Argument]) -> Optional[int]:
        """checks if a reconstruction is redundant
        i.e. no statements has been introduced before"""

        if parsed_pred is None:
            return None

        statement_texts = [s.text.strip() for s in parsed_pred.statements if s.text]

        no_redundancy = len(statement_texts) == len(set(statement_texts))

        return int(no_redundancy)

    @staticmethod
    def inferential_similarity(
        parsed_pred: Optional[Argument], parsed_ref: Optional[Argument]
    ) -> Optional[float]:
        """checks if predicted and target argument are inferentially similar"""

        if parsed_pred and parsed_ref:

            n_pp = len(list(s for s in parsed_pred.statements if not s.is_conclusion))
            n_pr = len(list(s for s in parsed_ref.statements if not s.is_conclusion))
            n_cp = len(list(s for s in parsed_pred.statements if s.is_conclusion))
            n_cr = len(list(s for s in parsed_ref.statements if s.is_conclusion))
            inf_sim = (1 - abs(n_pp - n_pr) / (n_pp + n_pr)) * (
                1 - abs(n_cp - n_cr) / (n_cp + n_cr)
            )
        else:
            inf_sim = None

        return inf_sim


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

    _MIN_SCORES = {
        "form_abstract_sim": 0,
        "form_acc_refs": 0,
        "form_bleu": 0,
    }

    def handle(self, prediction: str, reference: str) -> Optional[Dict]:
        ref_as_formulae = DeepA2Parser.parse_formalization(reference)
        if ref_as_formulae:
            if None not in ref_as_formulae:
                # reference is formalization
                pred_as_formulae = DeepA2Parser.parse_formalization(prediction)
                score = self.score(pred_as_formulae, ref_as_formulae)
                return score
        return super().handle(prediction, reference)

    def score(
        self,
        pred_as_formulae: Optional[List[Optional[Formalization]]],
        ref_as_formulae: Optional[List[Optional[Formalization]]],
    ):
        """calculate similiarity score by comparing two lists of formalizations"""

        # references should be a list of formalizations
        if ref_as_formulae is None:
            return {}
        if None in ref_as_formulae:
            return {}
        # minimum scores if predictions are not a list of formalizations
        if pred_as_formulae is None:
            return self._MIN_SCORES
        if None in pred_as_formulae:
            return self._MIN_SCORES
        if len(pred_as_formulae) != len(ref_as_formulae):
            return self._MIN_SCORES

        form_acc_refs = all(
            p.ref_reco == r.ref_reco
            for p, r in zip(pred_as_formulae, ref_as_formulae)
            if p is not None and r is not None  # redundant, for mypy
        )

        form_abstract_sim = np.mean(
            [
                self.abstract_sim(p.form, r.form)
                for p, r in zip(pred_as_formulae, ref_as_formulae)
                if p is not None and r is not None  # redundant, for mypy
            ]
        )

        # pairwise bleu
        pairwise_bleu = self.pairwise_bleu(
            [p.form if p is not None else " " for p in pred_as_formulae],
            [r.form if r is not None else " " for r in ref_as_formulae],
        )

        scores = {
            "form_abstract_sim": form_abstract_sim,
            "form_acc_refs": form_acc_refs,
            "form_bleu": pairwise_bleu,
        }

        return scores

    @staticmethod
    def abstract_sim(form1: str, form2: str) -> float:
        """calculates structural similarity between to formulas"""
        # remove white space
        af1 = form1.replace(" ", "")
        af2 = form2.replace(" ", "")
        # use a single propositional constant
        af1 = re.sub("[pqrst]", "p", af1)
        af2 = re.sub("[pqrst]", "p", af2)
        # use a single name
        af1 = re.sub("[abcde]", "a", af1)
        af2 = re.sub("[abcde]", "a", af2)
        # use a single predicate
        af1 = re.sub("[FGHIJKLM]", "F", af1)
        af2 = re.sub("[FGHIJKLM]", "F", af2)
        # use a single relation
        af1 = re.sub("[RSTUVW]", "R", af1)
        af2 = re.sub("[RSTUVW]", "R", af2)
        # use a single variable
        af1 = re.sub("[xyzuw]", "x", af1)
        af2 = re.sub("[xyzuw]", "x", af2)

        return 1 - editdistance.eval(af1, af2) / max(len(af1), len(af2))

    @staticmethod
    def pairwise_bleu(p_forms: List[str], r_forms: List[str]) -> float:
        """calculates pairwise bleu scores"""
        # replace blank string with white space char
        p_forms = [p if p else " " for p in p_forms]
        r_forms = [r if r else " " for r in r_forms]
        if p_forms:
            scb_output = scb.corpus_bleu(
                p_forms,
                [r_forms],
                lowercase=False,
            )
        score = round(scb_output.score, 2)
        return score


class DA2PredictionEvaluator:  # pylint: disable=too-few-public-methods
    """evaluates a list of predictions and references"""

    def __init__(self) -> None:
        self.argdown_evaluator = ArgdownHandler()
        self.formalization_evaluator = FormalizationHandler()
        self.statement_evaluator = StatementHandler()

        self.argdown_evaluator.set_next(self.formalization_evaluator).set_next(
            self.statement_evaluator
        )

        self._scores: Sequence[Optional[Dict[str, Any]]] = []

    @property
    def scores(self) -> Sequence[Optional[Dict[str, Any]]]:
        """
        The latest individual scores calculated by the evaluator.
        """
        return self._scores

    def compute_metrics(self, predictions: List[str], references: List[str]):
        """
        compute da2 metrics of predictions given references

        Args:
        predictions: list of predictions to score.
        references: list of reference for each prediction.
        """

        if len(predictions) != len(references):
            raise ValueError("Number of predictions and references must be the same.")

        # replace None with blank string
        predictions = [p if p else "" for p in predictions]
        references = [r if r else "" for r in references]

        scores = []
        prds_unprocessed = []
        refs_unprocessed = []
        for pred, ref in zip(predictions, references):
            score = self.argdown_evaluator.handle(pred, ref)
            if score:
                scores.append(score)
            else:
                prds_unprocessed.append(pred)
                refs_unprocessed.append(ref)

        # aggregate scores
        if scores:
            df_scores = pandas.DataFrame.from_records(scores)
        else:
            df_scores = pandas.DataFrame()

        # shelve individual scores
        self._scores = scores

        # average over da2 scores
        output_dict = df_scores.mean(axis=0).to_dict()  # type: ignore

        # replace blank string with white space char
        prds_unprocessed = [p if p else " " for p in prds_unprocessed]
        refs_unprocessed = [r if r else " " for r in refs_unprocessed]

        # process remaining predictions
        if prds_unprocessed:
            scb_output = scb.corpus_bleu(
                prds_unprocessed,
                [refs_unprocessed],
                lowercase=True,
            )

            output_dict["bleu-score"] = scb_output.score  # type: ignore

        # return aggregate scores
        return output_dict
