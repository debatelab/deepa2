"""Parsers and formatters for DA2 data structures"""

import dataclasses
import logging
import re
from typing import Any, List, Dict, Tuple, Optional, Union

import jinja2
from nltk.sem.logic import Expression, LogicalExpressionException  # type: ignore
from ttp import ttp  # type: ignore

from deepa2 import DeepA2Item, QuotedStatement, ArgdownStatement, Formalization


class DeepA2Layouter:  # pylint: disable=too-few-public-methods
    """formats DeepA2Items"""

    _IGNORED_FIELDS = ["metadata", "distractors"]

    _TEMPLATE_STRINGS = {
        QuotedStatement: "{{ text }} (ref: ({{ ref_reco }}))",
        ArgdownStatement: "{{ text }} (ref: ({{ ref_reco }}))",
        Formalization: "{{ form }} (ref: ({{ ref_reco }}))",
    }

    _LIST_SEPARATOR = " | "

    def __init__(self) -> None:
        """initialize DeepA2Parser"""

        # compile templates
        env = jinja2.Environment()
        self._templates = {
            k: env.from_string(v) for k, v in self._TEMPLATE_STRINGS.items()
        }

    @staticmethod
    def list_seperator() -> str:
        """returns _LIST_SEPARATOR"""
        return DeepA2Layouter._LIST_SEPARATOR

    @staticmethod
    def template_strings() -> Dict[Any, str]:
        """returns _TEMPLATE_STRINGS"""
        return DeepA2Layouter._TEMPLATE_STRINGS

    def _format_field(  # pylint: disable=too-many-return-statements
        self, data: Any, field: dataclasses.Field
    ) -> Optional[str]:
        """formats field"""
        if data is None:
            return None

        if not data:
            return " "

        if field.type == Union[str, None]:
            return data

        if field.type == Union[List[str], None]:
            return self._format_list(data)

        if field.type in [
            List[Tuple[str, str]],
            Union[List[Tuple[str, str]], None],
        ]:
            try:
                dictdata = dict(data)
            except Exception as exc:
                logging.error(
                    "DeepA2Layouter couldn't format data %s in field %s as dict",
                    data,
                    field,
                )
                logging.error(exc)
                raise ValueError(
                    f"DeepA2Layouter couldn't format data {data} in field {field} as dict"
                ) from exc
            return self._format_dict(dictdata)

        if field.type in [
            Union[List[QuotedStatement], None],
            Union[List[ArgdownStatement], None],
            Union[List[Formalization], None],
        ]:
            template = self._get_template(data)
            if template:  # pylint: disable=no-else-return
                da2list = [
                    template.render(**dataclasses.asdict(item))
                    for item in data
                    if dataclasses.asdict(item).get("text")
                    or dataclasses.asdict(item).get("form")
                ]
                return self._format_list(da2list)
            else:
                logging.warning("DeepA2Layouter no template found.")

        logging.warning("DeepA2Layouter couldn't format field %s", field)
        return "not-formatted"

    def _get_template(self, data: List) -> Optional[jinja2.Template]:
        """fetches template for DeepA2Item field"""
        template = self._templates.get(data[0].__class__)
        return template

    def _format_list(self, da2list: List[str]) -> str:
        """formats a list of strings"""
        formatted = " "
        if da2list:
            if len(da2list) == 1:
                formatted = da2list[0]
            else:
                formatted = self._LIST_SEPARATOR.join(da2list)
        return formatted

    def _format_dict(self, da2dict: Dict[str, str]) -> str:
        """formats a dict"""
        da2list = [f"{k} : {v}" for k, v in da2dict.items()]
        return self._format_list(da2list)

    def format(self, da2_item: DeepA2Item) -> Dict[str, Optional[str]]:
        """formats DeepA2Item fields as strings"""
        da2_formatted = {
            field.name: self._format_field(
                data=getattr(da2_item, field.name), field=field
            )
            for field in dataclasses.fields(da2_item)
            if field.name not in self._IGNORED_FIELDS
        }
        return da2_formatted


@dataclasses.dataclass
class ArgumentStatement:
    """dataclass representing a statement in an argument

    fields:
        text: str - the text of the statement
        label: int - the label of the statement
        is_conclusion: bool - whether the statement is a conclusion
        uses: List[int] - the ids of the statements the statement is inferred from
        inference_info: str - information about the inference (not parsed)
        schemes: List[str] - the schemes used to infer the statement
        variants: List[str] - the variants of the schemes used to infer the statement
    """

    text: Optional[str] = None
    is_conclusion: bool = False
    label: Optional[int] = None
    uses: Optional[List[int]] = None
    inference_info: Optional[str] = None
    schemes: Optional[List[str]] = None
    variants: Optional[List[str]] = None


@dataclasses.dataclass
class Argument:
    """dataclass representing an argument"""

    statements: List[ArgumentStatement] = dataclasses.field(default_factory=list)


class DeepA2Parser:
    """parses text as DeepA2Items"""

    @staticmethod
    def parse_argdown(text: str) -> Optional[Argument]:
        """parses argdown text as Argument"""
        parser = ArgdownParser()
        statements = parser.parse_argdown_block(text)
        if not statements:
            return None
        argument = Argument(statements=statements)
        return argument

    @staticmethod
    def parse_list(text: str):
        """parses list of statements"""

    @staticmethod
    def parse_quotes(text: str) -> Optional[List[Optional[QuotedStatement]]]:
        """parses formalizations"""
        parser = QuotationsParser()
        quotes = parser.parse_quotes(text)
        return quotes

    @staticmethod
    def parse_formalization(text: str) -> Optional[List[Optional[Formalization]]]:
        """parses formalizations"""
        parser = FormulaeParser()
        formalizations = parser.parse_formalizations(text)
        return formalizations

    @staticmethod
    def parse_keys(text: str) -> Optional[List[Optional[Tuple[str, str]]]]:
        """parses keys of formalization"""
        parser = KeysParser()
        keys = parser.parse_keys(text)
        return keys

    @staticmethod
    def parse_as_folf(
        formalizations: List[Optional[Formalization]] = None,
    ) -> Optional[List[Any]]:
        """parses formalizations as first-order-logic formulae"""
        parser = FOLParser()
        folfs = parser.parse(formalizations)
        return folfs


class FOLParser:
    """parser methods for first-order-logic formulae
    based on NLTK parser"""

    @staticmethod
    def parse(
        formalizations: List[Optional[Formalization]] = None,
    ) -> Optional[List[Optional[Expression]]]:
        """parses da2 formalizations as NLTK first-order-logic formulae"""
        # preprocess
        preprocessed_forms = FOLParser._preprocess(formalizations)
        print(preprocessed_forms)
        expressions = []
        for pp_form in preprocessed_forms:
            try:
                expressions.append(Expression.fromstring(pp_form))
            except LogicalExpressionException:
                expressions.append(None)
        return expressions

    @staticmethod
    def _preprocess(formalizations: List[Optional[Formalization]] = None) -> List[str]:
        """preprocesses formalizations"""
        if not formalizations:
            return []
        preprocessed_forms = []
        for formalization in formalizations:
            if formalization:
                formula = FOLParser._da2_to_nltk_format(formalization.form)
                preprocessed_forms.append(formula)
            else:
                preprocessed_forms.append("")
        return preprocessed_forms

    @staticmethod
    def _da2_to_nltk_format(form: str) -> str:
        """converts DeepA2 formalization to NLTK format"""
        formula = form

        # 1. reformat quantifiers
        # find all subsequences with quantifiers
        regex = r"(?:\s?\([E]?.\)\s?)+:"
        matches = re.finditer(regex, formula, re.MULTILINE)
        tmp = ""
        pointer = 0
        count_brackets_to_close = 0
        for match in matches:
            tmp += formula[pointer : match.start()]
            tmp += FOLParser.replace_quantifiers(match.group())
            # if formula doesn't continue with opening bracket, add one,
            # assuming that the remainder of the formula belongs to scope of
            # these quantors
            if not formula[match.end() :].strip().startswith("("):
                tmp += "("
                count_brackets_to_close += 1
            pointer = match.end()
        tmp += formula[pointer:]
        tmp += ")" * count_brackets_to_close
        formula = tmp

        # 2. reformat predicative subsentences
        # find all atomic predicative formulas
        regex = r"[A-Z](\s[a-uw-z])+(?:\s|\)|$)"
        matches = re.finditer(regex, formula, re.MULTILINE)
        tmp = ""
        pointer = 0
        for match in matches:
            tmp += formula[pointer : match.start()]
            tmp += FOLParser.format_predicative_f(match.group())
            pointer = match.end()
        tmp += formula[pointer:]
        formula = tmp

        # 3. replace junctors
        formula = formula.replace("v", "|")
        formula = formula.replace("not", "!")

        return formula

    @staticmethod
    def replace_quantifiers(quantifiers: str):
        """Replaces individual quantifiers in a sequence of quantifiers"""
        new_s = ""
        # find all individual quantifiers
        regex = r"\(([E]?)(.)\)"
        matches = re.finditer(regex, quantifiers, re.MULTILINE)
        for match in matches:
            quantifier = "exists" if bool(match.group(1)) else "all"
            variable = match.group(2)
            new_s += f"{quantifier} {variable}."

        return new_s

    @staticmethod
    def format_predicative_f(predicative_f: str):
        """formats predicative formula, e.g. 'A x y' to 'A(x,y)'"""
        predicate = predicative_f[0]
        arguments = predicative_f[1:]
        arguments = arguments.strip()
        arguments_list = arguments.split(" ")
        arguments = ",".join(arguments_list)
        new_f = f"{predicate}({arguments})"
        return new_f


class QuotationsParser:
    """parses text as list of quotes"""

    @staticmethod
    def parse_quotes(text: str) -> Optional[List[Optional[QuotedStatement]]]:
        """tries to parse text as list of Formalizations"""
        if text is None:
            return None
        if not text:
            return []
        sep = DeepA2Layouter.list_seperator()
        qtexts = text.split(sep)
        quotes = []
        for qtext in qtexts:
            quotes.append(QuotationsParser.parse_quotetext(qtext))
        return quotes

    @staticmethod
    def parse_quotetext(text: str) -> Optional[QuotedStatement]:
        """tries to parse text as a single quote"""

        template = "{{ text  | ORPHRASE }} (ref: ({{ ref_reco }}))"
        text = text.strip()

        parser = ttp(text, template)
        parser.parse()
        parse_results = parser.result()[0][0]
        if not ("text" in parse_results and "ref_reco" in parse_results):
            return None
        try:
            ref_reco = int(parse_results["ref_reco"])
        except ValueError:
            ref_reco = -1
        quote = QuotedStatement(text=parse_results["text"], ref_reco=ref_reco)
        return quote


class KeysParser:
    """parses text as list of keys"""

    @staticmethod
    def parse_keys(text: str) -> Optional[List[Optional[Tuple[str, str]]]]:
        """tries to parse text as list of keys"""
        if text is None:
            return None
        if not text:
            return []
        sep = DeepA2Layouter.list_seperator()
        ktexts = text.split(sep)
        keys = []
        for ktext in ktexts:
            keys.append(KeysParser.parse_key(ktext))
        return keys

    @staticmethod
    def parse_key(text: str) -> Optional[Tuple[str, str]]:
        """tries to parse text as a key:value pair"""
        if ":" not in text:
            return None
        text = text.strip()
        # split at first colon
        key, value = text.split(":", 1)
        return key.strip(), value.strip()


class FormulaeParser:
    """parses text as list of formalizations"""

    @staticmethod
    def parse_formalizations(text: str) -> Optional[List[Optional[Formalization]]]:
        """tries to parse text as list of Formalizations"""
        if text is None:
            return None
        if not text:
            return []
        sep = DeepA2Layouter.list_seperator()
        formulas = text.split(sep)
        formalizations = []
        for text_f in formulas:
            formalizations.append(FormulaeParser.parse_formula(text_f))
        return formalizations

    @staticmethod
    def parse_formula(text: str) -> Optional[Formalization]:
        """tries to parse text as a single formalizations"""

        template = "{{ form  | ORPHRASE }} (ref: ({{ ref_reco }}))"
        text = text.strip()

        parser = ttp(text, template)
        parser.parse()
        parse_results = parser.result()[0][0]
        if not ("form" in parse_results and "ref_reco" in parse_results):
            return None

        # HACK: simple check whether string represents a formula
        words = re.split(r"\W+", parse_results["form"])
        is_formula = all((len(w) <= 1 or w == "not") for w in words)
        if not is_formula:
            return None
        try:
            ref_reco = int(parse_results["ref_reco"])
        except ValueError:
            ref_reco = -1
        return Formalization(form=parse_results["form"], ref_reco=ref_reco)


class ArgdownParser:
    """parses text as Argdown"""

    INFERENCE_PATTERN_REGEX = (
        r" ---- |"
        r" -- with (?P<scheme>[^\(\)]*)(?P<variant> \([^-\(\))]*\))?"
        r" from (?P<uses>[\(\), 0-9]+) -- |"
        r" -- (?P<info>[^-]*) -- "
    )

    @staticmethod
    def preprocess_ad(ad_raw: str) -> str:
        """preprocess argdown text"""
        ad_raw = ad_raw.replace("\n", " ")
        ad_raw = re.sub(r"\s{2,}", " ", ad_raw)
        ad_raw = ad_raw.replace("with?? ", "with ?? ")
        return ad_raw

    def parse_argdown_block(self, ad_raw: str) -> Optional[List[ArgumentStatement]]:
        """parses argdown block"""
        # preprocess
        ad_raw = self.preprocess_ad(ad_raw)
        regex = self.INFERENCE_PATTERN_REGEX

        argument_statements = []

        # find all inferences
        matches = re.finditer(regex, ad_raw, re.MULTILINE)

        inf_args: Dict[str, Any] = {}
        pointer = 0
        # iterate over inferences
        for match in matches:
            # parse all propositions before inference matched that have not been parsed before
            new_statements = self.parse_proposition_block(
                ad_raw[pointer : match.start()], **inf_args
            )
            if not new_statements:
                # if failed to parse proposition block return None
                return None
            argument_statements.extend(new_statements)
            # update pointer and inf_args to be used for parsing next propositions block
            pointer = match.end()
            schemes = match.group("scheme")
            variants = match.group("variant")
            inference_info = match.group(0)
            inf_args = {
                "schemes": re.split("; |, | and ", schemes) if schemes else None,
                "variants": re.split("; |, | and ", variants) if variants else None,
                "uses": self.parse_uses(match.group("uses")),
                "inference_info": inference_info.strip("- ")
                if inference_info
                else None,
            }
        # parse remaining propositions
        if pointer > 0:
            new_statements = self.parse_proposition_block(ad_raw[pointer:], **inf_args)
            argument_statements.extend(new_statements)

        return argument_statements

    @staticmethod
    def parse_proposition_block(ad_raw: str, **inf_args) -> List[ArgumentStatement]:
        """parses proposition block"""
        statement_list: List[ArgumentStatement] = []
        if not ad_raw:
            return statement_list
        # preprocess
        if ad_raw[0] != " ":
            ad_raw = " " + ad_raw
        # match labels
        regex = r" \(([0-9]*)\) "
        if not re.match(regex, ad_raw):
            return statement_list
        matches = re.finditer(regex, ad_raw, re.MULTILINE)
        label = -1
        pointer = -1
        # iterate over matched labels
        for match in matches:
            # for matched label, we're adding the previous statement
            if label > -1:
                statement = ArgumentStatement(
                    text=ad_raw[pointer : match.start()].strip(), label=label
                )
                statement_list.append(statement)
            label = int(match.group(1))  # update label
            pointer = match.end()  # update pointer
        if label > -1:
            # add last statement
            statement = ArgumentStatement(text=ad_raw[pointer:].strip(), label=label)
            statement_list.append(statement)
        if statement_list and "uses" in inf_args:
            # update first statement with inference details
            statement_list[0].is_conclusion = True
            for key, value in inf_args.items():
                if hasattr(statement_list[0], key):
                    setattr(statement_list[0], key, value)

        return statement_list

    @staticmethod
    def parse_uses(uses_raw) -> List[int]:
        """parses list of labels used in an inference"""
        if not uses_raw:
            return []
        regex = r"\(([0-9]+)\)"
        matches = re.finditer(regex, str(uses_raw), re.MULTILINE)
        return [int(match.group(1)) for match in matches]
