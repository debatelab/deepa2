"""Parsers and formatters for DA2 data structures"""

import dataclasses
import logging
import re
from typing import Any, List, Dict, Tuple, Optional, Union

import jinja2

# import ttp

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
            return self._format_dict(dict(data))

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
    label: Optional[int]  = None
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

    def parse_argdown(self, text: str) -> Optional[Argument]:
        """parses argdown text as Argument"""
        parser = ArgdownParser()
        statements=parser.parse_argdown_block(text)
        if not statements:
            return None
        argument = Argument(
            statements=statements            
        )
        return argument

    def parse_list(self, text: str):
        """parses list of statements"""
        pass

    def parse_formalization(self, text: str):
        """parses formalizations"""
        pass

    def parse_keys(self, text: str):
        """parses keys of formalization"""
        pass

class ArgdownParser:
    """parses text as Argdown"""

    INFERENCE_PATTERN_REGEX = r" ---- | -- with (?<scheme>[^\(\)]*)(?<variant> \([^-\(\))]*\))? from (?<uses>[\(\), 0-9]+) -- | -- (?<info>[^-]*) -- "

    def preprocess_ad(self, ad_raw:str) -> str:
        """preprocess argdown text"""
        ad_raw = ad_raw.replace("\n"," ")
        ad_raw = ad_raw.replace("  "," ")
        ad_raw = ad_raw.replace("with?? ","with ?? ")        
        return ad_raw

    def parse_argdown_block(self, ad_raw:str) -> List(ArgumentStatement):
        """parses argdown block"""
        # preprocess
        ad_raw = self.preprocess_ad(ad_raw)
        regex = self.INFERENCE_PATTERN_REGEX

        argument_statements: List(ArgumentStatement) = []

        # find all inferences
        matches = re.finditer(regex, ad_raw, re.MULTILINE)

        inf_args = {}
        pointer = 0
        # iterate over inferences
        for match in matches:   
            # parse all propositions before inference matched that have not been parsed before
            new_statements = self.parse_proposition_block(ad_raw[pointer:match.start()],**inf_args)
            if not new_statements:
                # if failed to parse proposition block return None
                return None
            argument_statements.extend(new_statements)
            # update pointer and inf_args to be used for parsing next propositions block
            pointer = match.end()
            inf_args = {    
                'schemes': re.split('; |, | and ',match.group("scheme")),
                'variants': re.split('; |, | and ',match.group("variant")),
                'uses': self.parse_uses(match.group("uses")),
                'inference_info': match.group(0).strip("- ")
            }
        # parse remaining propositions
        if pointer > 0:
            new_statements = self.parse_proposition_block(ad_raw[pointer:],inf_args=inf_args)
            argument_statements.extend(new_statements)

        return argument_statements


    def parse_proposition_block(self,ad_raw:str,**inf_args) -> List[ArgdownStatement]:
        """parses proposition block"""
        statement_list = []
        if not ad_raw:
            return statement_list
        # preprocess
        if ad_raw[0]!=" ":
            ad_raw = " "+ad_raw
        # match labels
        regex = r" \(([0-9]*)\) " 
        if not re.match(regex,ad_raw):
            return statement_list
        matches = re.finditer(regex, ad_raw, re.MULTILINE) 
        label = -1
        pointer = -1
        # iterate over matched labels
        for match in matches:
            # for matched label, we're adding the previous statement    
            if label>-1:
                statement = ArgumentStatement(
                    text = ad_raw[pointer:match.start()].strip(),
                    label = label
                )
                statement_list.append(statement)
            label = int(match.group(1)) # update label
            pointer = match.end() # update pointer
        if label>-1:
            # add last statement
            statement = ArgumentStatement(text=ad_raw[pointer:].strip() ,label=label)
            statement_list.append(statement)
        if statement_list and "uses" in inf_args:
            # update first statement with inference details
            statement_list[0].is_conclusion = True
            for key, value in inf_args.items():
                if hasattr(statement_list[0], key):
                    setattr(statement_list[0], key, value)

        return statement_list


                
    def parse_uses(self, uses_raw) -> List[int]:
        """parses list of labels used in an inference"""
        if not uses_raw:
            return []
        regex = r"\(([0-9]+)\)"
        matches = re.finditer(regex, str(uses_raw), re.MULTILINE)
        return [int(match.group(1)) for match in matches]
