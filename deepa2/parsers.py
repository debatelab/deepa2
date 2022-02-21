"""Parsers and formatters for DA2 data structures"""

import dataclasses
import logging
from typing import Any, List, Dict, Tuple, Optional, Union

import jinja2

# import ttp

from deepa2 import DeepA2Item, QuotedStatement, ArgdownStatement, Formalization


class DeepA2Layouter:  # pylint: disable=too-few-public-methods
    """parses and formats DeepA2Items"""

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
            return ""

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
