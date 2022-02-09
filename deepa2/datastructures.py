"""Basic data structures"""

from abc import ABC
import dataclasses
import logging
from typing import Any, List, Dict

import datasets


@dataclasses.dataclass
class BaseExample(ABC):
    """Abstract Base Example dataclass"""

    @classmethod
    def from_batch(cls, batched_data: Dict[str, List]):
        """Unbatches data and returns a BaseExample"""
        unbatched_data = {k: v[0] for k, v in batched_data.items()}
        return cls(**unbatched_data)

    @classmethod
    def check_features(cls, dataset: datasets.DatasetDict):
        """Checks whether dataset has features of examples"""
        example_fields = [field.name for field in dataclasses.fields(cls)]
        fields_without_meta = [name for name in example_fields if name != "metadata"]
        for _, split in dataset.items():
            if split.column_names not in (example_fields, fields_without_meta):
                logging.error(
                    "Features of dataset with raw examples (%s) "
                    "don't match raw_example_class (%s).",
                    dataset.column_names,
                    example_fields,
                )
                raise ValueError(
                    "Features of dataset with raw examples don't match raw_example_class."
                )


@dataclasses.dataclass
class DeepA2BaseItem(ABC):
    """Abstract BaseItem of da2 datastructures"""


@dataclasses.dataclass
class QuotedStatement(DeepA2BaseItem):
    """dataclass representing verbatim quote in da2item"""

    text: str = ""
    starts_at: int = -1
    ref_reco: int = -1


@dataclasses.dataclass
class ArgdownStatement(DeepA2BaseItem):
    """dataclass representing argdown statement in da2item"""

    text: str = ""
    explicit: Any = ""
    ref_reco: int = -1


@dataclasses.dataclass
class Formalization(DeepA2BaseItem):
    """dataclass representing formalization in da2item"""

    form: str = ""
    ref_reco: int = -1


@dataclasses.dataclass
class DeepA2Item(
    BaseExample, DeepA2BaseItem
):  # pylint: disable=too-many-instance-attributes
    """
    Dataclass defining the structure of a DeepA2 example.

    Attributes:
        argument_source: source text that informally presents the reconstructed argument
        title: telling title of the reconstructed argument
        gist: very succinct summary of the argument, main point of the argument
        source_paraphrase: a maximally clear, though conservative summary of the argument
            (i.e., leavex out distractors, focuses on one argument, leaves out redundant parts,
            syntactic streamlining, adds inference indicators, but does generally not explicate
            implicit premises)
        context: provides informal or semi-formal description of the argument's context,
            ideally an argdown snippet (without detailed reconstruction) sketching the dialectic
            neighbourhood
        argdown_reconstruction: argdown snippet with reconstruction of the argument
        erroneous_argdown: a flawed reconstruction, similar to the correct one
        reasons: a list of reason statements (verbatim quotes from `argument_source`)
        conjectures: a list of conjectures (verbatim quotes from `argument_source`)
        premises: the premises of `argdown_reconstruction`
        intermediary_conclusions: the intermediary conclusions of `argdown_reconstruction`
        conclusion: the conclusion of `argdown_reconstruction`
        premises_formalized: formalizations of the `premises`
        intermediary_conclusions_formalized: formalizations of the `intermediary_conclusions`
        conclusion_formalized: formalizations of the `conclusion`
        predicate_placeholders: placeholders in formalizations
        entity_placeholders: placeholders in formalizations
        misc_placeholders: placeholders in formalizations
        plchd_substitutions: substitutions for placeholders
        distractors: list of disctractors in àrgument_source`
        metadata: metadata

    """

    argument_source: str = ""

    title: str = ""
    gist: str = ""
    source_paraphrase: str = ""
    context: str = ""

    argdown_reconstruction: str = ""
    erroneous_argdown: str = ""

    reasons: List[QuotedStatement] = dataclasses.field(
        default_factory=lambda: [QuotedStatement()]
    )
    conjectures: List[QuotedStatement] = dataclasses.field(
        default_factory=lambda: [QuotedStatement()]
    )

    premises: List[ArgdownStatement] = dataclasses.field(default_factory=lambda: [])
    intermediary_conclusions: List[ArgdownStatement] = dataclasses.field(
        default_factory=lambda: [ArgdownStatement()]
    )
    conclusion: List[ArgdownStatement] = dataclasses.field(
        default_factory=lambda: [ArgdownStatement()]
    )

    premises_formalized: List[Formalization] = dataclasses.field(
        default_factory=lambda: [Formalization()]
    )
    intermediary_conclusions_formalized: List[Formalization] = dataclasses.field(
        default_factory=lambda: [Formalization()]
    )
    conclusion_formalized: List[Formalization] = dataclasses.field(
        default_factory=lambda: [Formalization()]
    )
    predicate_placeholders: List[str] = dataclasses.field(default_factory=lambda: [])
    entity_placeholders: List[str] = dataclasses.field(default_factory=lambda: [])
    misc_placeholders: List[str] = dataclasses.field(default_factory=lambda: [])
    plchd_substitutions: Dict[str, str] = dataclasses.field(
        default_factory=lambda: {"": ""}
    )

    distractors: List[str] = dataclasses.field(default_factory=lambda: [])
    metadata: Dict = dataclasses.field(default_factory=lambda: {"": ""})

    @classmethod
    def from_batch(cls, batched_data: Dict[str, List]):
        """Unbatches data and returns a DeepA2Item"""
        unbatched_data = {k: v[0] for k, v in batched_data.items()}

        for field in dataclasses.fields(cls):
            if field.type in [
                List[QuotedStatement],
                List[ArgdownStatement],
                List[Formalization],
            ]:
                if field.name in unbatched_data:
                    # field requires re-initialization
                    unbatched_data[field.name] = [
                        field.type.__args__[0](**item)
                        for item in unbatched_data[field.name]
                    ]

        return cls(**unbatched_data)


@dataclasses.dataclass
class GenerativeMode:
    """Generative mode da2 datastructures"""

    name: str
    target: str
    input: List[str]
    weight: float = 1.0
