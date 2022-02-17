"""Basic data structures"""

from abc import ABC
import dataclasses
import logging
from typing import Any, List, Dict, Tuple

import datasets


@dataclasses.dataclass
class DA2_ANGLES_MAP:  # pylint: disable=invalid-name,too-many-instance-attributes
    """maps key to DA2 features (`DeepA2Item`)"""

    s: str = "source_text"
    t: str = "title"
    g: str = "gist"
    h: str = "source_paraphrase"
    x: str = "context"
    a: str = "argdown_reconstruction"
    e: str = "erroneous_argdown"
    r: str = "reasons"
    j: str = "conjectures"
    p: str = "premises"
    i: str = "intermediary_conclusions"
    c: str = "conclusion"
    fp: str = "premises_formalized"
    fi: str = "intermediary_conclusions_formalized"
    fc: str = "conclusion_formalized"
    k: str = "plchd_substitutions"


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
            if set(split.column_names) not in (
                set(example_fields),
                set(fields_without_meta),
            ):
                logging.error(
                    "Features of dataset with examples (%s) "
                    "don't match example_class %s (%s).",
                    dataset.column_names,
                    cls,
                    example_fields,
                )
                raise ValueError(
                    "Features of dataset with examples don't match example_class."
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
        source_text: source text that informally presents the reconstructed argument
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
        reasons: a list of reason statements (verbatim quotes from `source_text`)
        conjectures: a list of conjectures (verbatim quotes from `source_text`)
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
        metadata: metadata

    """

    source_text: str = ""

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
    plchd_substitutions: List[Tuple[str, str]] = dataclasses.field(
        default_factory=lambda: []
    )

    metadata: List[Tuple[str, Any]] = dataclasses.field(default_factory=lambda: [])

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

    @staticmethod
    def angles() -> Dict[str, str]:
        """maps keys to field names of DeepA2 Item"""
        return dataclasses.asdict(DA2_ANGLES_MAP())


@dataclasses.dataclass
class GenerativeMode:
    """Generative mode da2 datastructures"""

    name: str
    target: str
    input: List[str]
    weight: float = 1.0

    @staticmethod
    def from_keys(name: str):
        """
        Tries to create GenerativeMode from keys formula `name`
        of the form

          `a+b+... => o`
        """

        if " => " not in name:
            return None

        input_str, target = name.split(" => ")[:2]

        if "+" in input_str:
            input_list = input_str.split("+")
            input_list = [s.strip() for s in input_list]
        else:
            input_list = [input_str.strip()]

        target = target.strip()

        if target not in DeepA2Item.angles() or any(
            key not in DeepA2Item.angles() for key in input_list
        ):
            return None

        mode = GenerativeMode(
            name=name,
            input=[DeepA2Item.angles()[key] for key in input_list],
            target=DeepA2Item.angles()[target],
        )

        return mode
