"""Defines Builder for creating DeepA2 datasets from Entailment bank data."""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import random  # pylint: disable=duplicate-code
import shutil
import sys
from typing import List, Dict, Union

import datasets

# import jinja2
import pandas as pd
from tqdm import tqdm  # type: ignore

from deepa2 import (
    QuotedStatement,
    DeepA2Item,
)  # pylint: disable=duplicate-code
from deepa2.builder import (
    RawExample,
    PreprocessedExample,
    DatasetLoader,
    Builder,
    PipedBuilder,
    Pipeline,
    Transformer,
)  # pylint: disable=duplicate-code
from deepa2.config import (
    data_dir,
)  # pylint: disable=duplicate-code

tqdm.pandas()


@dataclasses.dataclass
class RawArgQExample(RawExample):  # pylint: disable=too-many-instance-attributes
    """
    Datatype describing a raw, unprocessed example
    in the Arg_Q_Rank dataset, possibly batched.
    """

    argument: Union[str, List[str]]
    topic: Union[str, List[str]]
    set: Union[str, List[str]]
    stance_WA: Union[str, List[str]]  # pylint: disable=invalid-name


@dataclasses.dataclass
class PreprocessedArgQExample(
    PreprocessedExample
):  # pylint: disable=too-many-instance-attributes
    """
    Datatype describing a preprocessed ArgQ example.
    """

    topic: str
    stance: int
    argument_stance_conf: str
    argument_stance_nonconf: str


class ArgQLoader(DatasetLoader):  # pylint: disable=too-few-public-methods
    """loads ArgQ raw data"""

    _ARG_Q_FILENAME = "arg_quality_rank_30k.csv"

    def __init__(self, **kwargs) -> None:
        super().__init__()
        if "path" not in kwargs:
            logging.info("No ArgQ file-path specified (via --path), assuming path is .")
        self._sourcepath: str = kwargs.get("path", f"./{self._ARG_Q_FILENAME}")

    def load_dataset(self) -> datasets.DatasetDict:

        # copy csv
        arg_q_source = pathlib.Path(self._sourcepath)
        if not arg_q_source.is_file():
            logging.error("No ArgQ file at %s found, exiting.", str(arg_q_source))
            sys.exit(-1)

        arg_q_raw = pathlib.Path(data_dir, "raw", "arg_q", "all.csv")
        arg_q_raw.parents[0].mkdir(parents=True, exist_ok=True)

        shutil.copy(arg_q_source, arg_q_raw)
        logging.info("Copied ArgQ file %s to %s.", str(arg_q_source), str(arg_q_raw))

        # load all argq data from disk and initialize dataset
        df_arg_q = pd.read_csv(arg_q_raw)
        df_arg_q = df_arg_q[["argument", "topic", "set", "stance_WA"]]
        df_arg_q.rename(columns={"stance_WA": "stance"})

        splits_mapping = {"train": "train", "dev": "validation", "test": "test"}
        dataset_dict = {}
        for split_key, target_key in splits_mapping.items():
            dataset_dict[target_key] = datasets.Dataset.from_pandas(
                df_arg_q[df_arg_q["set"] == split_key],
                preserve_index=False,
            )

        dataset_dict = datasets.DatasetDict(dataset_dict)

        return dataset_dict


class AddSourceText(Transformer):
    """add source"""

    _TEMPLATE_STRINGS = {
        "conjecture": "{{ topic_str }}? {{ stance_str }}",
        "source_text": '{{ conjecture }} {{ args | join(" ") }}',
    }
    PRO_EXPRS = ["yes!", "absolutely!", "I agree!"]
    CON_EXPRS = ["no!", "not at all!", "I disagree!"]

    def __init__(self, builder: Builder) -> None:
        super().__init__(builder)
        self._random = random.Random()
        # pylint: disable=duplicate-code

    def _generate_source(
        self,
        stance: int,
        topic: str,
        argument_stance_conf: str,
        argument_stance_nonconf: str,
        **kwargs,  # pylint: disable=unused-argument
    ):
        """generates source text"""
        topic_str = topic
        stance_exprs = self.PRO_EXPRS if stance == 1 else self.CON_EXPRS
        stance_str = self._random.choice(stance_exprs)
        args = self._random.sample(
            [
                argument_stance_conf,
                argument_stance_nonconf,
            ],
            k=2,
        )

        conjecture = self.templates["conjecture"].render(
            topic_str=topic_str,
            stance_str=stance_str,
        )
        source = self.templates["source_text"].render(
            conjecture=conjecture,
            args=args,
        )
        return source, conjecture

    def transform(  # type: ignore[override]
        self, da2_item: DeepA2Item, prep_example: PreprocessedArgQExample
    ) -> DeepA2Item:
        # print("da2_item: %s" % da2_item)
        # print("prep_example: %s" % prep_example)

        source, conjecture = self._generate_source(**dataclasses.asdict(prep_example))
        da2_item.source_text = source
        da2_item.metadata.append(("conjecture", conjecture))

        return da2_item


class AddReasons(Transformer):
    """add reasons"""

    def transform(  # type: ignore[override]
        self, da2_item: DeepA2Item, prep_example: PreprocessedArgQExample
    ) -> DeepA2Item:
        # print("da2_item: %s" % da2_item)
        # print("prep_example: %s" % prep_example)
        text = prep_example.argument_stance_conf
        text = text.strip(" .")
        reasons = [QuotedStatement(text=text, ref_reco=1)]
        da2_item.reasons = reasons
        return da2_item


class AddConjectures(Transformer):
    """adds conjectures"""

    def transform(  # type: ignore[override]
        self, da2_item: DeepA2Item, prep_example: PreprocessedArgQExample
    ) -> DeepA2Item:
        text = dict(da2_item.metadata)["conjecture"]
        conjectures = [QuotedStatement(text=text, ref_reco=3)]
        da2_item.conjectures = conjectures
        return da2_item


class ArgQBuilder(PipedBuilder):
    """builds enbank dataset"""

    @staticmethod
    def preprocess(dataset: datasets.Dataset) -> datasets.Dataset:

        df_raw = pd.DataFrame(dataset.to_pandas())
        df_raw.astype({"stance_WA": "int32"})
        # preprocess arguments
        df_raw.argument = df_raw.argument.str.strip(" .").str.lower()
        df_raw.argument = df_raw.argument.apply(
            lambda s: s + "." if s[-1] not in ["?", "!"] else s
        )
        # to each topic-stance, assigns et of cooresponding arguments
        args_by_topicstance = df_raw.groupby(["topic", "stance_WA"]).apply(
            lambda g: g["argument"].tolist()
        )

        df_raw["argument_stance_conf"] = df_raw.argument

        def sample_ca(row) -> str:
            # select all args with same topic but OPPOSITE stance
            cas = args_by_topicstance[(row["topic"], -row["stance_WA"])]
            if not cas:
                return ""
            return random.choice(cas)

        df_raw["argument_stance_nonconf"] = df_raw.apply(sample_ca, axis=1)
        df_raw["stance"] = df_raw.stance_WA
        df_raw["topic"] = df_raw["topic"].str.lower()

        # remove spare columns
        field_names = [
            field.name for field in dataclasses.fields(PreprocessedArgQExample)
        ]
        df_raw = df_raw[field_names]

        # return as Dataset
        return datasets.Dataset.from_pandas(df=df_raw, preserve_index=False)

    def _construct_pipeline(self, **kwargs) -> Pipeline:
        pipeline = Pipeline(
            [
                AddSourceText(self),
                AddReasons(self),
                AddConjectures(self),
            ]
        )
        return pipeline

    def set_input(self, batched_input: Dict[str, List]) -> None:
        prep_example = PreprocessedArgQExample.from_batch(batched_input)
        self._input = prep_example
