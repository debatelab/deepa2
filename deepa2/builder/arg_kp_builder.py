"""Defines Builder for creating DeepA2 datasets from IBM Argument Key Point data."""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import random
from typing import List, Dict, Union

import datasets

# import jinja2
import pandas as pd
from tqdm import tqdm  # type: ignore

from deepa2 import (
    ArgdownStatement,
    Formalization,
    DeepA2Item,
)
from deepa2.builder import (
    RawExample,
    PreprocessedExample,
    DatasetLoader,
    Builder,
    PipedBuilder,
    Pipeline,
    Transformer,
)

tqdm.pandas()


@dataclasses.dataclass
class RawArgKPExample(RawExample):  # pylint: disable=too-many-instance-attributes
    """
    Datatype describing a raw, unprocessed example
    in the ArgKP dataset as returned by dataset loader,
    possibly batched.
    """

    argument: Union[str, List[str]]
    topic: Union[str, List[str]]
    key_point: Union[str, List[str]]
    stance: Union[str, List[str]]  # pylint: disable=invalid-name


@dataclasses.dataclass
class PreprocessedArgKPExample(
    PreprocessedExample
):  # pylint: disable=too-many-instance-attributes
    """
    Datatype describing a preprocessed ArgKP example.
    """

    topic: str
    stance: int
    argument: str
    key_point: str


class ArgKPLoader(DatasetLoader):  # pylint: disable=too-few-public-methods
    """loads ArgKP raw data from multiple files"""

    _PATH_SPLITS = {
        "train": "kpm_data",
        "dev": "kpm_data",
        "test": "test_data",
    }

    _FILE_TEMPLATES = {
        "arguments": "arguments_{split}.csv",
        "key_points": "key_points_{split}.csv",
        "labels": "labels_{split}.csv",
    }

    def __init__(self, **kwargs) -> None:
        super().__init__()
        if "path" not in kwargs:
            logging.info(
                "No ArgKP root path specified (via --path), assuming path is ."
            )
        self._sourcepath: str = kwargs.get("path", ".")

    def load_dataset(self) -> datasets.DatasetDict:
        """loads ArgQP dat from root directory by merging multiple csv files"""
        splits_mapping = {"train": "train", "dev": "validation", "test": "test"}
        dataset_dict = {}
        for split_key, target_key in splits_mapping.items():
            logging.info("Loading %s split", split_key)
            dataset_dict[target_key] = datasets.Dataset.from_pandas(
                self.load_split(split_key),
                preserve_index=False,
            )
        dataset_dict = datasets.DatasetDict(dataset_dict)
        return dataset_dict

    def load_split(self, split: str) -> pd.DataFrame:
        """loads split as dataframe"""
        data_path = pathlib.Path(self._sourcepath, self._PATH_SPLITS[split])
        files = {
            k: pathlib.Path(v.format(split=split))
            for k, v in self._FILE_TEMPLATES.items()
        }
        # load arguments
        df_arguments = pd.read_csv(data_path / files["arguments"])
        logging.debug("Loadad arguments dataframe: %s", df_arguments.head())
        # load key_points
        df_key_points = pd.read_csv(data_path / files["key_points"])
        logging.debug("Loadad keypoints dataframe: %s", df_key_points.head())
        # load labels, which match arguments with keypoints
        df_labels = pd.read_csv(data_path / files["labels"])
        logging.debug("Loadad labels dataframe: %s", df_labels.head())
        # matching_labels
        df_labels_m = df_labels[df_labels.label == 1]
        logging.debug("Matching labels dataframe: %s", df_labels_m.head())
        # matched arguments
        df_arguments_m = df_arguments[
            df_arguments.arg_id.isin(df_labels_m.arg_id.tolist())
        ]
        if len(df_arguments) != len(df_arguments_m):
            logging.info(
                "Removing %s arguments without keypoints.",
                len(df_arguments) - len(df_arguments_m),
            )
            df_arguments = df_arguments_m

        def append_keypoints(row: pd.Series) -> pd.DataFrame:
            """returns df with all keypoints for given argument"""
            # get all keypoints
            arg_id = row.arg_id.iloc[0]
            key_point_ids = df_labels_m[df_labels_m.arg_id == arg_id][
                "key_point_id"
            ].tolist()
            df_key_points_sel = df_key_points[
                df_key_points.key_point_id.isin(key_point_ids)
            ]
            df_key_points_sel["argument"] = row.argument.iloc[0]
            return df_key_points_sel

        # append keypoints to each argument
        df_merged = pd.DataFrame(
            df_arguments.groupby("arg_id", as_index=False).progress_apply(
                append_keypoints
            )
        )
        # drop unnecessary columns & reindex
        df_merged = df_merged[["argument", "topic", "key_point", "stance"]]
        df_merged = df_merged.reset_index(drop=True)

        return df_merged


class AddContext(Transformer):
    """adds context and gist"""

    _TEMPLATE_STRINGS = {
        "context": "{{ topic_str }}? {{ stance_str }}",
    }
    PRO_EXPRS = ["yes!", "absolutely!", "I agree!"]
    CON_EXPRS = ["no!", "not at all!", "I disagree!"]

    def __init__(self, builder: Builder) -> None:
        super().__init__(builder)
        self._random = random.Random()

    def _generate_context(
        self,
        stance: int,
        topic: str,
        **kwargs,  # pylint: disable=unused-argument
    ):
        """generates context"""
        topic_str = topic
        stance_exprs = self.PRO_EXPRS if stance == 1 else self.CON_EXPRS
        stance_str = self._random.choice(stance_exprs)
        context = self.templates["context"].render(
            topic_str=topic_str,
            stance_str=stance_str,
        )
        return context

    def transform(  # type: ignore[override]
        self, da2_item: DeepA2Item, prep_example: PreprocessedArgKPExample
    ) -> DeepA2Item:
        # print("da2_item: %s" % da2_item)
        # print("prep_example: %s" % prep_example)
        context = self._generate_context(**dataclasses.asdict(prep_example))
        da2_item.context = context
        da2_item.gist = prep_example.key_point
        return da2_item


class AddSourceText(Transformer):
    """add source_text"""

    _TEMPLATE_STRINGS = {
        "source_text": "{{ topic_str }}? {{ argument_str }}",
    }

    def _generate_source(
        self,
        argument: str,
        topic: str,
        **kwargs,  # pylint: disable=unused-argument
    ):
        """generates source"""
        source = self.templates["source_text"].render(
            topic_str=topic,
            argument_str=argument,
        )
        return source

    def transform(  # type: ignore[override]
        self, da2_item: DeepA2Item, prep_example: PreprocessedArgKPExample
    ) -> DeepA2Item:
        # print("da2_item: %s" % da2_item)
        # print("prep_example: %s" % prep_example)
        source = self._generate_source(**dataclasses.asdict(prep_example))
        da2_item.source_text = source
        return da2_item


class AddArgument(Transformer):
    """adds premisses, conclusion, argdown, formalizations"""

    _TEMPLATE_STRINGS = {
        "negation": "it is not the case that {{ topic_str }}.",
        "connecting_premise": "if {{ key_point_str }} then {{ conclusion_str }}.",
        "argdown": "(1) {{ main_premise }} (2) {{ connecting_premise }} "
        "-- with modus ponens from (1) (2) -- (3) {{ conclusion }}",
    }

    def transform(  # type: ignore[override]
        self, da2_item: DeepA2Item, prep_example: PreprocessedArgKPExample
    ) -> DeepA2Item:
        # use topic (possibly negated if stance == -1) as conclusion
        topic_str = prep_example.topic.lower().strip(". ")
        if prep_example.stance == 1:
            ctext = f"{topic_str}."
        else:
            ctext = self.templates["negation"].render(topic_str=topic_str)
        conclusion = [ArgdownStatement(text=ctext, ref_reco=3)]

        # use keypoint as first premise, add connecting premise
        premises = [
            ArgdownStatement(text=prep_example.key_point, ref_reco=1),
            ArgdownStatement(
                text=self.templates["connecting_premise"].render(
                    key_point_str=prep_example.key_point.strip("."),
                    conclusion_str=ctext.strip("."),
                ),
                ref_reco=2,
            ),
        ]

        # add argdown
        argdown = self.templates["argdown"].render(
            main_premise=premises[0].text,
            connecting_premise=premises[1].text,
            conclusion=conclusion[0].text,
        )

        da2_item.conclusion = conclusion
        da2_item.premises = premises
        da2_item.argdown_reconstruction = argdown

        return da2_item


class AddFormalization(Transformer):
    """adds premisses, conclusion, argdown, formalizations"""

    _TEMPLATE_STRINGS = {
        "negation": "not {{ s }}",
        "connecting_premise": "p -> {{ s }}",
    }

    def transform(  # type: ignore[override]
        self, da2_item: DeepA2Item, prep_example: PreprocessedArgKPExample
    ) -> DeepA2Item:
        # formalize main premise
        p = "p"  # pylint: disable=invalid-name
        # formalize conclusion
        q = "q"  # pylint: disable=invalid-name
        if prep_example.stance == -1:
            q = self.templates["negation"].render(s=q)  # pylint: disable=invalid-name

        da2_item.premises_formalized = [
            Formalization(form=p, ref_reco=1),
            Formalization(
                form=self.templates["connecting_premise"].render(s=q),
                ref_reco=2,
            ),
        ]
        da2_item.conclusion_formalized = [Formalization(form=q, ref_reco=3)]
        da2_item.misc_placeholders = ["p", "q"]
        da2_item.plchd_substitutions = [
            ("p", prep_example.key_point.strip(".")),
            ("q", prep_example.topic.strip(".")),
        ]

        return da2_item


class ArgKPBuilder(PipedBuilder):
    """builds ArgKP dataset"""

    @staticmethod
    def preprocess(dataset: datasets.Dataset) -> datasets.Dataset:

        df_raw = pd.DataFrame(dataset.to_pandas())
        # stance
        df_raw.astype({"stance": "int32"})
        # arguments
        df_raw.argument = df_raw.argument.str.strip(" .").str.lower()
        df_raw.argument = df_raw.argument.apply(
            lambda s: s + "." if s[-1] not in ["?", "!"] else s
        )
        # arguments
        df_raw.key_point = df_raw.key_point.str.strip(" .").str.lower()
        df_raw.key_point = df_raw.key_point.apply(
            lambda s: s + "." if s[-1] not in ["?", "!"] else s
        )
        # topics
        df_raw["topic"] = df_raw["topic"].str.lower()

        # remove spare columns
        field_names = [
            field.name for field in dataclasses.fields(PreprocessedArgKPExample)
        ]
        df_raw = df_raw[field_names]

        # return as Dataset
        return datasets.Dataset.from_pandas(df=df_raw, preserve_index=False)

    def _construct_pipeline(self, **kwargs) -> Pipeline:
        pipeline = Pipeline(
            [
                AddSourceText(self),
                AddContext(self),
                AddArgument(self),
                AddFormalization(self),
            ]
        )
        return pipeline

    def set_input(self, batched_input: Dict[str, List]) -> None:
        prep_example = PreprocessedArgKPExample.from_batch(batched_input)
        self._input = prep_example
