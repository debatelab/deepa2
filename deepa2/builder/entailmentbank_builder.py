"""Defines Builder for creating DeepA2 datasets from Entailment bank data."""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import random
from typing import List, Dict, Union, Any
import zipfile

import datasets
import jinja2
import pandas as pd
from tqdm import tqdm  # type: ignore

from deepa2 import (
    ArgdownStatement,
    Formalization,
    QuotedStatement,
    DeepA2Item,
)
from deepa2.builder import (
    RawExample,
    PreprocessedExample,
    DatasetLoader,
    Builder,
    DownloadManager,
)
from deepa2.config import template_dir, data_dir

tqdm.pandas()


@dataclasses.dataclass
class RawEnBankExample(RawExample):
    """
    Datatype describing a raw, unprocessed example
    in a entailment bank dataset, possibly batched.
    """

    id: Union[str, List[str]]  # pylint: disable=invalid-name
    context: Union[str, List[str]]
    question: Union[int, List[int]]
    answer: Union[str, List[str]]
    hypothesis: Union[str, List[str]]
    proof: Union[str, List[str]]
    full_text_proof: Union[str, List[str]]
    depth_of_proof: Union[int, List[int]]
    length_of_proof: Union[int, List[int]]
    meta: Union[Dict[str, Any], List[Dict[str, Any]]]


@dataclasses.dataclass
class PreprocessedEnBankExample(PreprocessedExample):
    """
    Datatype describing a preprocessed entailment bank
    example.
    """

    id: str  # pylint: disable=invalid-name
    step_proof: str
    triples: Dict[str, str]
    intermediate_conclusions: Dict[str, str]
    question_text: str
    answer_text: str
    distractors: List[str]


class EnBankLoader(DatasetLoader):  # pylint: disable=too-few-public-methods
    """loads EntailmentBank raw data"""

    _ENBANK_BASE_URL = "https://drive.google.com/file/d/1EduT00qkDU6DAD-Bjgheh-o8MVbx1NZS/view?usp=sharing"  # pylint: disable=line-too-long
    _ENBANK_GDRIVE_ID = "1EduT00qkDU6DAD-Bjgheh-o8MVbx1NZS"

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._task: str = kwargs.get("name", "task_1")
        if "name" not in kwargs:
            logging.info("No EnBank task specified (via --name), using task_1.")
        if self._task not in ["task_1", "task_2"]:
            logging.info("Invalid EnBank task name %s, using task_1.", self._task)
            self._task = "task_1"

    def load_dataset(self) -> datasets.DatasetDict:

        # download and unpack corpora
        enbank_dir = pathlib.Path(data_dir, "raw", "entailment-bank")
        logging.info("Downloading entailment bank dataset to %s ...", enbank_dir)

        using_cache = False
        if enbank_dir.is_dir():
            if any(enbank_dir.iterdir()):
                logging.debug("Using cached %s.", enbank_dir)
                using_cache = True

        if not using_cache:
            enbank_dir.mkdir(parents=True, exist_ok=True)
            logging.debug("Downloading %s", self._ENBANK_BASE_URL)
            tmp_zip = pathlib.Path(enbank_dir, "enbank.zip")
            DownloadManager.download_file_from_google_drive(
                self._ENBANK_GDRIVE_ID, tmp_zip.resolve()
            )
            with zipfile.ZipFile(tmp_zip) as zip_file:
                zip_file.extractall(str(enbank_dir.resolve()))
            tmp_zip.unlink()
            logging.debug("Saved %s to %s.", self._ENBANK_BASE_URL, enbank_dir)

        # load entailment-bank dataset from disk
        splits_mapping = {"train": "train", "dev": "validation", "test": "test"}
        dataset_dict = {}
        for split_key, target_key in splits_mapping.items():
            # load task
            source_file = pathlib.Path(
                enbank_dir,
                "entailment_trees_emnlp2021_data_v2",
                "dataset",
                self._task,
                f"{split_key}.jsonl",
            )
            logging.debug("Loading local source file %s", source_file)
            dataset_dict[target_key] = datasets.Dataset.from_pandas(
                pd.read_json(source_file.resolve(), lines=True)
            )

        dataset_dict = datasets.DatasetDict(dataset_dict)

        return dataset_dict


class EnBankBuilder(Builder):
    """
    Entailment Bank Builder preprocesses and transforms
    e-SNLI records into DeepA2 items.
    """

    _TEMPLATE_STRINGS = {
        "premise": "({{ label }}) {{ premise }}",
        "conclusion": (
            "--\nwith ?? from{% for from in froml %} "
            "({{ from }}){% endfor %}\n--\n({label}) {conclusion}"
        ),
        "source_text": "{{ question_text }} {{ answer_text }}. that is because {{ statements }}",
    }

    @staticmethod
    def preprocess(dataset: datasets.Dataset) -> datasets.Dataset:

        # expand meta data
        def expand_meta(raw_example):
            return raw_example["meta"]

        dataset = dataset.map(expand_meta)

        # remove spare columns
        field_names = [
            field.name for field in dataclasses.fields(PreprocessedEnBankExample)
        ]
        spare_columns = [
            column for column in dataset.column_names if column not in field_names
        ]
        dataset = dataset.remove_columns(spare_columns)

        return dataset

    def __init__(self, **kwargs) -> None:
        """
        Initialize EnBank Builder.
        """
        super().__init__(**kwargs)
        self._input: PreprocessedEnBankExample

        self._random = random.Random()

        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(),
        )
        self._templates = {
            k: env.from_string(v) for k, v in self._TEMPLATE_STRINGS.items()
        }

    def _process_proof_step(
        self,
        proof_step: str,
        triples: str = None,
        intermediate_conclusions: str = None,
        labels: dict = None,
    ):
        """
        processes a single proof/inference step (sub-argument)
        """
        temp_split = proof_step.split(" -> ")
        conclusion = temp_split[-1]
        conclusion = conclusion.split(":")[0]
        if conclusion == "hypothesis":
            conclusion = sorted(intermediate_conclusions.keys())[-1]

        premises = temp_split[0].split(" & ")  # split antecendens

        # construct further labels
        labels = labels.copy() if labels else {}
        n_labels = len(labels)
        # construct premises and conclusions
        argdown_items = []
        i = 1
        froml = []
        for premise in premises:
            if premise[:4] == "sent":
                labels.update({premise: n_labels + i})
                i += 1
                argdown_items.append(
                    self._templates["premise"].render(
                        label=labels[premise], premise=triples[premise]
                    )
                )
            froml.append(str(labels[premise]))
        froml = ",".join(froml)

        labels.update({conclusion: len(labels) + 1})
        argdown_items.append(
            self._templates["conclusion"].render(
                label=labels[conclusion],
                froml=froml,
                conclusion=intermediate_conclusions[conclusion],
            )
        )

        return argdown_items, labels

    def _generate_argdown(
        self,
        proof: str = None,
        triples: str = None,
        intermediate_conclusions: str = None,
    ):
        """
        generates argdown and labels dict
        """
        labels = {}
        argdown_list = []
        step_list = proof.split("; ")[:-1]
        for step in step_list:
            argdown_items, labels = self._process_proof_step(
                step,
                triples=triples,
                intermediate_conclusions=intermediate_conclusions,
                labels=labels,
            )
            argdown_list = argdown_list + argdown_items
        argdown = "\n".join(argdown_list)
        return argdown, labels

    def _generate_source(
        self,
        question_text: str = None,
        answer_text: str = None,
        triples: dict = None,
        distractors: list = None,
    ):
        """generates source text"""
        if distractors is None:
            distractors = []
        statements = list(triples.keys())
        statements = self._random.sample(statements, k=len(statements))
        reason_order = [s for s in statements if s not in distractors]
        statements = [triples.get(s, s) for s in statements]
        statements = [s + "." for s in statements]
        statements = " ".join(statements)
        source = self._templates["source_text"].render(
            question_text=question_text.lower(),
            answer_text=answer_text.lower().strip("."),
            statements=statements,
        )
        return source, reason_order

    @staticmethod
    def _generate_reason_statements(
        triples: dict = None, reason_order: list = None, labels: dict = None
    ):
        """
        generates reasons
        """
        reason_s = [{"text": triples[k], "ref_reco": labels[k]} for k in reason_order]
        return reason_s

    @staticmethod
    def _generate_conjectures(
        question_text: str = None, answer_text: str = None, labels: dict = None
    ):
        """
        generates conjectures, final conclusion is sole conjectures
        all intermediary conclusions are implicit
        """
        question_text = question_text.split(". ")[-1].lower()
        answer_text = answer_text.lower().strip(".")
        text = f"{question_text} {answer_text}."
        conjectures = [{"text": text, "ref_reco": max(labels.values())}]
        return conjectures

    @property
    def input(self) -> PreprocessedEnBankExample:
        return self._input

    def set_input(self, batched_input: Dict[str, List]) -> None:
        self._input = PreprocessedEnBankExample.from_batch(batched_input)

    def configure_product(self) -> None:
        # populate product with configs
        self._product.append(DeepA2Item())

    def produce_da2item(self) -> None:
        # we produce a single da2item per input only
        record = self._product[0]
        record.argument_source = str(self.input.answer_text)
        # TODO!

    def postprocess_da2item(self) -> None:
        pass

    def add_metadata_da2item(self) -> None:
        pass
