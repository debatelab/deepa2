"""Defines Builder for creating DeepA2 datasets from Entailment bank data."""

from __future__ import annotations

import dataclasses
import logging
import pathlib
import random
from typing import List, Dict, Union, Any
import zipfile

import datasets

# import jinja2
import pandas as pd
from tqdm import tqdm  # type: ignore

from deepa2 import (
    ArgdownStatement,
    QuotedStatement,
    DeepA2Item,
)
from deepa2.builder import (
    RawExample,
    PreprocessedExample,
    DatasetLoader,
    Builder,
    DownloadManager,
    PipedBuilder,
    Pipeline,
    Transformer,
)
from deepa2.config import (
    # template_dir,
    data_dir,
)

tqdm.pandas()


@dataclasses.dataclass
class RawEnBankExample(RawExample):  # pylint: disable=too-many-instance-attributes
    """
    Datatype describing a raw, unprocessed example
    in a entailment bank dataset, possibly batched.
    """

    id: Union[str, List[str]]  # pylint: disable=invalid-name
    context: Union[str, List[str]]
    question: Union[str, List[str]]
    answer: Union[str, List[str]]
    hypothesis: Union[str, List[str]]
    proof: Union[str, List[str]]
    full_text_proof: Union[str, List[str]]
    depth_of_proof: Union[str, List[str]]
    length_of_proof: Union[str, List[str]]
    meta: Union[Dict[str, Any], List[Dict[str, Any]]]


@dataclasses.dataclass
class PreprocessedEnBankExample(
    PreprocessedExample
):  # pylint: disable=too-many-instance-attributes
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
    hypothesis: str
    core_concepts: List[str]
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


class AddArgdown(Transformer):
    """adds argdown"""

    _TEMPLATE_STRINGS = {
        "premise": "({{ label }}) {{ premise }}",
        "conclusion": (
            "--\nwith ?? from{% for from in froml %} "
            "({{ from }}){% endfor %}\n--\n({{ label }}) {{ conclusion }}"
        ),
    }

    def _process_proof_step(
        self,
        proof_step: str,
        triples: Dict[str, str],
        intermediate_conclusions: Dict[str, str],
        labels: Dict[str, int],
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
                    self.templates["premise"].render(
                        label=labels[premise], premise=triples[premise]
                    )
                )
            froml.append(str(labels[premise]))

        labels.update({conclusion: len(labels) + 1})
        argdown_items.append(
            self.templates["conclusion"].render(
                label=labels[conclusion],
                froml=froml,
                conclusion=intermediate_conclusions[conclusion],
            )
        )

        return argdown_items, labels

    def _generate_argdown(
        self,
        step_proof: str = None,
        triples: Dict[str, str] = None,
        intermediate_conclusions: Dict[str, str] = None,
        **kwargs,
    ):
        """
        generates argdown and labels dict
        """

        labels: Dict[str, int] = {}
        argdown_list: List[str] = []
        if step_proof is None:
            step_proof = ""
            logging.warning("Empty proof: %s", kwargs)
        if triples is None:
            triples = {}
            logging.warning("Empty triples: %s", kwargs)
        if intermediate_conclusions is None:
            intermediate_conclusions = {}
            logging.warning("Empty interm_conclusions: %s", kwargs)
        step_list = step_proof.split("; ")[:-1]
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

    def transform(  # type: ignore[override]
        self, da2_item: DeepA2Item, prep_example: PreprocessedEnBankExample
    ) -> DeepA2Item:
        argdown, labels = self._generate_argdown(**dataclasses.asdict(prep_example))
        da2_item.argdown_reconstruction = argdown
        da2_item.metadata.append(("labels", labels))
        return da2_item


class AddSourceText(Transformer):
    """add source"""

    _TEMPLATE_STRINGS = {
        "source_text": (
            "{{ answer_text }}. " 'that is because {{ statements | join(" ") }}'
        ),
    }

    def __init__(self, builder: Builder) -> None:
        super().__init__(builder)
        self._random = random.Random()

    def _generate_source(
        self,
        triples: Dict[str, str],
        question_text: str,
        answer_text: str,
        distractors: List[str],
        **kwargs,
    ):
        """generates source text"""
        if triples is None:
            triples = {}
            logging.warning("Empty triples: %s", kwargs)
        if question_text is None:
            question_text = ""
            logging.warning("Empty question_text: %s", kwargs)
        if answer_text is None:
            answer_text = ""
            logging.warning("Empty answer_text: %s", kwargs)
        if distractors is None:
            distractors = []
        statements = list(k for k, _ in triples.items())
        statements = self._random.sample(statements, k=len(statements))
        reason_order = [s for s in statements if s not in distractors]
        statements = [triples.get(s, s) for s in statements]
        if statements:
            statements = [s + "." if s else "" for s in statements]
        source = self.templates["source_text"].render(
            question_text=question_text.lower(),
            answer_text=answer_text.lower().strip("."),
            statements=statements,
        )
        return source, reason_order

    def transform(  # type: ignore[override]
        self, da2_item: DeepA2Item, prep_example: PreprocessedEnBankExample
    ) -> DeepA2Item:
        # print("da2_item: %s" % da2_item)
        # print("prep_example: %s" % prep_example)

        source, reason_order = self._generate_source(**dataclasses.asdict(prep_example))
        da2_item.source_text = source
        da2_item.metadata.append(("reason_order", reason_order))
        return da2_item


class AddReasons(Transformer):
    """add reasons"""

    def transform(  # type: ignore[override]
        self, da2_item: DeepA2Item, prep_example: PreprocessedEnBankExample
    ) -> DeepA2Item:
        # print("da2_item: %s" % da2_item)
        # print("prep_example: %s" % prep_example)
        reasons = [
            QuotedStatement(
                text=prep_example.triples[k],
                ref_reco=dict(da2_item.metadata)["labels"][k],
            )
            for k in dict(da2_item.metadata)["reason_order"]
        ]
        da2_item.reasons = reasons
        return da2_item


class AddConjectures(Transformer):
    """adds conjectures"""

    def transform(  # type: ignore[override]
        self, da2_item: DeepA2Item, prep_example: PreprocessedEnBankExample
    ) -> DeepA2Item:
        question_text = prep_example.question_text.split(". ")[-1].lower()
        answer_text = prep_example.answer_text.lower().strip(".")
        text = f"{question_text} {answer_text}."
        conjectures = [
            QuotedStatement(
                text=text, ref_reco=max(dict(da2_item.metadata)["labels"].values())
            )
        ]
        da2_item.conjectures = conjectures
        return da2_item


class AddPremisesConclusion(Transformer):
    """adds premises and conclusion"""

    def transform(  # type: ignore[override]
        self, da2_item: DeepA2Item, prep_example: PreprocessedEnBankExample
    ) -> DeepA2Item:
        premises = (
            [
                ArgdownStatement(text=reason.text, ref_reco=reason.ref_reco)
                for reason in da2_item.reasons
            ]
            if da2_item.reasons
            else []
        )
        conclusion_text = list(prep_example.intermediate_conclusions.values())[-1]
        conclusion_text = conclusion_text.strip(".") + "."
        conclusion = [
            ArgdownStatement(
                text=conclusion_text,
                ref_reco=max(dict(da2_item.metadata)["labels"].values()),
            )
        ]
        da2_item.premises = premises
        da2_item.conclusion = conclusion
        return da2_item


class AddParaphrase(Transformer):
    """adds source paraphrase, gist, and other fields"""

    _TEMPLATE_STRINGS = {
        "paraphrase": '{{ reasons | join(" ") }} Therefore: {{ answer_text }}.',
    }

    def __init__(self, builder: Builder) -> None:
        super().__init__(builder)
        # initialize Random generator
        self._random = random.Random()

    def transform(  # type: ignore[override]
        self, da2_item: DeepA2Item, prep_example: PreprocessedEnBankExample
    ) -> DeepA2Item:
        reasons = (
            [reason.text for reason in da2_item.reasons] if da2_item.reasons else []
        )
        paraphrase = self.templates["paraphrase"].render(
            reasons=reasons, answer_text=prep_example.answer_text
        )

        da2_item.source_paraphrase = paraphrase
        da2_item.gist = prep_example.hypothesis
        da2_item.context = prep_example.question_text
        if prep_example.core_concepts:
            da2_item.title = self._random.choice(prep_example.core_concepts)

        return da2_item


class EnBankBuilder(PipedBuilder):
    """builds enbank dataset"""

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

    def _construct_pipeline(self, **kwargs) -> Pipeline:
        pipeline = Pipeline(
            [
                AddArgdown(self),
                AddSourceText(self),
                AddReasons(self),
                AddConjectures(self),
                AddPremisesConclusion(self),
                AddParaphrase(self),
            ]
        )
        return pipeline

    def set_input(self, batched_input: Dict[str, List]) -> None:
        prep_example = PreprocessedEnBankExample.from_batch(batched_input)
        # strip dicts of None values
        prep_example.triples = {k: v for k, v in prep_example.triples.items() if v}
        prep_example.intermediate_conclusions = {
            k: v for k, v in prep_example.intermediate_conclusions.items() if v
        }
        self._input = prep_example
