"""Defines Builders for creating DeepA2 datasets from AIFdb corpora."""

from __future__ import annotations

import dataclasses
import io
import json
import logging
from pathlib import Path
import random
import re
from typing import Any, List, Dict, Union
import zipfile

import jinja2
import datasets
import networkx as nx
import requests

from deepa2datasets.core import (
    Builder,
    DatasetLoader,
    DeepA2Item,
    QuotedStatement,
    PreprocessedExample,
    RawExample,
)
from deepa2datasets.config import template_dir, package_dir


class RawAIFDBExample(RawExample):
    nodeset: Union[str, List[str]]
    text: Union[str, List[str]]
    corpus: Union[str, List[str]]


class PreprocessedAIFDBExample(PreprocessedExample):
    text: Union[str, List[str]]
    corpus: Union[str, List[str]]
    type: Union[str, List[str]]
    reasons: Union[List[str], List[Any]]
    conjectures: Union[List[str], List[Any]]
    premises: Union[List[str], List[Any]]
    conclusions: Union[List[str], List[Any]]


@dataclasses.dataclass
class AIFDBConfig:
    """configuration class for AIFdb import"""
    name: str
    cache_dir: Path
    corpora: List[str]
    splits: Dict[str, float] = dataclasses.field(
        default_factory=lambda: {"train": 0.8, "validation": 0.1, "test": 0.1}
    )
    templates_sp_ca: List[str] = dataclasses.field(
        default_factory=lambda: [
            "aifdb/source_paraphrase_ca-01.txt",
            "aifdb/source_paraphrase_ca-02.txt",
            "aifdb/source_paraphrase_ca-03.txt",
            "aifdb/source_paraphrase_ca-04.txt",
            "aifdb/source_paraphrase_ca-05.txt",
        ]
    )
    templates_sp_ra: List[str] = dataclasses.field(
        default_factory=lambda: [
            "aifdb/source_paraphrase_ra-01.txt",
            "aifdb/source_paraphrase_ra-02.txt",
        ]
    )


class AIFDBLoader(DatasetLoader):
    """loads aifdb raw data"""

    def __init__(
        self, aifdb_config: AIFDBConfig = None
    ):  # pylint: disable=super-init-not-called
        self._aifdb_config = aifdb_config

    def load_dataset(self) -> datasets.DatasetDict:
        splits = self._aifdb_config.splits

        # download and unpack corpora
        aifdb_dir = Path(self._aifdb_config.cache_dir)
        logging.info("Downloading aifdb dataset to %s ...", aifdb_dir)
        for url in self._aifdb_config.corpora:
            destination = Path(aifdb_dir, url.split("/")[-1])
            if destination.is_dir():
                logging.debug("Using cached %s.", destination)
            else:
                destination.mkdir(parents=True, exist_ok=True)
                logging.debug("Downloading %s", url)
                request = requests.get(url + "/download")
                with zipfile.ZipFile(io.BytesIO(request.content)) as zip_file:
                    zip_file.extractall(str(destination.resolve()))
                logging.debug("Saved %s to %s.", url, destination)

        # load aifdb dataset from disk
        data = {"nodeset": [], "text": [], "corpus": []}
        for corpus_dir in aifdb_dir.iterdir():
            if corpus_dir.is_dir():
                for nodefile in corpus_dir.iterdir():
                    if nodefile.suffix == ".json":
                        textfile = nodefile.parent / (nodefile.stem + ".txt")
                        if textfile.exists():
                            data["nodeset"].append(json.load(nodefile.open()))
                            data["text"].append("".join(textfile.open().readlines()))
                            data["corpus"].append(corpus_dir.name)
        dataset = datasets.Dataset.from_dict(data)

        # create train-validation-test splits
        dataset = dataset.train_test_split(
            test_size=(1 - splits["train"])
        )  # split once
        dataset_tmp = dataset["test"].train_test_split(
            test_size=(splits["test"] / (splits["test"] + splits["validation"]))
        )  # split test-split again
        dataset = datasets.DatasetDict(
            train=dataset["train"],
            validation=dataset_tmp["train"],
            test=dataset_tmp["test"],
        )

        return dataset


class Utils:
    """utilities for preprocessing AIFdb"""

    cleanr = re.compile("<.*?>")

    @staticmethod
    def cleanhtml(example):
        """cleans html in source texts"""
        example["text"] = re.sub(Utils.cleanr, "", example["text"])
        return example

    @staticmethod
    def split_nodeset_per_inference(examples: Dict[str, List]) -> Dict[str, List]:
        """extracts individual inferences from nodesets, and splits nodesets accordingly"""

        inference_chunks = {
            k: [] for k in PreprocessedAIFDBExample.__annotations__.keys()  # pylint: disable=no-member
        }
        node_type: Dict = {}
        node_text: Dict = {}
        graph = None
        # for each example
        for i, nodeset in enumerate(examples["nodeset"]):
            # initialize graph representing the argumentative analysis
            nodeset["directed"] = True
            attrs = {
                "source": "fromID",
                "target": "toID",
                "name": "nodeID",
                "key": "key",
                "link": "edges",
            }
            graph = nx.readwrite.json_graph.node_link_graph(nodeset, attrs=attrs)
            node_type = nx.get_node_attributes(graph, "type")
            # logging.debug(f"node types: {node_type}")
            node_text = nx.get_node_attributes(graph, "text")
            if not (node_type and node_text):
                logging.warning(
                    "No node types / texts in nodeset no %s in corpus %s: skipping this nodeset.",
                    i,
                    examples["corpus"][i],
                )
                continue

            # construct alternative_text by joining L-nodes
            alternative_text = [
                node_text.get(n, "")
                for n in graph.nodes
                if node_type.get(n, None) == "L"
            ]  # L-nodes
            alternative_text = " ".join(alternative_text)
            alternative_text = alternative_text.replace("  ", " ")

            # use longer text
            text = examples["text"][i]
            if len(alternative_text) > 2 * (len(text) - text.count("\n")):
                logging.debug(
                    "Using alternative text '%s' rather than original text '%s' in corpus '%s'.",
                    alternative_text,
                    text,
                    examples["corpus"][i],
                )
                text = alternative_text

            # get all nodes of type CA / RA
            inference_nodes = [
                n for n in graph.nodes if node_type.get(n, None) in ["CA", "RA"]
            ]
            # each inference node gives rise to a separate chunk
            for inference_node in inference_nodes:
                # get conclusion (ids)
                conclusions = [
                    n
                    for n in graph.successors(inference_node)
                    if node_type[n] == "I"
                ]
                # get premises (ids)
                premises = [
                    n
                    for n in graph.predecessors(inference_node)
                    if node_type[n] == "I"
                ]

                # get conjectures and reasons (ids)
                def get_L_grandparent(node):
                    if node_type[node] != "I":
                        return None
                    ya_predecessors = [
                        n for n in graph.predecessors(node) if node_type[n] == "YA"
                    ]
                    if not ya_predecessors:
                        return None
                    l_grandparents = [
                        n
                        for m in ya_predecessors
                        for n in graph.predecessors(m)
                        if node_type[n] == "L" and node_text[n] != "analyses"
                    ]
                    return l_grandparents

                conjectures = [get_L_grandparent(n) for n in conclusions]
                if conjectures:
                    # flatten
                    conjectures = [
                        x for l in conjectures if l for x in l  # noqa: E741
                    ]
                    # sort, ids correspond to location in text
                    conjectures = sorted(
                        conjectures
                    )
                reasons = [get_L_grandparent(n) for n in premises]
                if reasons:
                    # flatten
                    reasons = [x for l in reasons if l for x in l]  # noqa: E741
                    reasons = sorted(
                        reasons
                    )  # sort, ids correspond to location in text
                # subst text for ids
                conjectures = [node_text[n] for n in conjectures]
                reasons = [node_text[n] for n in reasons]
                conclusions = [node_text[n] for n in conclusions]
                premises = [node_text[n] for n in premises]
                # create new record
                inference_chunks["text"].append(text)
                inference_chunks["corpus"].append(examples["corpus"][i])
                inference_chunks["premises"].append(premises)
                inference_chunks["conclusions"].append(conclusions)
                inference_chunks["reasons"].append(reasons)
                inference_chunks["conjectures"].append(conjectures)
                inference_chunks["type"].append(node_type[inference_node])
        logging.debug(
            "Sizes of chunks: %s", {k: len(v) for k, v in inference_chunks.items()}
        )
        return inference_chunks


class AIFDBBuilder(Builder):
    """
    AIFDBBuilder preprocesses, splits, and transforms AIFdb nodesets into DeepA2 items
    """

    @staticmethod
    def preprocess(dataset: datasets.Dataset) -> datasets.Dataset:
        """preprocessed AIFdb dataset"""

        dataset = dataset.map(Utils.cleanhtml)

        dataset = dataset.map(
            Utils.split_nodeset_per_inference,
            batched=True,
            remove_columns=dataset.column_names,
        )

        return dataset

    def __init__(self, aifdb_config: AIFDBConfig) -> None:
        """
        A fresh builder instance should contain a blank product object, which is
        used in further assembly.
        """
        # check whether template files are accessible
        if not (template_dir / "aifdb").exists():
            logging.debug("Package dir: %s", package_dir)
            logging.debug("Resolve template dir: %s", template_dir)
            logging.debug("List template dir: %s", list(template_dir.glob("*")))
            err_m = f'No "aifdb" subdirectory in template_dir {template_dir.resolve()}'
            raise ValueError(err_m)
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(),
        )
        self._aifdb_config = aifdb_config

        super().__init__()

    @property
    def input(self) -> PreprocessedAIFDBExample:
        """
        The input of any builder is a proprocessed example
        """
        return self._input

    @input.setter
    def input(self, preprocessed_example: PreprocessedAIFDBExample) -> None:
        """
        Sets input for building next product.
        """
        # unbatch:
        self._input = {k: v[0] for k, v in preprocessed_example.items()}

    def configure_product(self) -> None:
        # create configuration and add empty da2 item to product
        itype = self._input["type"]
        sp_template = random.choice(
            self._aifdb_config.templates_sp_ra
            if itype == "RA"
            else self._aifdb_config.templates_sp_ca
        )
        metadata = {
            "corpus": self._input["corpus"],
            "type": itype,
            "config": {"sp_template": sp_template},
        }
        self._product.append(DeepA2Item(metadata=metadata))

    def produce_da2item(self) -> None:
        # we produce a single da2item per input only
        record = self._product[0]
        record.argument_source = self.input["text"]
        record.reason_statements = [
            QuotedStatement(text=r, starts_at=None, ref_reco=e + 1)
            for e, r in enumerate(self.input["reasons"])
        ]
        n_reas = len(record.reason_statements)
        record.conclusion_statements = [
            QuotedStatement(text=j, starts_at=None, ref_reco=n_reas + 1)
            for j in self.input["conjectures"]
        ]
        # source paraphrase
        sp_template = self._env.get_template(record.metadata["config"]["sp_template"])
        record.source_paraphrase = sp_template.render(
            premises=self.input["premises"], conclusion=self.input["conclusions"]
        )

    def postprocess_da2item(self) -> None:
        pass

    def add_metadata_da2item(self) -> None:
        pass
