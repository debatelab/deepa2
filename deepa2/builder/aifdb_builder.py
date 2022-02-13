"""Defines Builders for creating DeepA2 datasets from AIFdb corpora."""

from __future__ import annotations

import dataclasses
import io
import json
import logging
from pathlib import Path
import random
import re
from typing import List, Dict, Union, Any
import zipfile

import jinja2
import datasets
import networkx as nx  # type: ignore
import requests

from deepa2.builder.core import (
    Builder,
    DatasetLoader,
    DeepA2Item,
    QuotedStatement,
    PreprocessedExample,
    RawExample,
)
from deepa2.config import template_dir, package_dir, data_dir


@dataclasses.dataclass
class RawAIFDBExample(RawExample):
    """dataclass of raw aifdb example"""

    nodeset: Union[Dict[str, Any], List[Dict[str, Any]]]
    text: Union[str, List[str]]
    corpus: Union[str, List[str]]


@dataclasses.dataclass
class PreprocessedAIFDBExample(PreprocessedExample):
    """dataclass of preprocessed aifdb example"""

    text: str
    corpus: str
    type: str
    reasons: List[str]
    conjectures: List[str]
    premises: List[str]
    conclusions: List[str]


@dataclasses.dataclass
class AIFDBConfig:
    """configuration class for AIFdb import"""

    name: str
    cache_dir: Path = dataclasses.field(default_factory=lambda: data_dir)
    corpora: List[str] = dataclasses.field(default_factory=lambda: [])
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

    def __post_init__(
        self,
    ):
        if self.name == "moral-maze":
            self.cache_dir = data_dir / "raw" / "aifdb" / "moral-maze"
            self.corpora = [
                "http://corpora.aifdb.org/zip/britishempire",
                "http://corpora.aifdb.org/zip/Money",
                "http://corpora.aifdb.org/zip/welfare",
                "http://corpora.aifdb.org/zip/problem",
                "http://corpora.aifdb.org/zip/mm2012",
                "http://corpora.aifdb.org/zip/mm2012a",
                "http://corpora.aifdb.org/zip/bankingsystem",
                "http://corpora.aifdb.org/zip/mm2012b",
                "http://corpora.aifdb.org/zip/mmbs2",
                "http://corpora.aifdb.org/zip/mm2012c",
                "http://corpora.aifdb.org/zip/MMSyr",
                "http://corpora.aifdb.org/zip/MoralMazeGreenBelt",
                "http://corpora.aifdb.org/zip/MM2019DDay",
            ]
        elif self.name == "vacc-itc":
            self.cache_dir = data_dir / "raw" / "aifdb" / "vacc-itc"
            self.corpora = [
                "http://corpora.aifdb.org/zip/VaccITC1",
                "http://corpora.aifdb.org/zip/VaccITC2",
                "http://corpora.aifdb.org/zip/VaccITC3",
                "http://corpora.aifdb.org/zip/VaccITC4",
                "http://corpora.aifdb.org/zip/VaccITC5",
                "http://corpora.aifdb.org/zip/VaccITC6",
                "http://corpora.aifdb.org/zip/VaccITC7",
                "http://corpora.aifdb.org/zip/VaccITC8",
            ]
        elif self.name == "us2016":
            self.cache_dir = data_dir / "raw" / "aifdb" / "us2016"
            self.corpora = [
                "http://corpora.aifdb.org/zip/US2016",
            ]


class AIFDBLoader(DatasetLoader):  # pylint: disable=too-few-public-methods
    """loads aifdb raw data"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._aifdb_config = AIFDBConfig(
            **{
                field.name: kwargs[field.name]
                for field in dataclasses.fields(AIFDBConfig)
                if field.name in kwargs
            }
        )
        logging.debug(
            "AIFDBLoader using config: %s", dataclasses.asdict(self._aifdb_config)
        )

    @staticmethod
    def _read_textfile(textfile: Path) -> str:
        """tries to read text file"""
        lines: List[str] = []
        if textfile.exists():
            for enc in ["utf8", "ascii", "windows-1252", "cp500"]:
                if lines:
                    break
                try:
                    lines = textfile.open(encoding=enc).readlines()
                except UnicodeDecodeError as err:
                    logging.debug(
                        "Couldn't decode %s as %s, error: %s", textfile, enc, err
                    )
        text = "No source text."
        if lines:
            text = "".join(lines)
        else:
            logging.warning("Couldn't decode %s, using dummy text.", textfile)

        return text

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
        data: Dict[str, List] = {"nodeset": [], "text": [], "corpus": []}
        for corpus_dir in aifdb_dir.iterdir():
            if corpus_dir.is_dir():
                for nodefile in corpus_dir.iterdir():
                    if nodefile.suffix == ".json":
                        textfile = nodefile.parent / (nodefile.stem + ".txt")
                        data["nodeset"].append(json.load(nodefile.open()))
                        data["text"].append(self._read_textfile(textfile))
                        data["corpus"].append(corpus_dir.name)
        dataset = datasets.Dataset.from_dict(data)

        # create train-validation-test splits
        dataset_split1 = dataset.train_test_split(
            test_size=(1 - splits["train"]), seed=42
        )  # split once
        dataset_split2 = dataset_split1["test"].train_test_split(
            test_size=(splits["test"] / (splits["test"] + splits["validation"])),
            seed=42,
        )  # split test-split again
        dataset_dict = datasets.DatasetDict(
            train=dataset_split1["train"],
            validation=dataset_split2["train"],
            test=dataset_split2["test"],
        )

        return dataset_dict


class _Utils:
    """utilities for preprocessing AIFdb"""

    cleanr = re.compile("<.*?>")

    @staticmethod
    def cleanhtml(example):
        """cleans html in source texts"""
        example["text"] = re.sub(_Utils.cleanr, "", example["text"])
        return example

    @staticmethod
    def split_nodeset_per_inference(  # pylint: disable=too-many-locals, too-many-statements
        examples: Dict[str, List]
    ) -> Dict[str, List]:
        """extracts individual inferences from nodesets, and splits nodesets accordingly"""

        inference_chunks: Dict[str, List] = {
            k: []
            for k in PreprocessedAIFDBExample.__annotations__.keys()  # pylint: disable=no-member
        }
        node_type: Dict = {}
        node_text: Dict = {}
        graph: nx.Graph
        # for each example
        for i, nodeset in enumerate(examples["nodeset"]):
            corpus = examples["corpus"][i]
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
            alternative_text_list = [
                node_text.get(n, "")
                for n in graph.nodes
                if node_type.get(n, None) == "L"
            ]  # L-nodes
            alternative_text = " ".join(alternative_text_list)
            alternative_text = alternative_text.replace("  ", " ")

            # use longer text
            text = examples["text"][i]
            if len(alternative_text) > 2 * (len(text) - text.count("\n")):
                text = alternative_text
                logging.debug(
                    "Using alternative text with length %s", len(alternative_text)
                )

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
                    if node_type.get(n) == "I"
                ]
                # get premises (ids)
                premises = [
                    n
                    for n in graph.predecessors(inference_node)
                    if node_type.get(n) == "I"
                ]

                # get conjectures and reasons (ids)
                def get_l_grandparent(node, corpus) -> List:
                    if node_type[node] != "I":
                        logging.warning(
                            "`get_l_grandparent` called with node of type != `I`"
                        )
                        return []
                    ya_predecessors = [
                        n for n in graph.predecessors(node) if node_type.get(n) == "YA"
                    ]
                    if not ya_predecessors:
                        logging.warning(
                            "node %s in corpus %s has no grandparents", node, corpus
                        )
                        return []
                    l_grandparents = [
                        n
                        for m in ya_predecessors
                        for n in graph.predecessors(m)
                        if node_type.get(n) == "L" and node_text.get(n) != "analyses"
                    ]
                    if not l_grandparents:
                        logging.warning(
                            "node %s in corpus %s has no L-grandparents", node, corpus
                        )
                    return l_grandparents

                conjectures = [get_l_grandparent(n, corpus) for n in conclusions]
                if conjectures:
                    # flatten
                    conjectures = [x for sublist in conjectures for x in sublist]
                    # sort, ids correspond to location in text
                    conjectures = sorted(conjectures)
                reasons = [get_l_grandparent(n, corpus) for n in premises]
                if reasons:
                    # flatten
                    reasons = [x for sublist in reasons for x in sublist]
                    # sort, ids correspond to location in text
                    reasons = sorted(reasons)
                # subst text for ids
                conjectures = [node_text.get(n, "") for n in conjectures]
                reasons = [node_text.get(n, "") for n in reasons]
                conclusions = [node_text.get(n, "") for n in conclusions]
                premises = [node_text.get(n, "") for n in premises]
                # create new record
                inference_chunks["text"].append(text)
                inference_chunks["corpus"].append(corpus)
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

        dataset = dataset.map(_Utils.cleanhtml)

        dataset = dataset.map(
            _Utils.split_nodeset_per_inference,
            batched=True,
            remove_columns=dataset.column_names,
        )

        return dataset

    def __init__(self, **kwargs) -> None:
        """
        A fresh builder instance should contain a blank product object, which is
        used in further assembly.
        """
        super().__init__(**kwargs)
        self._input: PreprocessedAIFDBExample

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
        self._aifdb_config: AIFDBConfig = AIFDBConfig(
            **{
                field.name: kwargs[field.name]
                for field in dataclasses.fields(AIFDBConfig)
                if field.name in kwargs
            }
        )

    @property
    def input(self) -> PreprocessedAIFDBExample:
        return self._input

    def set_input(self, batched_input: Dict[str, List]) -> None:
        self._input = PreprocessedAIFDBExample.from_batch(batched_input)

    def configure_product(self) -> None:
        # create configuration and add empty da2 item to product
        itype = self.input.type
        sp_template = random.choice(
            self._aifdb_config.templates_sp_ra
            if itype == "RA"
            else self._aifdb_config.templates_sp_ca
        )
        metadata = [
            ("corpus", self.input.corpus),
            ("type", itype),
            ("config", {"sp_template": sp_template}),
        ]
        self._product.append(DeepA2Item(metadata=metadata))

    def produce_da2item(self) -> None:
        # we produce a single da2item per input only
        record = self._product[0]
        record.source_text = str(self.input.text)
        if self.input.reasons:
            record.reasons = [
                QuotedStatement(text=r, starts_at=-1, ref_reco=e + 1)
                for e, r in enumerate(self.input.reasons)
            ]
        n_reas = len(record.reasons)
        if self.input.conjectures:
            record.conjectures = [
                QuotedStatement(text=j, starts_at=-1, ref_reco=n_reas + 1)
                for j in self.input.conjectures
            ]
        # source paraphrase
        sp_template = self._env.get_template(
            dict(record.metadata)["config"]["sp_template"]
        )
        record.source_paraphrase = sp_template.render(
            premises=self.input.premises, conclusion=self.input.conclusions
        )

    def postprocess_da2item(self) -> None:
        pass

    def add_metadata_da2item(self) -> None:
        pass
