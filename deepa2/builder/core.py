"""Defines core abstract classes and data structures for DeepA2 datasets.

This module defines core abstract classes and data structures for building
DeepA2 datasets. The core classess follow the director-builder design pattern.
The `Director` sets up a universal pipeline for transforming a raw dataset.
Scripts that build concrete datasets only have to implement the abstract `Builder`
interface, specify `RawExample` and `PreprocessedExample`, and possibly adapt
the `DatasetLoader`.

In addition, the module defines the structure of DeepA2 datasets by
means of the dataclass `DeepA2Item()`.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
import dataclasses
import logging
from pathlib import Path
from typing import Optional, Any, List, Dict, Type
import uuid

import datasets

from deepa2 import (
    BaseExample,
    DeepA2Item,
    QuotedStatement,
    ArgdownStatement,
    Formalization,
)
import deepa2


@dataclasses.dataclass
class RawExample(BaseExample):
    """Raw Example dataclass"""


@dataclasses.dataclass
class PreprocessedExample(BaseExample):
    """Preprocessed Example dataclass"""


class DatasetLoader:  # pylint: disable=too-few-public-methods
    """
    Provides a method for loading the raw dataset.
    """

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def load_dataset(self) -> datasets.DatasetDict:
        """
        Default DatasetLoader uses HF `Dataset.load_dataset`.
        """
        dataset = datasets.load_dataset(*self._args, **self._kwargs)
        if not isinstance(dataset, datasets.DatasetDict):
            raise ValueError("HF Dataset is not of type DatasetDict.")

        return datasets.DatasetDict(dataset)


class Builder(ABC):
    """Defines interface for source-data specific builders.

    The Builder interface specifies a static preprocessing method as well as
    methods for configuring and building DeepA2Items.
    """

    @staticmethod
    @abstractmethod
    def preprocess(dataset: datasets.Dataset) -> datasets.Dataset:
        """
        Preprocesses the dataset.
        """

    def __init__(self, **kwargs):
        logging.debug("Initializing Builder with config: %s", kwargs)
        self._product: List[DeepA2Item] = []
        self._input: PreprocessedExample

    @property
    def product(self) -> List[Dict]:
        """
        The product of any builder is a list of DeepA2Items,
        rendered as dicts
        """
        product = [dataclasses.asdict(deepa2item) for deepa2item in self._product]
        # serialize metadata values
        for record in product:
            record["metadata"] = [(k, str(v)) for k, v in record["metadata"]]
        self._reset()
        return product

    def _reset(self) -> None:
        self._product = []

    @property
    def input(self) -> PreprocessedExample:
        """
        The input of any builder is a preprocessed example.
        """
        return self._input

    @abstractmethod
    def set_input(self, batched_input: Dict[str, List]) -> None:
        """Sets input for building next product.

        As we're processing one pre-processed item per build-step,
        we internally represent the batched input as a single, unbatched
        `PreprocessedExample`
        """
        self._input = PreprocessedExample.from_batch(batched_input)

    @abstractmethod
    def configure_product(self) -> None:
        """Configures product"""

    @abstractmethod
    def produce_da2item(self) -> None:
        """Produces product"""

    @abstractmethod
    def postprocess_da2item(self) -> None:
        """Postprocesses product"""

    @abstractmethod
    def add_metadata_da2item(self) -> None:
        """Adds metadata to product"""


class Director:  # pylint: disable=too-many-instance-attributes
    """Implements a universal pipeline for building DeepA2 datasets.

    Typical usage example:
        .. code-block:: python

            from deepa2 import core

            class MyBuilder(core.Builder):
                ...

            class MyRawExample(core.RawExample):
                ...

            class MyPreprocessedExample(core.PreprocessedExample):
                ...

            director = core.Director()
            builder = MyBuilder()
            dataset_loader = core.DatasetLoader("some-dataset-at-hf-hub")
            director.builder = builder
            director.dataset_loader = dataset_loader
            director.raw_example_class = MyRawExample
            director.preprocessed_example_class = MyPreprocessedExample

            director.transform(export_path="some-path")
    """

    def __init__(self) -> None:
        self._builder: Builder
        self._dataset_loader: DatasetLoader
        self._raw_example_class: Type[RawExample]
        self._preprocessed_example_class: Type[PreprocessedExample]

    @property
    def builder(self) -> Builder:
        """builder property"""
        return self._builder

    @builder.setter
    def builder(self, builder: Builder) -> None:
        """
        Builder instace to use for preprocessing and constructing DeepA2Items.
        """
        self._builder = builder

    @property
    def dataset_loader(self) -> DatasetLoader:
        """dataset_loader property"""
        return self._dataset_loader

    @dataset_loader.setter
    def dataset_loader(self, dataset_loader: DatasetLoader) -> None:
        """
        DatasetLoader for loading the raw dataset to-be processed.
        """
        self._dataset_loader = dataset_loader

    @property
    def raw_example_class(self) -> Type[RawExample]:
        """raw_example_class property"""
        return self._raw_example_class

    @raw_example_class.setter
    def raw_example_class(self, raw_example_class: Type[RawExample]) -> None:
        """
        Class of raw examples, used for sanity checks during execution of pipeline.
        """
        self._raw_example_class = raw_example_class

    @property
    def preprocessed_example_class(self) -> Type[PreprocessedExample]:
        """preprocessed_example_class"""
        return self._preprocessed_example_class

    @preprocessed_example_class.setter
    def preprocessed_example_class(
        self, preprocessed_example_class: Type[PreprocessedExample]
    ) -> None:
        """
        Class of preprocessed examples, used for sanity checks during execution of pipeline.
        """
        self._preprocessed_example_class = preprocessed_example_class

    def process(self, batched_input: Dict[str, List]) -> Dict[str, List]:
        """
        The Director provides a function that can be mapped over a dataset
        (requiring batches of size 1).
        """
        if any(len(v) != 1 for v in batched_input.values()):
            raise ValueError("Director.transform() can only handle batches of size 1.")
        if len(set(len(v) for v in batched_input.values())) > 1:
            raise ValueError(
                "Director.transform(): batched_input is not of uniform length."
            )
        self.builder.set_input(batched_input)
        self.builder.configure_product()
        self.builder.produce_da2item()
        self.builder.postprocess_da2item()
        self.builder.add_metadata_da2item()
        da2items = self.builder.product  # product is a list of dicts

        # sanity checks
        for da2item in da2items:
            if list(da2item.keys()) != [
                field.name for field in dataclasses.fields(DeepA2Item)
            ]:
                logging.warning(
                    "Builder product contains item that is not a DeepA2 item: %s",
                    da2item,
                )
                raise ValueError(
                    "Builder product contains item that is not a DeepA2 item."
                )
            try:
                isinstance(DeepA2Item(**da2item), DeepA2Item)
            except Exception:
                logging.error(
                    "Builder product contains item that cannot be cast as DeepA2 item: %s",
                    da2item,
                )
                raise

        # transpose to dict of lists
        batched_result = {}
        for k in da2items[0].keys():
            batched_result[k] = [record[k] for record in da2items]
        return batched_result

    @staticmethod
    def postprocess(
        deepa2_record: Dict[str, Any], active_features: List[str]
    ) -> Dict[str, Any]:
        """postprocesses DeepA2 items"""
        postprocessed = deepa2_record
        for key, value in postprocessed.items():
            if key not in active_features:
                postprocessed[key] = None
            elif value in [
                [QuotedStatement()],
                [Formalization()],
                [ArgdownStatement()],
            ]:
                postprocessed[key] = []

        return postprocessed

    def transform(
        self,
        export_path: Optional[str] = None,
        active_features: Optional[List[str]] = None,
        debug_size: Optional[int] = None,
        name: str = "default_name",
        **kwargs,
    ) -> None:
        """
        Implements the universal pipeline for transforming datasets.
        """

        logging.info(
            "#################################################################"
        )
        logging.info("Starting new %s transformation: {datetime.datetime.now()}", name)
        logging.debug("Configuration: %s", kwargs)

        # 1. Load dataset
        dataset = self.dataset_loader.load_dataset()
        # check splits
        if list(dataset.keys()) != ["train", "validation", "test"]:
            logging.warning(
                "Expected split ['train','validation','test'] but dataset has splits: %s",
                list(dataset.keys()),
            )
        # check features
        self.raw_example_class.check_features(dataset)
        logging.info("Loaded dataset: %s", dataset)

        # 2. Work on small subset for debugging
        if debug_size:
            for key, split in dataset.items():
                dataset[key] = split.filter(
                    lambda ex, idx: 1 if (idx < debug_size) else 0, with_indices=True
                )
            logging.info("Debug mode, working with filtered raw dataset: %s", dataset)

        # 3. Preprocess each split
        for key, split in dataset.items():
            logging.info("Preprocessing split %s ...", split)
            dataset[key] = self.builder.preprocess(split)
        # check features
        self.preprocessed_example_class.check_features(dataset)
        logging.info("Preprocessed dataset: %s", dataset)

        # 4. Transform
        pp_example_fields = [
            field.name for field in dataclasses.fields(self.preprocessed_example_class)
        ]
        # Infer features from empty DeepA2Item
        # da2_features: datasets.Features = datasets.Dataset.from_dict(
        #    {
        #        k: [v] for k, v in
        #        dataclasses.asdict(DeepA2Item()).items()
        #    }
        # ).features
        # Map
        dataset = dataset.map(
            self.process,
            batched=True,
            batch_size=1,
            remove_columns=pp_example_fields,
            features=deepa2.DA2_FEATURES,
        )
        logging.info("Created new %s deepa2 dataset: %s", name, dataset)
        logging.debug("Features: %s", dataset["train"].info.features)

        # 6. Postprocess the dataset
        if not debug_size:
            if all("metadata" in split.column_names for _, split in dataset.items()):
                for key, split in dataset.items():
                    dataset[key] = split.remove_columns("metadata")
                logging.info("Removed metadata from deepa2 dataset")

            def add_uuid(example):
                uid = name + "_" + str(uuid.uuid4())
                example["metadata"] = [("id", uid)]
                return example

            dataset = dataset.map(add_uuid)
            logging.info("Added uuids deepa2 dataset")
        if not active_features:
            active_features = list(dataset.column_names.values())[0]

        def function(example):
            return self.postprocess(example, active_features=active_features)

        dataset = dataset.map(function)

        # 6. Save to disk
        if export_path:
            path = Path(export_path, name)
            for key, split in dataset.items():
                logging.info("Saving %s split %s ...", name, key)
                file_name = f"{key}.parquet"
                (path / key).mkdir(
                    parents=True, exist_ok=True
                )  # create dirs if necessary
                split.to_parquet(path / key / file_name)
            logging.info("Saved %s deepa2 dataset to %s.", name, path)
