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
from typing import Optional,Any,List,Dict,TypedDict,Type

import logging
from pathlib import Path
import datetime

import dataclasses
from dataclasses import dataclass

import datasets



class RawExample(TypedDict):
    pass

class PreprocessedExample(TypedDict):
    pass


@dataclass
class QuotedStatement():
    text:str
    starts_at:int
    ref_reco:int

@dataclass
class ArgdownStatement():
    text:str
    explicit:Any
    ref_reco:int

@dataclass
class Formalization():
    form:str
    ref_reco:int


@dataclass
class DeepA2Item():
    """
    Dataclass defining the structure of a DeepA2 example.
    """

    argument_source:str = None

    title:str = None # telling title of the reconstructed argument
    gist:str = None # very succinct summary of the argument, main point of the argument
    source_paraphrase:str = None # a maximally clear, though conservative summary of the argument 
        # (i.e., leave out distractors, focus on one argument, leave out redundant parts, syntactic 
        # streamlining, add inference indicators, but generally NO explication of implicit premises)
    context:str = None # provides informal or semi-formal description of the argument's context, ideally
        # an argdown snippet (without detailed reconstruction) sketching the dialectic neighbourhood 

    argdown_reconstruction:str = None
    erroneous_argdown:str = None

    reason_statements:List[QuotedStatement] = None
    conclusion_statements:List[QuotedStatement] = None
    
    premises:List[ArgdownStatement] = None
    intermediary_conclusion:List[ArgdownStatement] = None
    conclusion:List[ArgdownStatement] = None
    
    premises_formalized:List[Formalization] = None
    intermediary_conclusion_formalized:List[Formalization] = None
    conclusion_formalized:List[Formalization] = None
    predicate_placeholders:List[str] = None
    entity_placeholders:List[str] = None
    misc_placeholders:List[str] = None

    distractors:List[str] = None
    metadata:Dict = None

    

class DatasetLoader():
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
        return datasets.load_dataset(*self._args, **self._kwargs)


class Builder(ABC):
    """Defines interface for source-data specific builders.

    The Builder interface specifies a static preprocessing method as well as 
    methods for configuring and building DeepA2Items.
    """

    @staticmethod
    @abstractmethod
    def preprocess(dataset:datasets.Dataset) -> datasets.Dataset:
        """
        Preprocesses the dataset.
        """
        pass

    @property
    def product(self) -> List[Dict]:
        """
        The product of any builder is a list of DeepA2Items, 
        rendered as dicts
        """
        product = self._product
        product = [dataclasses.asdict(deepa2item) for deepa2item in product]
        self.reset()
        return product

    def reset(self) -> None:
        self._product:List[DeepA2Item] = []


    @property
    @abstractmethod
    def input(self) -> PreprocessedExample:
        """
        The input of any builder is a proprocessed example.
        """
        pass

    @input.setter
    @abstractmethod
    def input(self, preprocessed_example: PreprocessedExample) -> None:
        """
        Sets input for building next product.
        """
        self._input = preprocessed_example

    @abstractmethod
    def configure_product(self) -> None:
        pass

    @abstractmethod
    def produce_da2item(self) -> None:
        pass

    @abstractmethod
    def postprocess_da2item(self) -> None:
        pass

    @abstractmethod
    def add_metadata_da2item(self) -> None:
        pass




class Director:
    """Implements a universal pipeline for building DeepA2 datasets.

        Typical usage example:
            .. code-block:: python

                from deepa2datasets import core

                class MyBuilder(core.Builder):
                    pass

                class MyRawExample(core.RawExample):
                    pass

                class MyPreprocessedExample(core.PreprocessedExample):
                    pass

                director = core.Director()
                builder = MyBuilder()
                dataset_loader = core.DatasetLoader("some-dataset-at-hf-hub")
                director.builder = builder
                director.dataset_loader = dataset_loader
                director.raw_example_type = MyRawExample
                director.preprocessed_example_type = MyPreprocessedExample  

                director.transform(export_path="some-path")
    """

    def __init__(self) -> None:
        self._builder = None
        self._dataset_loader = None
        self._raw_example_type = None
        self._builder = None

    @property
    def builder(self) -> Builder:
        return self._builder

    @builder.setter
    def builder(self, builder: Builder) -> None:
        """
        Builder instace to use for preprocessing and constructing DeepA2Items.
        """
        self._builder = builder

    @property
    def dataset_loader(self) -> DatasetLoader:
        return self._dataset_loader

    @dataset_loader.setter
    def dataset_loader(self, dataset_loader: DatasetLoader) -> None:
        """
        DatasetLoader for loading the raw dataset to-be processed.
        """
        self._dataset_loader = dataset_loader


    @property
    def raw_example_type(self) -> Type[RawExample]:
        return self._raw_example_type

    @raw_example_type.setter
    def raw_example_type(self, raw_example_type: Type[RawExample]) -> None:
        """
        Class of raw examples, used for sanity checks during execution of pipeline.
        """
        self._raw_example_type = raw_example_type


    @property
    def preprocessed_example_type(self) -> Type[RawExample]:
        return self._preprocessed_example_type

    @preprocessed_example_type.setter
    def preprocessed_example_type(self, preprocessed_example_type: Type[RawExample]) -> None:
        """
        Class of preprocessed examples, used for sanity checks during execution of pipeline.
        """
        self._preprocessed_example_type = preprocessed_example_type


    def process(self,batched_input:Dict[List]) -> Dict[List]:
        """
        The Director provides a function that can me mapped over a dataset (requiring batches of size 1).
        """
        if any(len(v)!=1 for v in batched_input.values()):
            raise ValueError("Director.transform() can only handle batches of size 1.")
        if len(set(len(v) for v in batched_input.values()))>1:
            raise ValueError("Director.transform(): batched_input is not of uniform length.")
        self.builder.input = batched_input
        self.builder.configure_product()
        self.builder.produce_da2item()
        self.builder.postprocess_da2item()
        self.builder.add_metadata_da2item()
        da2items = self.builder.product # product is a list of dicts
        # sanity checks
        for da2item in da2items:
            if list(da2item.keys()) != [field.name for field in dataclasses.fields(DeepA2Item)]:
                logging.warning(f"Builder product contains item that is not a DeepA2 item: {da2item}")
                raise ValueError("Builder product contains item that is not a DeepA2 item.")
        # transpose to dict of lists
        batched_result = {}
        for k in da2items[0].keys():
            batched_result[k] = [record[k] for record in da2items]
        return batched_result


    def transform(self, export_path: Optional[str] = None, debug_size: Optional[int] = None, name: str = "default_name") -> None:
        """
        Implements the universal pipeline for transforming datasets.
        """

        logging.info(f"#################################################################")
        logging.info(f"Starting new {name} transformation: {datetime.datetime.now()}")

        # 1. Load dataset
        dataset = self.dataset_loader.load_dataset()
        # check splits
        if not list(dataset.keys()) == ["train","validation","test"]:
            logging.warning(f"Expected split ['train','validation','test'] but dataset has splits: {list(dataset.keys())}")
        ## check features
        for split in dataset.keys():
            if not (dataset[split].column_names == list(self.raw_example_type.__annotations__.keys())):
                logging.error(f"Features of dataset with raw examples ({dataset.column_names}) don't match raw_example_type ({list(self.raw_example_type.__annotations__.keys())}).")
                raise ValueError("Features of dataset with raw examples don't match raw_example_type.")
        logging.info(f"Loaded dataset: {dataset}")

        # 2. Work on small subset for debugging
        if debug_size:
            for split in dataset.keys():
                dataset[split] = dataset[split].filter(lambda ex,idx: 1 if (idx<debug_size) else 0, with_indices=True)
            logging.info(f"Debug mode, working with filtered raw dataset: {dataset}")

        # 3. Preprocess each split
        for split in dataset.keys():
            logging.info(f"Preprocessing split {split} ...")
            dataset[split] = self.builder.preprocess(dataset[split])
        ## check features
        for split in dataset.keys():
            if not (dataset[split].column_names == list(self.preprocessed_example_type.__annotations__.keys())):
                logging.error(f"Features of dataset with preprocessed examples ({dataset.column_names}) don't match raw_example_type ({list(self.preprocessed_example_type.__annotations__.keys())}).")
                raise ValueError("Features of dataset with preprocessed examples don't match preprocessed_example_type.")
        logging.info(f"Preprocessed dataset: {dataset}")

        # 4. Transform
        dataset = dataset.map(self.process, batched=True, batch_size=1, remove_columns=list(self.preprocessed_example_type.__annotations__.keys()))
        logging.info(f"Created new {name} deepa2 dataset: {dataset}")

        # 5. Remove metadata
        if (not debug_size) and all("metadata" in dataset[split].column_names for split in dataset.keys()):
            dataset = dataset.remove_columns("metadata")
            logging.info(f"Removed metadata from deepa2 dataset")

        # 6. Save to disk
        if export_path:
            path = Path(export_path,name)
            for split in dataset.keys():
                logging.info(f"Saving {name} split {split} ...")
                file_name = f"{split}.parquet"
                (path / split).mkdir(parents=True, exist_ok=True) # create dirs if necessary
                dataset[split].to_parquet(path / split / file_name)
            logging.info(f"Saved {name} deepa2 dataset to {path}.")

