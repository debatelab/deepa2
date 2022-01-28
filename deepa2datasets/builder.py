from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any,List,Dict,TypedDict

from dataclasses import dataclass, asdict

from datasets import Dataset


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
    It makes sense to use the Builder pattern only when your products are quite
    complex and require extensive configuration.

    Unlike in other creational patterns, different concrete builders can produce
    unrelated products. In other words, results of various builders may not
    always follow the same interface.
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

    


class Builder(ABC):
    """
    The Builder interface specifies methods for creating the different parts of
    the Product objects.
    """

    @staticmethod
    @abstractmethod
    def preprocess(dataset:Dataset) -> Dataset:
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
        product = [asdict(deepa2item) for deepa2item in product]
        self.reset()
        return product

    def reset(self) -> None:
        self._product:List[DeepA2Item] = []


    @property
    @abstractmethod
    def input(self) -> PreprocessedExample:
        """
        The input of any builder is a proprocessed example
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
    """
    The Director is only responsible for executing the building steps in a
    particular sequence. It is helpful when producing products according to a
    specific order or configuration. Strictly speaking, the Director class is
    optional, since the client can control builders directly.
    """

    def __init__(self) -> None:
        self._builder = None

    @property
    def builder(self) -> Builder:
        return self._builder

    @builder.setter
    def builder(self, builder: Builder) -> None:
        """
        The Director works with any builder instance that the client code passes
        to it. This way, the client code may alter the final type of the newly
        assembled product.
        """
        self._builder = builder


    def transform(self,batched_input:Dict[List]) -> Dict[List]:
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
        # transpose to dict of lists
        batched_result = {}
        for k in da2items[0].keys():
            batched_result[k] = [record[k] for record in da2items]
        return batched_result

