from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any,List,Dict

from dataclasses import dataclass, asdict


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
    id:str = "123-456"
    metadata:Dict = None

    


class Builder(ABC):
    """
    The Builder interface specifies methods for creating the different parts of
    the Product objects.
    """

    @property
    @abstractmethod
    def product(self) -> None:
        pass

    @abstractmethod
    def fetch_input(self,input_dataset) -> Any:
        """
        Fetches items to be processed for building next product and returns 
        dataset with remaining input items which will be processed later.
        """
        pass

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

    """
    The Director can construct several product variations using the same
    building steps.
    """

    def build_minimal_viable_product(self) -> None:
        self.builder.produce_part_a()

    def build_full_featured_product(self) -> None:
        self.builder.produce_part_a()
        self.builder.produce_part_b()
        self.builder.produce_part_c()
