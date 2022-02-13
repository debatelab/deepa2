"""Pipelines and PipedBuilder"""

from abc import ABC, abstractmethod
from typing import List, Dict

import jinja2

from deepa2.builder import (
    Builder,
    DeepA2Item,
    PreprocessedExample,
)
from deepa2.config import template_dir


class Transformer(ABC):  # pylint: disable=too-few-public-methods
    """Transformer interface"""

    _TEMPLATE_STRINGS: Dict[str, str] = {}

    def __init__(self, builder: Builder) -> None:
        """init transformer"""
        self._calling_builder = builder
        self._compile_templates()

    def _compile_templates(self) -> None:
        """compiles templates"""
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(),
        )
        self._templates = {
            k: env.from_string(v) for k, v in self._TEMPLATE_STRINGS.items()
        }

    @property
    def templates(self) -> Dict[str, jinja2.Template]:
        """
        The pipeline to execute for building a single DeepA2Item
        """
        return self._templates

    @abstractmethod
    def transform(
        self, da2_item: DeepA2Item, prep_example: PreprocessedExample
    ) -> DeepA2Item:
        """transforms tuple of DeepA2Item and PreprocessedExample"""


class Pipeline:  # pylint: disable=too-few-public-methods
    """pipeline for executing transformer-chain"""

    def __init__(self, transformer_chain: List[Transformer]) -> None:
        self._transformer_chain = transformer_chain

    def transform(
        self, da2_item: DeepA2Item, prep_example: PreprocessedExample
    ) -> DeepA2Item:
        """passes input through transformers in pipeline"""
        output = da2_item
        for transformer in self._transformer_chain:
            output = transformer.transform(da2_item=output, prep_example=prep_example)

        return output


class PipedBuilder(Builder):
    """Builder interface that uses pipeline to create DeepA2Items"""

    def __init__(self, **kwargs):
        super().__init__()
        self._pipeline = self._construct_pipeline(**kwargs)

    @property
    def pipeline(self) -> Pipeline:
        """
        The pipeline to execute for building a single DeepA2Item
        """
        return self._pipeline

    @abstractmethod
    def _construct_pipeline(self, **kwargs) -> Pipeline:
        """constructs pipeline"""
        pipeline = Pipeline([])
        return pipeline

    def configure_product(self) -> None:
        # populate product with configs
        self._product.append(DeepA2Item())

    def produce_da2item(self) -> None:
        """Produces product"""
        for record in self._product:
            record = self.pipeline.transform(record, self.input)

    def postprocess_da2item(self) -> None:
        pass

    def add_metadata_da2item(self) -> None:
        pass
