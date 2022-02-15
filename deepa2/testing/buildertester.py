"""Base Builder Test used in pytest"""

import dataclasses
from typing import List

import datasets
import pytest

from deepa2.builder import (
    Builder,
    RawExample,
    PreprocessedExample,
    DeepA2Item,
)


class BaseBuilderTest:
    """Base class for testimg builders"""

    RAW_EXAMPLES: List[RawExample] = []
    PREPROCESSED_EXAMPLES: List[PreprocessedExample] = []
    DEEP_A2_ITEMS: List[DeepA2Item] = []

    builder: Builder

    skip_features: List[str] = ["metadata"]

    def test_preprocessor(self):
        """tests builder's preprocessor"""

        if self.RAW_EXAMPLES:
            prep_example_class = self.PREPROCESSED_EXAMPLES[
                0
            ].__class__
            raw_data = {}
            for field in dataclasses.fields(self.RAW_EXAMPLES[0]):
                key = field.name
                raw_data[key] = [getattr(example, key) for example in self.RAW_EXAMPLES]
            dataset = datasets.Dataset.from_dict(raw_data)
            dataset = self.builder.preprocess(dataset)
            preprocessed_data = dataset.to_dict(
                batch_size=1, batched=True
            )  # return iterator
            preprocessed_examples = []
            for batch in preprocessed_data:
                # unbatch
                preprocessed_example = prep_example_class(
                    **{k: v[0] for k, v in batch.items()}
                )
                preprocessed_examples.append(preprocessed_example)
            assert preprocessed_examples == self.PREPROCESSED_EXAMPLES

    @pytest.fixture(name="processed_examples")
    def fixture_processed_examples(self):
        """processes examples"""
        da2items = []
        for preprocessed_example in self.PREPROCESSED_EXAMPLES:
            batched_input = {
                k: [v] for k, v in dataclasses.asdict(preprocessed_example).items()
            }
            builder = self.builder
            builder.set_input(batched_input)
            builder.configure_product()
            builder.produce_da2item()
            builder.postprocess_da2item()
            builder.add_metadata_da2item()
            da2items.extend(builder.product)  # product is a list of dicts
        return da2items

    def test_processor(self, processed_examples):
        """compares given and constructed da2items feature-wise"""

        for i, processed_example in enumerate(processed_examples):
            da2item_gen = DeepA2Item.from_batch(
                {k: [v] for k, v in processed_example.items()}
            )
            for key, _ in processed_example.items():
                if key not in self.skip_features:
                    value_given = getattr(self.DEEP_A2_ITEMS[i], key)
                    value_gen = getattr(da2item_gen, key)
                    print(f"{key}: {value_gen} <<<>>> {value_given}")
                    assert value_gen == value_given
