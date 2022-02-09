"""tests the builder.Director pipeline by means of a dummy builder"""

import dataclasses
import pathlib
from typing import List, Union, Dict

import datasets

from deepa2 import (
    DeepA2Item,
)

from deepa2.builder import (
    DatasetLoader,
    Director,
    PreprocessedExample,
    RawExample,
    Builder,
)


RAW_EXAMPLES = {"text": ["premise conclusion", "another_premise another_conclusion"]}


@dataclasses.dataclass
class DummyRawExample(RawExample):
    """Dummy raw example structure"""

    text: Union[str, List[str]]


@dataclasses.dataclass
class DummyPreprocessedExample(PreprocessedExample):
    """Dummy preprocessed example structure"""

    text: str
    premise: str
    conclusion: str


class DummyDatasetLoader(DatasetLoader):  # pylint: disable=too-few-public-methods
    """Dummy dataset loader"""

    def load_dataset(self) -> datasets.DatasetDict:
        dataset = datasets.Dataset.from_dict(RAW_EXAMPLES)
        splits = {"train": dataset, "validation": dataset, "test": dataset}
        return datasets.DatasetDict(splits)


class DummyBuilder(Builder):
    """Dummy builder for tests"""

    @staticmethod
    def preprocess(dataset: datasets.Dataset) -> datasets.Dataset:
        def function(example):
            preprocessed_example = DummyPreprocessedExample(
                text=example["text"],
                premise=example["text"].split()[0],
                conclusion=example["text"].split()[-1],
            )
            preprocessed_record = dataclasses.asdict(preprocessed_example)
            return preprocessed_record

        dataset = dataset.map(function)
        return dataset

    def __init__(self) -> None:
        super().__init__()
        self._input: DummyPreprocessedExample

    @property
    def input(self) -> DummyPreprocessedExample:
        return self._input

    def set_input(self, batched_input: Dict[str, List]) -> None:
        self._input = DummyPreprocessedExample.from_batch(batched_input)

    def configure_product(self) -> None:
        metadata = {
            "configured": True,
        }
        self._product.append(DeepA2Item(metadata=metadata))

    def produce_da2item(self) -> None:
        record = self._product[0]  # we produce a single da2item per input only
        record.argument_source = str(self.input.text)
        record.argdown_reconstruction = (
            f"{self.input.premise}\n----\n{self.input.conclusion}"
        )

    def postprocess_da2item(self) -> None:
        record = self._product[0]  # we produce a single da2item per input only
        record.metadata["postprocessed"] = True

    def add_metadata_da2item(self) -> None:
        record = self._product[0]  # we produce a single da2item per input only
        record.metadata["metadata_added"] = True


def test_pipeline(tmp_path):
    """tests the pipeline"""
    director = Director()
    builder = DummyBuilder()
    dataset_loader = DummyDatasetLoader()
    director.builder = builder
    director.dataset_loader = dataset_loader
    director.raw_example_class = DummyRawExample
    director.preprocessed_example_class = DummyPreprocessedExample

    director.transform(export_path=tmp_path, debug_size=10, name="dummy")

    da2_train_split = datasets.Dataset.from_parquet(
        str(pathlib.Path(tmp_path, "dummy", "train", "train.parquet"))
    )

    da2_train_split = da2_train_split.to_dict()

    text_check = da2_train_split["argument_source"] == RAW_EXAMPLES["text"]
    print(da2_train_split["argument_source"])

    argdown_check = da2_train_split["argdown_reconstruction"] == [
        "premise\n----\nconclusion",
        "another_premise\n----\nanother_conclusion",
    ]
    print(da2_train_split["argdown_reconstruction"])

    metadata_check = da2_train_split["metadata"][0] == {
        "configured": True,
        "postprocessed": True,
        "metadata_added": True,
    }
    print(da2_train_split["metadata"])

    assert text_check and argdown_check and metadata_check
