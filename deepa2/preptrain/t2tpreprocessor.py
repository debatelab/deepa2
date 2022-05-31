"""Provides classes for exporting DA2 datasets as Text2Text datasets"""

import dataclasses
import logging
from pathlib import Path
import random
from typing import Dict, Any, List, Tuple

import datasets

from deepa2 import DeepA2Item, DeepA2Layouter, GenerativeMode
import deepa2


class T2TPreprocessor:  # pylint: disable=too-many-instance-attributes
    """merges and exports DA2 datasets as Text2Text dataset"""

    # default T5 mask tokens:
    _MASK_TOKEN = "<extra_id_1>"
    _END_MASK_TOKEN = "<extra_id_2>"
    _MAX_MASK_LENGTH = 15

    def __init__(self, **config) -> None:
        self._sources: Dict = config["sources"]
        self._export_path: str = config["export_path"]
        self._export_format: str = config["export_format"]
        self._mask_probability: float = config["mask_probability"]
        self._generative_modes: List[GenerativeMode] = []
        for mode_data in config["generative_modes"]:
            if "input" not in mode_data or "target" not in mode_data:
                mode = GenerativeMode.from_keys(mode_data.get("name", "invalid_mode"))
            else:
                mode = GenerativeMode(**mode_data)
            if mode is None:
                raise ValueError(f"Invalid mode: {mode_data}")
            self._generative_modes.append(mode)
        self._deactive_modes_with_weights = any(
            mode.weight < 1.0 for mode in self._generative_modes
        )

        self._input_column_name = config["input_column_name"]
        self._target_column_name = config["target_column_name"]

        self._layouter = DeepA2Layouter()

        self._random = random.Random()  # uses its own generator

        # check generative modes
        self.check_modes()

    def check_modes(self) -> None:
        """Checks whether generative modes are well-formed and defined"""
        da2_angles = [field.name for field in dataclasses.fields(DeepA2Item)]
        for mode in self._generative_modes:
            if any(input not in da2_angles for input in mode.input):
                logging.error("Input of mode not a DeepA2 field: %s", mode)
                raise ValueError(f"Input of mode not a DeepA2 field: {mode}.")
            if mode.target not in da2_angles:
                logging.error("Target of mode not a DeepA2 field: %s", mode)
                raise ValueError(f"Target of mode not a DeepA2 field: {mode}.")

    def mask_input(self, input_raw: str) -> Tuple[str, str]:
        """masks a single input string, return masked_input and substitution"""
        input_words = input_raw.split()

        if len(input_words) <= self._MAX_MASK_LENGTH:
            masked_input = self._MASK_TOKEN
            substitution = (
                self._MASK_TOKEN + " " + input_raw + " " + self._END_MASK_TOKEN
            )
        else:
            # choose a random position in input
            pos = self._random.randint(0, len(input_words) - self._MAX_MASK_LENGTH)
            masked_input_w = (
                input_words[0:pos]
                + [self._MASK_TOKEN]
                + input_words[pos + self._MAX_MASK_LENGTH :]
            )
            masked_input = " ".join(masked_input_w)
            substitution_w = (
                [self._MASK_TOKEN]
                + input_words[pos : pos + self._MAX_MASK_LENGTH]
                + [self._END_MASK_TOKEN]
            )
            substitution = " ".join(substitution_w)

        return masked_input, substitution

    def map_to_t2t(self, da2_dict: Dict[str, Any]) -> Dict[str, List[str]]:
        """create multiple t2t items from a single Deep A2 item"""
        t2t_item: Dict[str, List[str]] = {}
        da2_item = DeepA2Item.from_batch(da2_dict)
        da2_item = self._layouter.format(da2_item)

        texts = []
        targets = []

        # selectively deactivate modes acc to weights
        if self._deactive_modes_with_weights:
            active_modes = [
                mode
                for mode in self._generative_modes
                if self._random.random() < mode.weight
            ]
        else:
            active_modes = self._generative_modes

        for mode in active_modes:
            # check whether target or any input is None?
            if not any(da2_item[k] is None for k in mode.input + [mode.target]):
                # determine input key to-be masked
                mask_key = None
                if (
                    len(mode.input) > 1
                    and self._random.random() < self._mask_probability
                ):
                    mask_key = self._random.choice(
                        [
                            k
                            for k in mode.input
                            if k != "source_text"  # don't mask source_text
                        ]
                    )

                inputs = []
                inputs.append(mode.target + ":")  # prefix
                for key in mode.input:  # inputs of mode
                    if key == mask_key:
                        input_raw, mask_substitution = self.mask_input(da2_item[key])
                    else:
                        input_raw = da2_item[key]
                        mask_substitution = None
                    inputs.append(f"{key}: {input_raw}")

                texts.append(" ".join(inputs))

                # determine target
                target = da2_item[mode.target]
                if mask_substitution is not None:
                    target += " " + mask_substitution
                targets.append(target)

        t2t_item[self._input_column_name] = texts
        t2t_item[self._target_column_name] = targets

        return t2t_item

    def _save_dataset(self, dataset: datasets.Dataset, path: Path) -> None:
        """saves the dataset in format given by `self._export_format`"""
        if self._export_format == "parquet":
            dataset.to_parquet(path)
        elif self._export_format == "csv":
            dataset.to_csv(path)
        elif self._export_format == "jsonl":
            dataset.to_json(path, orient="records", lines=True)
        else:
            logging.warning("Unknown format: {self._export_format}, dataset not saved.")

    def transform(self) -> None:
        """transforms sources"""

        logging.info(
            "#################################################################"
        )
        logging.info("Starting new preptrain transformation: {datetime.datetime.now()}")

        t2t_datasets: Dict[str, List] = {}

        for source in self._sources:
            logging.info("Loading dataset dict from source: %s", source)
            kwargs = source
            kwargs["features"] = deepa2.DA2_FEATURES
            try:
                da2_dataset = datasets.load_dataset(**kwargs)
            except KeyError:
                kwargs["features"].pop("metadata", None)
                da2_dataset = datasets.load_dataset(**kwargs)
            if isinstance(da2_dataset, datasets.DatasetDict):
                logging.info("Processing dataset dict %s", da2_dataset)
                for key, split in da2_dataset.items():
                    if key not in t2t_datasets:
                        t2t_datasets[key] = []
                    t2t_dataset = split.map(
                        self.map_to_t2t,
                        batched=True,
                        batch_size=1,
                        remove_columns=split.column_names,
                    )
                    t2t_datasets[key].append(t2t_dataset)
            else:
                logging.warning("Not a dataset_dict: %s; skipping.", source)

        # Save to disk
        dataset = datasets.DatasetDict(
            {
                k: datasets.concatenate_datasets(v).shuffle()
                for k, v in t2t_datasets.items()
            }
        )

        if self._export_path:
            path = Path(
                self._export_path,
            )
            (path).mkdir(parents=True, exist_ok=True)  # create dirs if necessary
            if self._export_format == "arrow":
                # save entire datasetdict
                dataset.save_to_disk(str(path))
            else:
                # save splits individually
                for key, split in dataset.items():
                    logging.info("Saving processed t2t split %s ...", key)
                    file_name = f"{key}.{self._export_format}"
                    self._save_dataset(split, path / file_name)
            logging.info("Saved t2t dataset to %s.", path)
        else:
            logging.warning("No export path, t2t dataset is not saved.")
