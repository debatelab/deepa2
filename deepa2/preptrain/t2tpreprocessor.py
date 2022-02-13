"""Provides classes for exporting DA2 datasets as Text2Text datasets"""

import dataclasses
import logging
import random
from typing import Dict, Any, List

import datasets

from deepa2 import DeepA2Item, DeepA2Layouter, GenerativeMode


class T2TPreprocessor:  # pylint: disable=too-many-instance-attributes
    """merges and exports DA2 datasets as Text2Text dataset"""

    def __init__(self, **config) -> None:
        self._sources: Dict = config["sources"]
        self._export_path: str = config["export_path"]
        self._generative_modes: List[GenerativeMode] = []
        for mode_data in config["generative_modes"]:
            if "input" not in mode_data or "target" not in mode_data:
                mode = GenerativeMode.from_keys(mode_data.get("name","invalid_mode"))
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
                raise ValueError("Input of mode not a DeepA2 field.")
            if mode.target not in da2_angles:
                logging.error("Target of mode not a DeepA2 field: %s", mode)
                raise ValueError("Target of mode not a DeepA2 field.")

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
                inputs = []
                inputs.append(mode.target + ":")  # prefix
                for key in mode.input:  # inputs of mode
                    inputs.append(f"{key}: {da2_item[key]}")

                texts.append(" ".join(inputs))
                targets.append(da2_item[mode.target])

        t2t_item[self._input_column_name] = texts
        t2t_item[self._target_column_name] = targets

        return t2t_item

    def transform(self) -> None:
        """transforms sources"""

        t2t_datasets: Dict[str, List] = {}

        for source in self._sources:
            da2_dataset = datasets.load_dataset(**source)
            if isinstance(da2_dataset, datasets.DatasetDict):
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

        datasets.DatasetDict(
            {k: datasets.concatenate_datasets(v) for k, v in t2t_datasets.items()}
        ).save_to_disk(self._export_path)
