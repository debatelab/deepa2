"""Typer app for building DeepA2 datasets afrom AIFdb corpora."""

import logging

from typing import Optional
import typer

from deepa2datasets.core import Director
from deepa2datasets.aifdb_builder import (
    AIFDBBuilder,
    AIFDBLoader,
    AIFDBConfig,
    RawAIFDBExample,
    PreprocessedAIFDBExample,
)
from deepa2datasets.config import data_dir

logging.basicConfig(filename="aifdb_builder.log", level=logging.DEBUG)

app = typer.Typer()


def build_from_aifdb(
    aifdb_config: AIFDBConfig,
    export_path: Optional[str] = None,
    debug_size: Optional[int] = None,
):
    """
    Exports all corpora contained (as subdirectories) in aifdb_dir as a single deepa2 dataset
    """

    name = aifdb_config.name

    director = Director()
    builder = AIFDBBuilder(aifdb_config)
    dataset_loader = AIFDBLoader(aifdb_config)
    director.builder = builder
    director.dataset_loader = dataset_loader
    director.raw_example_type = RawAIFDBExample
    director.preprocessed_example_type = PreprocessedAIFDBExample

    director.transform(export_path=export_path, debug_size=debug_size, name=name)


@app.command()
def moral_maze(export_path: Optional[str] = None, debug_size: Optional[int] = None):
    """
    Loads aifdb moral maze corpora, preprocesses dataset, and builds da2-moralmaze.
    """
    moral_maze_config = AIFDBConfig(
        name="moral-maze",
        cache_dir=data_dir / "raw" / "aifdb" / "moral-maze",
        corpora=[
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
        ],
    )
    build_from_aifdb(
        aifdb_config=moral_maze_config, export_path=export_path, debug_size=debug_size
    )


@app.command()
def vacc_itc(export_path: Optional[str] = None, debug_size: Optional[int] = None):
    """
    Loads aifdb VaccITC corpora, preprocesses dataset, and builds da2-vaccitc.
    """
    vacc_itc_config = AIFDBConfig(
        name="vacc-itc",
        cache_dir=data_dir / "raw" / "aifdb" / "vacc-itc",
        corpora=[
            "http://corpora.aifdb.org/zip/VaccITC1",
            "http://corpora.aifdb.org/zip/VaccITC2",
            "http://corpora.aifdb.org/zip/VaccITC3",
            "http://corpora.aifdb.org/zip/VaccITC4",
            "http://corpora.aifdb.org/zip/VaccITC5",
            "http://corpora.aifdb.org/zip/VaccITC6",
            "http://corpora.aifdb.org/zip/VaccITC7",
            "http://corpora.aifdb.org/zip/VaccITC8",
        ],
    )
    build_from_aifdb(
        aifdb_config=vacc_itc_config, export_path=export_path, debug_size=debug_size
    )


@app.command()
def us2016(export_path: Optional[str] = None, debug_size: Optional[int] = None):
    """
    Loads aifdb us2016 presidiential debate corpora, preprocesses dataset, and builds da2-vaccitc.
    """
    us2016_config = AIFDBConfig(
        name="us2016",
        cache_dir=data_dir / "raw" / "aifdb" / "us2016",
        corpora=[
            "http://corpora.aifdb.org/zip/US2016",
        ],
    )
    build_from_aifdb(
        aifdb_config=us2016_config, export_path=export_path, debug_size=debug_size
    )


if __name__ == "__main__":
    app()
