"""DeepA2 main app"""

import logging
import pathlib
from typing import Optional

import typer
import yaml

from deepa2.builder.core import Builder, Director, DatasetLoader
from deepa2.builder.aifdb_builder import (
    AIFDBBuilder,
    AIFDBLoader,
    RawAIFDBExample,
    PreprocessedAIFDBExample,
)
from deepa2.builder.nli_builder import (
    ESNLIBuilder,
    RawESNLIExample,
    PreprocessedESNLIExample,
)

logging.basicConfig(filename="deepa2.log", level=logging.DEBUG)

app = typer.Typer()


@app.command()
def build(  # pylint: disable=too-many-arguments
    name: Optional[str] = None,
    source_type: Optional[str] = None,
    path: Optional[str] = None,
    export_path: Optional[str] = None,
    debug_size: Optional[int] = None,
    configfile: Optional[str] = None,
):
    """
    Builds a new DeepA2 dataset
    """

    config = {}
    if configfile:
        config_path = pathlib.Path(configfile)
        if config_path.exists():
            with config_path.open(encoding="utf8") as yaml_file:
                config = yaml.load(yaml_file, Loader=yaml.Loader)
    # cmd-line args overwrite configfile
    if name:
        config["name"] = name
    if source_type:
        config["source_type"] = source_type
    elif name:
        config["source_type"] = name
    if path:
        config["path"] = path
    elif name:
        config["path"] = name
    if export_path:
        config["export_path"] = export_path
    if debug_size:
        config["debug_size"] = debug_size

    director = Director()
    builder: Builder
    if config["source_type"] == "esnli":
        builder = ESNLIBuilder(**config)
        dataset_loader = DatasetLoader(config["path"])
        director.raw_example_type = RawESNLIExample
        director.preprocessed_example_type = PreprocessedESNLIExample
    elif config["source_type"] == "aifdb":
        builder = AIFDBBuilder(**config)
        dataset_loader = AIFDBLoader(**config)
        director.raw_example_type = RawAIFDBExample
        director.preprocessed_example_type = PreprocessedAIFDBExample
    else:
        typer.echo(f"Unknown source_type: {config['source_type']}")

    director.builder = builder
    director.dataset_loader = dataset_loader
    director.transform(**config)


@app.command()
def preptrain():
    """
    Prepares a DeepA2 dataset for text-2-text training
    """

    typer.echo("Preparing deepa2 dataset")
