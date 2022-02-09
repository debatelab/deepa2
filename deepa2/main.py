"""DeepA2 main app"""

import logging
import pathlib
import sys
from typing import Optional

import typer
import yaml

from deepa2.builder import (
    Builder,
    Director,
    DatasetLoader,
)
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
from deepa2.preptrain import T2TPreprocessor

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
        director.raw_example_class = RawESNLIExample
        director.preprocessed_example_class = PreprocessedESNLIExample
    elif config["source_type"] == "aifdb":
        builder = AIFDBBuilder(**config)
        dataset_loader = AIFDBLoader(**config)
        director.raw_example_class = RawAIFDBExample
        director.preprocessed_example_class = PreprocessedAIFDBExample
    else:
        typer.echo(f"Unknown source_type: {config['source_type']}")

    director.builder = builder
    director.dataset_loader = dataset_loader
    director.transform(**config)


@app.command()
def preptrain(  # pylint: disable=too-many-arguments
    path: Optional[str] = None,
    revision: Optional[str] = None,
    export_path: Optional[str] = None,
    input_column_name: Optional[str] = "text",
    target_column_name: Optional[str] = "target",
    configfile: Optional[str] = None,
):
    """
    Prepares a DeepA2 dataset for text-2-text training
    """

    config = {}
    if configfile:
        config_path = pathlib.Path(configfile)
        if config_path.exists():
            with config_path.open(encoding="utf8") as yaml_file:
                config = yaml.load(yaml_file, Loader=yaml.Loader)
    # cmd-line args overwrite configfile
    if path:
        config["sources"] = [{"path": path, "revision": revision}]
    if export_path:
        config["export_path"] = export_path
    config["input_column_name"] = input_column_name
    config["target_column_name"] = target_column_name

    if "sources" not in config:
        typer.echo(
            "No source dataset specified, exiting without having run preprocessor."
        )
        sys.exit(-1)

    if "export_path" not in config:
        logging.warning("No export specified, defaulting to ./exported.")
        config["export_path"] = "exported"

    t2t_preprocessor = T2TPreprocessor(**config)
    t2t_preprocessor.transform()
