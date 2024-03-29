"""DeepA2 main app"""

import dataclasses
import logging
import pathlib
import pprint
import sys
from typing import Optional

import typer
import yaml

import deepa2
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
from deepa2.builder.arg_q_builder import (
    ArgQBuilder,
    ArgQLoader,
    PreprocessedArgQExample,
    RawArgQExample,
)
from deepa2.builder.arg_kp_builder import (
    ArgKPBuilder,
    ArgKPLoader,
    PreprocessedArgKPExample,
    RawArgKPExample,
)
from deepa2.builder.nli_builder import (
    ESNLIBuilder,
    RawESNLIExample,
    PreprocessedESNLIExample,
)
from deepa2.builder.entailmentbank_builder import (
    EnBankBuilder,
    EnBankLoader,
    RawEnBankExample,
    PreprocessedEnBankExample,
)
from deepa2.preptrain import T2TPreprocessor

logging.basicConfig(filename="deepa2.log", level=logging.DEBUG)

app = typer.Typer()


@app.command()  # noqa: C901
def bake(  # noqa: C901  pylint: disable=too-many-arguments,too-many-branches,too-many-statements
    source_type: Optional[str] = typer.Option(
        None,
        help="type of the source dataset, used to"
        "choose a compatible Builder; currently supported source types:"
        "`esnli`, `aifdb`, `enbank`, `argq`, `argkp`.",
    ),
    name: Optional[str] = typer.Option(
        None,
        help="name of preconfigured dataset(s) to load given `source_type`; "
        "see documentation of Builders for more info.",
    ),
    path: Optional[str] = typer.Option(None, help="path to the input dataset"),
    export_path: Optional[str] = typer.Option(
        None, help="local directory to which built DeepA2 dataset is saved."
    ),
    debug_size: Optional[int] = typer.Option(
        None, help="number of items to process for debugging"
    ),
    configfile: Optional[str] = typer.Option(
        None,
        help="path to yml file that contains a configuration "
        "for `deepa2 bake`. The configfile will typically set "
        "builder-specific parameters, see documentation of Builders "
        "for more info.",
    ),
):
    """
    Builds a new 🎂 DeepA2 dataset from `path` using a Builder that
    fits `source_type`.
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

    if "source_type" not in config:
        typer.echo("No `source_type` provided. Idle and exiting.")
        sys.exit(-1)

    director = Director()
    builder: Builder
    if config.get("source_type") == "esnli":
        builder = ESNLIBuilder(**config)
        dataset_loader = DatasetLoader(config["path"])
        director.raw_example_class = RawESNLIExample
        director.preprocessed_example_class = PreprocessedESNLIExample
    elif config.get("source_type") == "aifdb":
        builder = AIFDBBuilder(**config)
        dataset_loader = AIFDBLoader(**config)
        director.raw_example_class = RawAIFDBExample
        director.preprocessed_example_class = PreprocessedAIFDBExample
    elif config.get("source_type") == "enbank":
        builder = EnBankBuilder(**config)
        dataset_loader = EnBankLoader(**config)
        director.raw_example_class = RawEnBankExample
        director.preprocessed_example_class = PreprocessedEnBankExample
    elif config.get("source_type") == "argq":
        builder = ArgQBuilder(**config)
        dataset_loader = ArgQLoader(**config)
        director.raw_example_class = RawArgQExample
        director.preprocessed_example_class = PreprocessedArgQExample
    elif config.get("source_type") == "argkp":
        builder = ArgKPBuilder(**config)
        dataset_loader = ArgKPLoader(**config)
        director.raw_example_class = RawArgKPExample
        director.preprocessed_example_class = PreprocessedArgKPExample
    else:
        typer.echo(f"Unknown source_type: {config.get('source_type')}")
        sys.exit(-1)

    director.builder = builder
    director.dataset_loader = dataset_loader
    director.transform(**config)


@app.command()
def serve(  # pylint: disable=too-many-arguments
    path: Optional[str] = typer.Option(None, help="path to DeepA2 dataset"),
    revision: Optional[str] = typer.Option(
        None, help="version of the dataset (script) to load"
    ),
    export_path: Optional[str] = typer.Option(
        None, help="local directory to which t2t dataset is saved"
    ),
    export_format: Optional[str] = typer.Option(
        None,
        help="format in t2t dataset is saved (parquet, csv, "
        "jsonl, arrow), will use arrow if left blank",
    ),
    input_column_name: Optional[str] = typer.Option(
        "text", help="name of input column of t2t dataset"
    ),
    target_column_name: Optional[str] = typer.Option(
        "target", help="name of target column of t2t dataset"
    ),
    mask_probability: Optional[float] = typer.Option(
        None, help="probability that an input sequence will be masked"
    ),
    configfile: Optional[str] = typer.Option(
        None,
        help="path to yml configuration while; commandline "
        "options overwrite config file; using a config file "
        "allows for serving multiple deepa2 datasets as a "
        "single t2t dataset; generative modes covered can"
        "also be specified in config file.",
    ),
):
    """
    Prepares 🎂 DeepA2 datasets for text-2-text training, and
    serves a single 🍰 t2t dataset.
    """

    config = {}
    if configfile:
        config_path = pathlib.Path(configfile)
        if config_path.exists():
            with config_path.open(encoding="utf8") as yaml_file:
                config = yaml.load(yaml_file, Loader=yaml.Loader)
        else:
            typer.echo(f"No configfile: {config_path}.")
    # cmd-line args overwrite configfile
    if path:
        config["sources"] = [{"path": path, "revision": revision}]
    if export_path:
        config["export_path"] = export_path
    if export_format:
        config["export_format"] = export_format
    if mask_probability:
        config["mask_probability"] = mask_probability
    config["input_column_name"] = input_column_name
    config["target_column_name"] = target_column_name

    if "sources" not in config:
        typer.echo(
            "No source dataset specified, exiting without having run preprocessor."
        )
        sys.exit(-1)

    if "export_path" not in config:
        typer.echo("No export path specified, defaulting to ./exported.")
        config["export_path"] = "exported"

    if "mask_probability" not in config:
        typer.echo("No mask_probability specified, defaulting to 0.")
        config["mask_probability"] = 0.0

    if "export_format" not in config:
        typer.echo("No export format specified, saving datasetdict as arrow tables.")
        config["export_format"] = "arrow"
    elif config["export_format"] not in ["csv", "jsonl", "parquet", "arrow"]:
        typer.echo(
            f"Unknown export format: {config['export_format']}, defaulting to parquet."
        )
        config["export_format"] = "parquet"

    t2t_preprocessor = T2TPreprocessor(**config)
    t2t_preprocessor.transform()


@app.command()
def keys():
    """prints keys for specifying generative modes, and corresponding features."""

    typer.echo(pprint.pformat(dataclasses.asdict(deepa2.DA2_ANGLES_MAP())))
