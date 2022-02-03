"""Typer app for converting NLI-type data to DeepA2.
"""
from __future__ import annotations
import logging

from typing import Optional
import typer

from deepa2datasets.core import Director, DatasetLoader
from deepa2datasets.nli_builder import (
    eSNLIBuilder,
    RawESNLIExample,
    PreprocessedESNLIExample,
)

logging.basicConfig(filename="nli_builder.log", level=logging.INFO)

app = typer.Typer()


@app.command()
def esnli(export_path: Optional[str] = None, debug_size: Optional[int] = None):
    """Loads and preprocesses the raw esnli dataset, builds da2-esnli."""

    name = "esnli"

    director = Director()
    builder = eSNLIBuilder()
    dataset_loader = DatasetLoader("esnli")  # using default Dataset Loader
    director.builder = builder
    director.dataset_loader = dataset_loader
    director.raw_example_type = RawESNLIExample
    director.preprocessed_example_type = PreprocessedESNLIExample

    director.transform(export_path=export_path, debug_size=debug_size, name=name)


@app.command()
def goodbye(name: str, formal: bool = False):
    if formal:
        typer.echo(f"Goodbye Ms. {name}. Have a good day.")
    else:
        typer.echo(f"Bye {name}!")


if __name__ == "__main__":
    app()
