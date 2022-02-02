# build_aifdb.py
from __future__ import annotations
import logging 
logging.basicConfig(filename='aifdb_builder.log', level=logging.INFO)

from deepa2datasets.core import Director
from deepa2datasets.aifdb_builder import AIFDBBuilder,AIFDBLoader,RawAIFDBExample,PreprocessedAIFDBExample
from deepa2datasets.config import moral_maze_config


from typing import Optional
import typer


app = typer.Typer()


def build_from_aifdb(aifdb_config, export_path: Optional[str] = None, debug_size: Optional[int] = None):
    """
    Exports all corpora contained (as subdirectories) in aifdb_dir as a single deepa2 dataset
    """
    
    name = aifdb_config.get("name","default_aifdb")

    director = Director()
    builder = AIFDBBuilder()
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
    aifdb_config = moral_maze_config
    build_from_aifdb(aifdb_config=aifdb_config, export_path=export_path, debug_size=debug_size)


@app.command()
def goodbye(name: str, formal: bool = False):
    if formal:
        typer.echo(f"Goodbye Ms. {name}. Have a good day.")
    else:
        typer.echo(f"Bye {name}!")


if __name__ == "__main__":
    app()