# build_nli.py
# typer build_nli.py run bye --name Camila

from deepa2datasets.builder import Director
from deepa2datasets.nli_builder import eSNLIBuilder

import logging 
from typing import Optional
import typer

from datasets import load_dataset, DatasetDict

app = typer.Typer()
logger = logging.getLogger(__name__)


@app.command()
def esnli(name: Optional[str] = None):
    """Reads e-snli, preprocesses dataset, and builds da2-esnli"""

    logger.setLevel(level=logging.INFO)
    logger.warning("Warning works fine.")

    director = Director()
    builder = eSNLIBuilder()
    director.builder = builder

    # load esnli dataset from hf hub
    dataset:DatasetDict = load_dataset("esnli")
    # preprocess each split
    for split in dataset.keys():
        dataset[split] = eSNLIBuilder.preprocess_esnli(dataset[split])
    # transform (batches of size three with <E,C,N>)
    new_dataset = dataset.map(Director.transform, batched=True, batch_size=3, remove_columns=dataset.column_names)

    # save & upload to hub
    if False:
        new_dataset.save_to_disk(dataset_dict_path)
        new_dataset.push_to_hub()

    print("\n")

    print(f"Created new dataset: {new_dataset}")


@app.command()
def bye(name: Optional[str] = None):
    """Dummy function to test typer"""
    if name:
        typer.echo(f"Bye {name}")
    else:
        typer.echo("Goodbye!")


if __name__ == "__main__":
    app()