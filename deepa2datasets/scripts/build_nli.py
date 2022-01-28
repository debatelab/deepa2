# build_nli.py
from __future__ import annotations
import logging 
logging.basicConfig(filename='nli_builder.log', level=logging.DEBUG)

from deepa2datasets.builder import Director
from deepa2datasets.nli_builder import eSNLIBuilder, PreprocessedESNLIExample

from pathlib import Path
import datetime

from typing import Optional
import typer

from datasets import load_dataset, DatasetDict

app = typer.Typer()


@app.command()
def esnli(export_path: Optional[str] = None, debug_mode: Optional[bool] = False):
    """Reads e-snli, preprocesses dataset, and builds da2-esnli"""
    
    logging.info(f"#################################################################")
    logging.info(f"Starting new esnli transformation: {datetime.datetime.now()}")

    director = Director()
    builder = eSNLIBuilder()
    director.builder = builder

    # load esnli dataset from hf hub
    dataset:DatasetDict = load_dataset("esnli")
    logging.info(f"Loaded esnli dataset: {dataset}")

    # work on small subset for debugging
    if debug_mode:
        dataset = dataset.sort(column="premise")
        dataset = dataset.filter(lambda ex,idx: 1 if (idx<3000) else 0, with_indices=True)
        logging.info(f"Debug mode, working with filtered dataset: {dataset}")

    # preprocess each split
    for split in dataset.keys():
        logging.info(f"Preprocessing esnli split {split} ...")
        dataset[split] = eSNLIBuilder.preprocess(dataset[split])
    logging.info(f"Preprocessed esnli dataset: {dataset}")

    # transform
    new_dataset = dataset.map(director.transform, batched=True, batch_size=1, remove_columns=list(PreprocessedESNLIExample.__annotations__.keys()))
    logging.info(f"Created new esnli deepa2 dataset: {new_dataset}")

    # remove metadata
    if not debug_mode:
        new_dataset = new_dataset.remove_columns("metadata")
        logging.info(f"Removed metadata from deepa2 dataset")

    # save to disk
    if export_path:
        path = Path(export_path,"esnli")
        for split in new_dataset.keys():
            logging.info(f"Saving esnli split {split} ...")
            file_name = f"{split}.parquet"
            (path / split).mkdir(parents=True, exist_ok=True) # create dirs if necessary
            new_dataset[split].to_parquet(path / split / file_name)
        logging.info(f"Saved esnli deepa2 dataset to {path}.")

@app.command()
def goodbye(name: str, formal: bool = False):
    if formal:
        typer.echo(f"Goodbye Ms. {name}. Have a good day.")
    else:
        typer.echo(f"Bye {name}!")


if __name__ == "__main__":
    app()