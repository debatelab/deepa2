# build_aifdb.py
from __future__ import annotations
import logging 
logging.basicConfig(filename='aifdb_builder.log', level=logging.DEBUG)

from deepa2datasets.builder import Director
from deepa2datasets.aifdb_builder import AIFDBBuilder
from deepa2datasets.config import moral_maze_config

from pathlib import Path
import requests, zipfile, io
import json
import datetime

from typing import Optional,List,Dict
import typer

from datasets import Dataset

app = typer.Typer()


def build_from_aifdb(aifdb_config, export_path: Optional[str] = None, debug_mode: Optional[bool] = False):
    """Exports all corpora contained (as subdirectories) in aifdb_dir as a single deepa2 dataset"""
    
    logging.info(f"#################################################################")
    logging.info(f"Starting new aifdb transformation: {datetime.datetime.now()}")

    splits = aifdb_config.get('splits')

    director = Director()
    builder = AIFDBBuilder()
    director.builder = builder

    # download and unpack corpora
    aifdb_dir = Path(aifdb_config.get('cache_dir'))
    logging.info(f"Downloading aifdb dataset to {aifdb_dir} ...")
    for url in aifdb_config.get('corpora',[]):
        destination = Path(aifdb_dir, url.split("/")[-1])
        if destination.is_dir():
            logging.debug(f"Using cached {destination}.")
        else:
            destination.mkdir(parents=True, exist_ok=True)
            logging.debug(f"Downloading {url}")
            r = requests.get(url+"/download")
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(str(destination.resolve()))    
            logging.debug(f"Saved {url} to {destination}.")


    # load aifdb dataset from disk
    data = {"nodeset":[],"text":[],"corpus":[]}
    for corpus_dir in aifdb_dir.iterdir():
        if corpus_dir.is_dir():
            for nodefile in corpus_dir.iterdir():
                if nodefile.suffix == '.json':
                    textfile = nodefile.parent / (nodefile.stem + ".txt")
                    data["nodeset"].append(json.load(nodefile.open()))
                    data["text"].append("".join(textfile.open().readlines()))
                    data["corpus"].append(corpus_dir.name)
    dataset = Dataset.from_dict(data)
    logging.info(f"Loaded aifdb dataset: {dataset}")
    
    # work on small subset for debugging
    if debug_mode:
        dataset = dataset.shuffle()
        dataset = dataset.filter(lambda ex,idx: 1 if (idx<100) else 0, with_indices=True)
        logging.info(f"Debug mode, working with filtered dataset: {dataset}")

    # preprocess 
    logging.info(f"Preprocessing aifdb dataset {aifdb_dir.name} ...")
    dataset = AIFDBBuilder.preprocess(dataset)
    logging.info(f"Preprocessed aifdb dataset {aifdb_dir.name}: {dataset}")

    # transform 
    new_dataset = dataset.map(director.transform, batched=True, batch_size=1)
    logging.info(f"Created new aifdb deepa2 dataset: {new_dataset}")

    # remove metadata
    if (not debug_mode) and ("metadata" in new_dataset.column_names):
        new_dataset = new_dataset.remove_columns("metadata")
        logging.info(f"Removed metadata from deepa2 dataset.")

    # create splits and save to disk
    new_dataset = new_dataset.shuffle(seed=42)
    split_sum = sum(splits.values())
    split_sizes = [int((w/split_sum)*new_dataset.num_rows) for w in splits.values()]
    split_ranges = {k:(sum(split_sizes[:i]),sum(split_sizes[:i+1])) for i,k in enumerate(splits.keys())}
    if export_path:
        for split_name,split_range in split_ranges.items():
            logging.info(f"Creating aifdb split {split_name} with index range {split_range}.")
            file_path = Path(export_path, aifdb_dir.name, split_name, f"{split_name}.parquet")
            logging.info(f"Saving split {file_path} ...")
            file_path.parent.mkdir(parents=True, exist_ok=True) # create dirs if necessary
            new_dataset.select(range(split_range)).to_parquet(file_path)
        logging.info(f"Saved aifdb deepa2 dataset {aifdb_dir.name}.")



@app.command()
def moral_maze(export_path: Optional[str] = None, debug_mode: Optional[bool] = False):
    """Reads aifdb moral maze corpora, preprocesses dataset, and builds da2-moralmaze"""
    aifdb_config = moral_maze_config
    build_from_aifdb(aifdb_config=aifdb_config, export_path=export_path, debug_mode=debug_mode)


@app.command()
def goodbye(name: str, formal: bool = False):
    if formal:
        typer.echo(f"Goodbye Ms. {name}. Have a good day.")
    else:
        typer.echo(f"Bye {name}!")


if __name__ == "__main__":
    app()