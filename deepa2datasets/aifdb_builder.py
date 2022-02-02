from __future__ import annotations

from deepa2datasets.core import Builder, DatasetLoader
from deepa2datasets.core import DeepA2Item, QuotedStatement, PreprocessedExample, RawExample
from deepa2datasets.config import template_dir, package_dir, moral_maze_config

import random

from datasets import Dataset, DatasetDict
from tqdm import tqdm
tqdm.pandas()

from pathlib import Path
import requests, zipfile, io
import json

from networkx.readwrite import json_graph
import networkx as nx

from jinja2 import Environment, FileSystemLoader, select_autoescape
import re

from typing import Any,List,Dict,Union,Optional
import logging



class RawAIFDBExample(RawExample):
    nodeset:Union[str,List[str]]
    text:Union[str,List[str]]
    corpus:Union[str,List[str]]

class PreprocessedAIFDBExample(PreprocessedExample):
    text:Union[str,List[str]]  
    corpus:Union[str,List[str]]
    type:Union[str,List[str]]
    reasons:Union[List[str],List[Any]]
    conjectures:Union[List[str],List[Any]]
    premises:Union[List[str],List[Any]]
    conclusions:Union[List[str],List[Any]]


class AIFDBLoader(DatasetLoader):
    def __init__(self, aifdb_config:Optional[Dict]=None):
        self._aifdb_config = aifdb_config
    def load_dataset(self) -> DatasetDict:
        splits = self._aifdb_config.get('splits')

        # download and unpack corpora
        aifdb_dir = Path(self._aifdb_config.get('cache_dir'))
        logging.info(f"Downloading aifdb dataset to {aifdb_dir} ...")
        for url in self._aifdb_config.get('corpora',[]):
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
                        if textfile.exists():
                            data["nodeset"].append(json.load(nodefile.open()))
                            data["text"].append("".join(textfile.open().readlines()))
                            data["corpus"].append(corpus_dir.name)
        dataset = Dataset.from_dict(data)

        # create train-validation-test splits 
        dataset = dataset.train_test_split(test_size=(1-splits["train"])) # split once
        dataset_tmp = dataset["test"].train_test_split(test_size=(splits["test"]/(splits["test"]+splits["validation"]))) # split test-split again
        dataset = DatasetDict(train=dataset["train"],validation=dataset_tmp["train"],test=dataset_tmp["test"])


        return dataset


class AIFDBBuilder(Builder):
    """
    AIFDBBuilder preprocesses, splits, and transforms AIFdb nodesets into DeepA2 items
    """

    @staticmethod
    def preprocess(dataset:Dataset) -> Dataset:

        # clean html
        CLEANR = re.compile('<.*?>') 
        def cleanhtml(example):
            example["text"] = re.sub(CLEANR, '', example["text"])
            return example
        dataset = dataset.map(cleanhtml)

        # split per inference
        def split_nodeset_per_inference(examples:Dict[str,List]) -> Dict[str,List]:
            inference_chunks = {k:[] for k in PreprocessedAIFDBExample.__annotations__.keys()}
            # for each example
            for i,nodeset in enumerate(examples["nodeset"]):
                # initialize graph representing the argumentative analysis
                nodeset['directed']=True
                attrs = {'source':'fromID', 'target':'toID', 'name':'nodeID','key':'key', 'link':'edges'}
                G = json_graph.node_link_graph(nodeset,attrs=attrs)
                node_type = nx.get_node_attributes(G, "type")
                #logging.debug(f"node types: {node_type}")
                node_text = nx.get_node_attributes(G, "text")
                if not (node_type and node_text):
                    logging.warning(f"No node types / texts in nodeset no {i} in corpus {examples['corpus'][i]}: skipping this nodeset.")
                    continue

                # construct alternative_text by joining L-nodes
                alternative_text = [node_text.get(n,"") for n in G.nodes if node_type.get(n,None)=="L"] # L-nodes
                alternative_text = " ".join(alternative_text)
                alternative_text = alternative_text.replace("  "," ")

                # use longer text
                text = examples["text"][i]
                if len(alternative_text) > 2*(len(text)-text.count("\n")):
                    logging.debug(f"Using alternative text '{alternative_text}' rather than original text '{text}' in corpus '{examples['corpus'][i]}'.")
                    text = alternative_text

                # get all nodes of type CA / RA
                inference_nodes = [n for n in G.nodes if node_type.get(n,None) in ["CA","RA"]]
                # each inference node gives rise to a separate chunk
                for inference_node in inference_nodes:
                    # get conclusion (ids)
                    conclusions = [n for n in G.successors(inference_node) if node_type[n]=="I"]
                    # get premises (ids)
                    premises = [n for n in G.predecessors(inference_node) if node_type[n]=="I"]
                    # get conjectures and reasons (ids)
                    def get_L_grandparent(node):
                        if node_type[node]!="I":
                            return None
                        ya_predecessors = [n for n in G.predecessors(node) if node_type[n]=="YA"]     
                        if not ya_predecessors:
                            return None
                        l_grandparents = [n for m in ya_predecessors for n in G.predecessors(m) if node_type[n]=="L" and node_text[n]!="analyses"]
                        return l_grandparents
                    conjectures = [get_L_grandparent(n) for n in conclusions]
                    if conjectures:
                        conjectures = [x for l in conjectures if l for x in l] # flatten 
                        conjectures = sorted(conjectures) # sort, ids correspond to location in text 
                    reasons = [get_L_grandparent(n) for n in premises]
                    if reasons:
                        reasons = [x for l in reasons if l for x in l] # flatten 
                        reasons = sorted(reasons) # sort, ids correspond to location in text
                    # subst text for ids
                    conjectures = [node_text[n] for n in conjectures] 
                    reasons = [node_text[n] for n in reasons] 
                    conclusions = [node_text[n] for n in conclusions] 
                    premises = [node_text[n] for n in premises] 
                    # create new record
                    inference_chunks["text"].append(text)
                    inference_chunks["corpus"].append(examples["corpus"][i])
                    inference_chunks["premises"].append(premises)
                    inference_chunks["conclusions"].append(conclusions)
                    inference_chunks["reasons"].append(reasons)
                    inference_chunks["conjectures"].append(conjectures)
                    inference_chunks["type"].append(node_type[inference_node])
            logging.debug({k:len(v) for k,v in inference_chunks.items()})
            return inference_chunks
        dataset = dataset.map(split_nodeset_per_inference, batched=True, remove_columns=dataset.column_names)

        ## check features
        assert dataset.column_names == list(PreprocessedAIFDBExample.__annotations__.keys())

        return dataset




    def __init__(self) -> None:
        """
        A fresh builder instance should contain a blank product object, which is
        used in further assembly.
        """
        # check whether template files are accessible
        if not (template_dir / "aifdb").exists():
            logging.debug(f"Package dir: {package_dir}")
            logging.debug(f"Resolve template dir: {template_dir}")
            logging.debug(f"List template dir: {list(template_dir.glob('*'))}")
            err_m = f'No "aifdb" subdirectory in template_dir {template_dir.resolve()}'
            raise ValueError(err_m)
        self._env = Environment(
            loader = FileSystemLoader(template_dir),
            autoescape=select_autoescape()
        )


        self.reset()


    @property
    def input(self) -> PreprocessedAIFDBExample:
        """
        The input of any builder is a proprocessed example
        """
        return self._input

    @input.setter
    def input(self, preprocessed_example: PreprocessedAIFDBExample) -> None:
        """
        Sets input for building next product.
        """
        # unbatch:
        self._input = {k:v[0] for k,v in preprocessed_example.items()}

    def configure_product(self) -> None:
        # create empty DeepA2Items and store them in _product
        itype = self._input['type']
        sp_template = random.choice(moral_maze_config["templates_sp_ra"] if itype=="RA" else moral_maze_config["templates_sp_ca"])
        metadata = {
            "corpus": self._input['corpus'],
            "type": itype,
            "config": {'sp_template':sp_template}
        }
        self._product.append(DeepA2Item(metadata=metadata))


    def produce_da2item(self) -> None:
        # we produce a single da2item per input only
        record = self._product[0]
        record.argument_source = self.input["text"]
        record.reason_statements = [QuotedStatement(text=r,starts_at=None,ref_reco=e) for e,r in enumerate(self.input['reasons'])]
        record.conclusion_statements = [QuotedStatement(text=j,starts_at=None,ref_reco=len(record.reason_statements)+1) for j in self.input['conjectures']]
        # source paraphrase
        sp_template = self._env.get_template(record.metadata["config"]["sp_template"])
        record.source_paraphrase = (sp_template.render(premises=self.input['premises'],conclusion=self.input['conclusions']))


    def postprocess_da2item(self) -> None:
        pass

    def add_metadata_da2item(self) -> None:
        pass
