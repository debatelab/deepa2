from deepa2datasets.builder import ArgdownStatement, Builder, QuotedStatement
from deepa2datasets.builder import DeepA2Item
from deepa2datasets.config import template_dir,package_dir
import deepa2datasets.jinjafilters as jjfilters

import random

from datasets import Dataset
from tqdm import tqdm
tqdm.pandas()

from networkx.readwrite import json_graph
import networkx as nx

from jinja2 import Environment, FileSystemLoader, select_autoescape
import re

from typing import Any,List,Dict
from pathlib import Path
import uuid
import logging

from dataclasses import dataclass, field, asdict



class AIFDBBuilder(Builder):
    """
    The Concrete Builder classes follow the Builder interface and provide
    specific implementations of the building steps. Your program may have
    several variations of Builders, implemented differently.
    """

    raw_aifdb_features = ["nodeset","text","corpus"]
    preprocessed_aifdb_features = ["text","corpus","reasons","conjectures","premises","conclusions", "type"]


    def preprocess(dataset:Dataset) -> Dataset:

        # clean html
        CLEANR = re.compile('<.*?>') 
        def cleanhtml(example):
            example["text"] = re.sub(CLEANR, '', example["text"])
            return example
        dataset = dataset.map(cleanhtml)

        # split per inference
        def split_nodeset_per_inference(examples:Dict[str,List]) -> Dict[str,List]:
            inference_chunks = {k:[] for k in AIFDBBuilder.preprocessed_aifdb_features}
            # for each example
            for i,nodeset in enumerate(examples["nodeset"]):
                # initialize graph representing the argumentative analysis
                nodeset['directed']=True
                attrs = {'source':'fromID', 'target':'toID', 'name':'nodeID','key':'key', 'link':'edges'}
                G = json_graph.node_link_graph(nodeset,attrs=attrs)
                node_type = nx.get_node_attributes(G, "type")
                node_text = nx.get_node_attributes(G, "text")

                # get all nodes of type CA / RA
                inference_nodes = [n for n in G.nodes if node_type[n] in ["CA","RA"]]
                # each inference node gives rise to a separate chunk
                for inference_node in inference_nodes:
                    # get conclusion
                    conclusions = [n for n in G.successors(inference_node) if node_type[n]=="I"]
                    # get premises
                    premises = [n for n in G.predecessors(inference_node) if node_type[n]=="I"]
                    # get conjectures and reasons
                    def get_L_grandparent(node):
                        if node_type[node]!="I":
                            return None
                        ya_predecessors = [n for n in G.predecessors(node) if node_type[n]=="YA"]     
                        if not ya_predecessors:
                            return None
                        l_grandparents = [n for m in ya_predecessors for n in G.predecessors(m) if node_type[n]=="L" and node_text[n]!="analyses"]
                        return l_grandparents
                    conjectures = sorted([get_L_grandparent(n) for n in conclusions])
                    reasons = sorted([get_L_grandparent(n) for n in premises])
                    # create new record
                    inference_chunks["text"].append(examples["text"][i])
                    inference_chunks["corpus"].append(examples["corpus"][i])
                    inference_chunks["premises"].append(premises)
                    inference_chunks["conclusions"].append(conclusions)
                    inference_chunks["reasons"].append(reasons)
                    inference_chunks["conjectures"].append(conjectures)
                    inference_chunks["type"].append(node_type[inference_node])
            logging.debug({k:len(v) for k,v in inference_chunks.items()})
            return inference_chunks
        dataset = dataset.map(split_nodeset_per_inference, batched=True, remove_columns=dataset.column_names)

        return dataset




    def __init__(self) -> None:
        """
        A fresh builder instance should contain a blank product object, which is
        used in further assembly.
        """
        # check whether template files are accessible
        #if not (template_dir / "esnli").exists():
        #    logging.debug(f"Package dir: {package_dir}")
        #    logging.debug(f"Resolve template dir: {template_dir}")
        #    logging.debug(f"List template dir: {list(template_dir.glob('*'))}")
        #    err_m = f'No "esnli" subdirectory in template_dir {template_dir.resolve()}'
        #    raise ValueError(err_m)
        #self._env = Environment(
        #    loader = FileSystemLoader(template_dir),
        #    autoescape=select_autoescape()
        #)

        self.reset()

    def reset(self) -> None:
        self._input = {}
        self._product:List[DeepA2Item] = []

    @property
    def product(self) -> List[Dict]:
        """
        Concrete Builders are supposed to provide their own methods for
        retrieving results. That's because various types of builders may create
        entirely different products that don't follow the same interface.
        Therefore, such methods cannot be declared in the base Builder interface
        (at least in a statically typed programming language).

        Usually, after returning the end result to the client, a builder
        instance is expected to be ready to start producing another product.
        That's why it's a usual practice to call the reset method at the end of
        the `getProduct` method body. However, this behavior is not mandatory,
        and you can make your builders wait for an explicit reset call from the
        client code before disposing of the previous result.
        """
        product = self._product
        product = [asdict(rec) for rec in product]
        self.reset()
        return product



    def fetch_batch(self, input_batch) -> None:
        """
        Fetches items to be processed for building product from input batch.
        """
        ### sanity checks
        # features present?
        if not all(f in input_batch.keys() for f in self.preprocessed_aifdb_features):
            logging.warning(f"incomplete aifdb batch with keys {str(list(input_batch.keys()))}.")
            return None

        self._input = input_batch




    def configure_product(self) -> None:
        pass

    def produce_da2item(self) -> None:
        for i,_ in enumerate(self._product):
            self.populate_record(i)


    def populate_record(self,i) -> None:
        record = self._product[i]
        # choose template for paraphrased source according to type of relation between premise and conclusion
        pass


    def postprocess_da2item(self) -> None:
        pass

    def add_metadata_da2item(self) -> None:
        pass
