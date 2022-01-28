from deepa2datasets.builder import ArgdownStatement, Builder, Formalization, QuotedStatement
from deepa2datasets.builder import DeepA2Item
from deepa2datasets.config import template_dir,package_dir
import deepa2datasets.jinjafilters as jjfilters

import random

from datasets import Dataset
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

from jinja2 import Environment, FileSystemLoader, select_autoescape

from typing import Any, List, Dict, TypedDict, Union
from pathlib import Path
import uuid
import logging

from dataclasses import dataclass, field, asdict


class RawESNLIExample(TypedDict):
     premise:Union[str,List[str]]
     hypothesis:Union[str,List[str]]
     label:Union[str,List[str]]
     explanation_1:Union[str,List[str]]
     explanation_2:Union[str,List[str]]
     explanation_3:Union[str,List[str]]


class PreprocessedESNLIExample(TypedDict):
    premise:Union[str,List[str]]  
    hypothesis_ent:Union[str,List[str]]
    hypothesis_neu:Union[str,List[str]]
    hypothesis_con:Union[str,List[str]]
    explanation_ent:Union[List[str],List[Any]]
    explanation_neu:Union[List[str],List[Any]]
    explanation_con:Union[List[str],List[Any]]


@dataclass
class eSNLIConfiguration():
    label:str
    argdown_template_path:str = "esnli/argdown_generic.txt"
    argdown_err_template_path:str = None
    source_paraphrase_template_path:str = "esnli/source_paraphrase.txt"
    scheme_name:str = "modus ponens"
    formal_scheme:List = field(default_factory=lambda: ["{p}","{p} -> {q}", "{q}"])
    placeholders:Dict = field(default_factory=lambda: {'p':"{premise}", "q":"{hypothesis}"})

    _nl_schemes_dict = {
        "{p}": "{{ {p} | lower }}",
        "{q}": "{{ {q} | lower }}",
        "¬{p}": "{{ {p} | negation }}",
        "¬{q}": "{{ {q} | negation }}",
        "{p} -> {q}": "{{ {p} | conditional({q}) }}",
        "{p} -> ¬{q}": "{{ {p} | conditional({q} | negation) }}",
    }

    @property
    def nl_scheme(self):
        placeholders = {k:v.strip("{}") for k,v in self.placeholders.items()}
        nl_scheme = [self._nl_schemes_dict[s].format(**placeholders) for s in self.formal_scheme]
        nl_scheme = ["{"+s+"}" for s in nl_scheme] # postprocess: re-add {} which got lost in previous format() call
        assert all((s[:2]=="{{" and s[-2:]=="}}") for s in nl_scheme) # jinja2 templates?
        return nl_scheme


class eSNLIBuilder(Builder):
    """
    The Concrete Builder classes follow the Builder interface and provide
    specific implementations of the building steps. Your program may have
    several variations of Builders, implemented differently.
    """

    # list of errorneous argdown templates

    argdown_err_templates = [
        "esnli/argdown_err-01.txt",
        "esnli/argdown_err-02.txt",
        "esnli/argdown_err-03.txt",
        "esnli/argdown_err-04.txt",
        "esnli/argdown_err-05.txt",
        "esnli/argdown_err-06.txt",
        "esnli/argdown_err-07.txt",
        "esnli/argdown_err-08.txt",
        "esnli/argdown_err-09.txt",
    ]


    def preprocess(dataset:Dataset) -> Dataset:
        df_esnli = dataset.to_pandas()
        df_esnli = df_esnli.drop_duplicates()
        # count explanations per row
        df_esnli["n_explanations"] = 3-df_esnli[['explanation_1','explanation_2','explanation_3']].eq("").sum(axis=1)
        # keep records with at least one explanation
        df_esnli = df_esnli[df_esnli.n_explanations.ge(1)]
        # count how frequently premise occurs in the dataset (default = three times)
        counts = df_esnli.groupby(["premise"]).size()
        tqdm.write("Preprocessing 1/7")
        df_esnli["premise_counts"] = df_esnli.premise.progress_apply(lambda x: counts[x])
        # drop records whose premise occurs less than 3 times
        df_esnli = df_esnli[df_esnli.premise_counts.ge(3)]

        # we split df in two parts which will be processed separately and are finally merged

        ## Split 1
        # get all rows whose premise occurs more than 3 times
        df_esnli_tmp = df_esnli[df_esnli.premise_counts.gt(3)].copy()
        df_esnli_tmp.reset_index(inplace=True)
        # for each premise, what is the minimum number of labels?
        df2 = df_esnli_tmp.groupby(["premise","label"]).size().unstack()
        df2.fillna(0,inplace=True)
        tqdm.write("Preprocessing 2/7")
        df_esnli_tmp["min_label_counts"] = df_esnli_tmp.premise.progress_apply(lambda x: int(df2.min(axis=1)[x]))     # df2.min(axis=1) tells us how many records for each premise will go into preprocessed esnli dataset
        # make sure that for each premise, we have the same number of records for labels 0,1,2
        tqdm.write("Preprocessing 3/7")
        df_esnli_tmp = df_esnli_tmp.groupby(["premise","label"],as_index=False).progress_apply(lambda x: x.iloc[:x.min_label_counts.iloc[0]])
        # reorder row so as to obtain alternating labels
        def reorder_premise_group(pg):
            return pg.groupby("label").apply(lambda g: g.reset_index(drop=True)).sort_index(level=1)
        tqdm.write("Preprocessing 4/7")
        df_esnli_tmp = df_esnli_tmp.groupby(["premise"],as_index=False).progress_apply(reorder_premise_group)

        ## Split 2
        # get all rows whose premise occurs exactly 3 times
        df_esnli_tmp2 = df_esnli[df_esnli.premise_counts.eq(3)].copy()
        # determine premises with incomplete labels (at least one label is missing)
        tqdm.write("Preprocessing 5/7")
        labels_complete = df_esnli_tmp2.groupby(["premise"]).progress_apply(lambda g: len(set(g["label"]))==3)
        tqdm.write("Preprocessing 6/7")
        df_esnli_tmp2["complete"] = df_esnli_tmp2.premise.progress_apply(lambda x: labels_complete[x])
        # retain only complete records
        df_esnli_tmp2 = df_esnli_tmp2[df_esnli_tmp2.complete]

        ## Merge
        df_esnli_final = pd.concat([
            df_esnli_tmp2[list(RawESNLIExample.__annotations__.keys())],
            df_esnli_tmp[list(RawESNLIExample.__annotations__.keys())]
        ])
        df_esnli_final.reset_index(drop=True,inplace=True)

        ## Sanity check
        tqdm.write("Preprocessing 7/7")
        for start in tqdm(range(0, df_esnli_final.shape[0], 3)):
            triple = df_esnli_final.iloc[start:start + 3]
            assert len(set(triple.premise))==1
            assert len(set(triple.label))==3

        ## we now merge any three raw items into a single record
        dataset = Dataset.from_pandas(df_esnli_final)

        def merge_triple_batches(examples:RawESNLIExample) -> PreprocessedESNLIExample:
            ### sanity checks
            # features present?
            if not all(f in examples.keys() for f in RawESNLIExample.__annotations__.keys()):
                logging.warning(f"incomplete esnli batch with keys {str(list(examples.keys()))}.")
                return None
            # batch size = 3?
            if not len(examples["label"])==3:
                logging.warning(f"flawed esnli batch with batch size {len(examples['label'])}.")
                return None
            # three different labels?
            if not set(examples["label"])=={0, 1, 2}:
                logging.warning(f"flawed esnli batch with labels {str(examples['label'])}.")
                return None
            # one and the same premise?
            if not len(set(examples["premise"]))==1:
                logging.warning(f"flawed esnli batch with different premises {str(examples['premise'])}.")
                return None
            # at least explanation1 given?
            if any(e=='' for e in examples["explanation_1"]):
                logging.warning(f"missing explanation for premise={str(examples['premise'][0])} (proceeding nonetheless).")

            ### map to PreprocessedESNLIExample format
            idx = {label:i for i,label in enumerate(examples["label"])}
            preprocessed_example:PreprocessedESNLIExample = {
                "premise":   examples["premise"][0],
                "hypothesis_ent":  examples["hypothesis"][idx[0]],
                "hypothesis_neu":  examples["hypothesis"][idx[1]],
                "hypothesis_con":  examples["hypothesis"][idx[2]],
                "explanation_ent": [
                    examples["explanation_1"][idx[0]],
                    examples["explanation_2"][idx[0]],
                    examples["explanation_3"][idx[0]]
                ],
                "explanation_neu": [
                    examples["explanation_1"][idx[1]],
                    examples["explanation_2"][idx[1]],
                    examples["explanation_3"][idx[1]]
                ],
                "explanation_con": [
                    examples["explanation_1"][idx[2]],
                    examples["explanation_2"][idx[2]],
                    examples["explanation_3"][idx[2]]
                ]
            }

            # fill in explanations in case they are missing
            # we assume that "explanation_1" is given
            for key in ["explanation_ent","explanation_neu","explanation_con"]:
                for i in [1,2]:
                    if preprocessed_example[key][i] == "":
                        preprocessed_example[key][i] = preprocessed_example[key][0]

            # batch (with batchsize=1)
            preprocessed_example = {k:[v] for k,v in preprocessed_example.items()}

            return preprocessed_example

        dataset = dataset.map(merge_triple_batches, batched=True, batch_size=3, remove_columns=dataset.column_names)

        return dataset


    # stores argument configurations used for creating DeepA2 data records 
    configurations = {
        "entailment": [
            eSNLIConfiguration(
                label="entailment",
                scheme_name = "modus ponens",
                formal_scheme = ["{p}","{p} -> {q}", "{q}"],
                placeholders = {'p':"{premise}", "q":"{hypothesis}"},
            ),
        ],
        "contradiction": [
            eSNLIConfiguration(
                label="contradiction",
                scheme_name = "modus ponens",
                formal_scheme = ["{p}","{p} -> ¬{q}", "¬{q}"],
                placeholders = {'p':"{premise}", "q":"{hypothesis}"},
            ),
            eSNLIConfiguration(
                label="contradiction",
                scheme_name = "modus ponens",
                formal_scheme = ["{p}","{p} -> ¬{q}", "¬{q}"],
                placeholders = {'p':"{hypothesis}", "q":"{premise}"},
            ),
            eSNLIConfiguration(
                label="contradiction",
                scheme_name = "modus tollens",
                formal_scheme = ["{q}","{p} -> ¬{q}", "¬{p}"],
                placeholders = {'q':"{premise}", "p":"{hypothesis}"},
            ),
            eSNLIConfiguration(
                label="contradiction",
                scheme_name = "modus tollens",
                formal_scheme = ["{q}","{p} -> ¬{q}", "¬{p}"],
                placeholders = {'q':"{hypothesis}", "p":"{premise}"},
            ),
        ]
    }

    def __init__(self) -> None:
        """
        A fresh builder instance should contain a blank product object, which is
        used in further assembly.
        """
        # check whether template files are accessible
        if not (template_dir / "esnli").exists():
            logging.debug(f"Package dir: {package_dir}")
            logging.debug(f"Resolve template dir: {template_dir}")
            logging.debug(f"List template dir: {list(template_dir.glob('*'))}")
            err_m = f'No "esnli" subdirectory in template_dir {template_dir.resolve()}'
            raise ValueError(err_m)
        self._env = Environment(
            loader = FileSystemLoader(template_dir),
            autoescape=select_autoescape()
        )


        # register filters
        self._env.filters['lowerall'] = jjfilters.lowerall
        self._env.filters['negation'] = jjfilters.negation
        self._env.filters['conditional'] = jjfilters.conditional

        self.reset()


    @property
    def input(self) -> PreprocessedESNLIExample:
        """
        The input of any builder is a proprocessed example
        """
        return self._input

    @input.setter
    def input(self, preprocessed_example: PreprocessedESNLIExample) -> None:
        """
        Sets input for building next product.
        """
        self._input = {k:v[0] for k,v in preprocessed_example.items()}

    def configure_product(self) -> None:
        # populate product with configs
        i = 0
        for label in ["entailment","contradiction"]:        
            # argument_mask specifies which part of argument will be dropped in source text:
            for argument_mask in [[1,1,1],[0,1,1],[1,0,1],[1,1,0]]:
                # distractor_mask specifies which distractors will be dropped in source text:
                for distractor_mask in [[1,1],[1,0],[0,1],[0,0]]:
                    config = self.configurations[label][ i % len(self.configurations[label]) ]
                    config.argdown_err_template_path = random.choice(self.argdown_err_templates)
                    metadata = {"id":str(uuid.uuid4()), "config":config, "argument_mask":argument_mask, "distractor_mask":distractor_mask, "label":label}
                    deepa2record = DeepA2Item(metadata=metadata)
                    self._product.append(deepa2record)
                    i += 1

    def produce_da2item(self) -> None:
        for i,_ in enumerate(self._product):
            self.populate_record(i)


    def populate_record(self,i) -> None:
        record = self._product[i]
        config = record.metadata["config"]

        ### Initialize: mapping input data to argumentative roles
        if record.metadata["label"] == "entailment":
            data = {
                "premise": self.input["premise"],
                "hypothesis": self.input["hypothesis_ent"],
                "premise_cond": self.input["explanation_ent"][i%3], # used in source text
                "distractors": [self.input["hypothesis_neu"],self.input["explanation_con"][i%3]], # used in source text
            }
        else: # label == contradiction
            data = {
                "premise": self.input["premise"],
                "hypothesis": self.input["hypothesis_con"],
                "premise_cond": self.input["explanation_con"][i%3], # used in source text
                "distractors": [self.input["hypothesis_neu"],self.input["explanation_ent"][i%3]], # used in source text
            }


        ### Step 1: construct argdown 
        # argument list
        argument_list = [self._env.from_string(t).render(data) for t in config.nl_scheme]
        # argdown
        argdown_template = self._env.get_template(config.argdown_template_path)
        record.argdown_reconstruction = (argdown_template.render(premise1=argument_list[0],premise2=argument_list[1],conclusion=argument_list[-1],scheme=config.scheme_name))
        # erroneous argdown
        argdown_err_template = self._env.get_template(config.argdown_err_template_path)
        record.erroneous_argdown = (argdown_err_template.render(premise1=argument_list[0],premise2=argument_list[1],conclusion=argument_list[-1],scheme=config.scheme_name))


        ### Step 2: premises and conclusion lists
        # premises
        record.premises = []
        for i in range(2):
            explicit = bool(record.metadata["argument_mask"][i])
            argdown_statement = ArgdownStatement(text=argument_list[i],explicit=explicit,ref_reco=i+1)
            record.premises.append(argdown_statement)
        # conclusion
        i=2
        explicit = bool(record.metadata["argument_mask"][i])
        argdown_statement = ArgdownStatement(text=argument_list[i],explicit=explicit,ref_reco=i+1)
        record.conclusion = [argdown_statement]


        ### Step 3: formalizations
        # premises
        record.premises_formalized = []
        for i in range(2):
            formalization = Formalization(form=config.formal_scheme[i],ref_reco=i+1)
            record.premises_formalized.append(formalization)
        # conclusion
        i=2
        formalization = Formalization(form=config.formal_scheme[i],ref_reco=i+1)
        record.conclusion_formalized = [formalization]
        # placeholders
        record.misc_placeholders = {k:v.format(**data) for k,v in config.placeholders.items()}


        ### Step 4: source text, reasons, conjectures

        # 4.a) compile list with all sentences in source text
        argument_source_list = []
        # add distractors
        for i,s in enumerate(data["distractors"]):
            if record.metadata["distractor_mask"][i]:
                argument_source_list.append(["distractor",s])
        # add reasons
        argument_list2 = argument_list.copy()
        argument_list2[1] = data["premise_cond"] # replace conditional
        for i,s in enumerate(argument_list2[:-1]):
            if record.metadata["argument_mask"][i]:
                argument_source_list.append(["reason",QuotedStatement(text=s,ref_reco=i+1,starts_at=None)])
        # add conclusion
        i=2
        if record.metadata["argument_mask"][i]:
            s = argument_list2[i]
            argument_source_list.append(["conjecture",QuotedStatement(text=s,ref_reco=i+1,starts_at=None)])
        # shuffle                 
        random.shuffle(argument_source_list)

        # 4.b) walk through list and compile source text as well as reason, conclusions, distractors
        record.argument_source = ""
        record.reason_statements = []
        record.conclusion_statements = []
        record.distractors = []
        for item in argument_source_list:
            pointer = len(record.argument_source)
            if item[0]=="distractor":
                record.argument_source += item[1]
                record.distractors.append(item[1])
            elif item[0] in ["reason","conjecture"]:
                record.argument_source += item[1].text
                item[1].starts_at = pointer
                if item[0] == "reason":
                    record.reason_statements.append(item[1])
                else:
                    record.conclusion_statements.append(item[1])
            record.argument_source += " "

        record.argument_source = record.argument_source.strip(" ") 



        ### Step 5: gist, source_paraphrase, context, title
        # use premise2 as gist
        record.gist = data["premise_cond"]
        # source paraphrase
        sp_template = self._env.get_template(config.source_paraphrase_template_path)
        record.source_paraphrase = (sp_template.render(premises=[d.text for d in record.reason_statements],conclusion=[d.text for d in record.conclusion_statements]))
        # title 
        # TODO


    def postprocess_da2item(self) -> None:
        pass

    def add_metadata_da2item(self) -> None:
        pass
