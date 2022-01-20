from deepa2datasets.builder import ArgdownStatement, Builder, Formalization, QuotedStatement
from deepa2datasets.builder import DeepA2Item
from deepa2datasets.config import template_dir
import deepa2datasets.jinjafilters as jjfilters

import random

from jinja2 import Environment, FileSystemLoader, select_autoescape

from typing import Any,List,Dict

from dataclasses import dataclass, field

@dataclass
class eSNLIConfiguration():
    label:str
    argdown_template_path:str = "esnli/argdown_generic.txt"
    scheme_name:str = "modus ponens"
    formal_scheme:List = field(default_factory=lambda: ["{p}","{p} -> {q}", "{q}"])
    placeholders:Dict = field(default_factory=lambda: {'p':"{premise}", "q":"{hypothesis}"})
    nl_scheme:List = field(default_factory=lambda: ["{{ premise }}", "{{ premise | conditional(hypothesis) }}", "{{ hypothesis }}"]) # Jinja templates



class eSNLIBuilder(Builder):
    """
    The Concrete Builder classes follow the Builder interface and provide
    specific implementations of the building steps. Your program may have
    several variations of Builders, implemented differently.
    """

    # stores argument configurations used for creating DeepA2 data records 
    configurations = {
        "entailment": [
            eSNLIConfiguration(
                label="entailment",
                scheme_name = "modus ponens",
                formal_scheme = ["{p}","{p} -> {q}", "{q}"],
                nl_scheme = ["{{ premise | lower }}", "{{ premise | conditional(hypothesis) }}", "{{ hypothesis | lower }}"],
                placeholders = {'p':"{premise}", "q":"{hypothesis}"},
            ),
        ],
        "contradiction": [
            eSNLIConfiguration(
                label="contradiction",
                scheme_name = "modus ponens",
                formal_scheme = ["{p}","{p} -> ¬{q}", "¬{q}"],
                nl_scheme = ["{{ premise | lower }}", "{{ premise | conditional(hypothesis | negation) }}", "{{ hypothesis | negation }}"],
                placeholders = {'p':"{premise}", "q":"{hypothesis}"},
            ),
            eSNLIConfiguration(
                label="contradiction",
                scheme_name = "modus ponens",
                formal_scheme = ["{p}","{p} -> ¬{q}", "¬{q}"],
                nl_scheme = ["{{ hypothesis | lower }}", "{{ hypothesis | conditional(premise | negation) }}", "{{ premise | negation }}"],
                placeholders = {'p':"{hypothesis}", "q":"{premise}"},
            ),
            eSNLIConfiguration(
                label="contradiction",
                scheme_name = "modus tollens",
                formal_scheme = ["{p}","{q} -> ¬{p}", "¬{q}"],
                nl_scheme = ["{{ premise | lower }}", "{{ hypothesis | conditional(premise | negation) }}", "{{ hypothesis | negation }}"],
                placeholders = {'p':"{premise}", "q":"{hypothesis}"},
            ),
            eSNLIConfiguration(
                label="contradiction",
                scheme_name = "modus tollens",
                formal_scheme = ["{p}","{q} -> ¬{p}", "¬{q}"],
                nl_scheme = ["{{ hypothesis | lower }}", "{{ premise | conditional(hypothesis | negation) }}", "{{ premise | negation }}"],
                placeholders = {'p':"{hypothesis}", "q":"{premise}"},
            ),
        ]
    }

    def __init__(self) -> None:
        """
        A fresh builder instance should contain a blank product object, which is
        used in further assembly.
        """
        self._env = Environment(
            loader = FileSystemLoader(template_dir),
            autoescape=select_autoescape()
        )

        # register filters
        self._env.filters['lowerall'] = jjfilters.lowerall
        self._env.filters['negation'] = jjfilters.negation
        self._env.filters['conditional'] = jjfilters.conditional

        self.reset()

    def reset(self) -> None:
        self._input = {}
        self._product = []

    @property
    def product(self) -> List[DeepA2Item]:
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
        self.reset()
        return product

    def fetch_input(self,input_dataset) -> Any:
        # DUMMY
        self._input = {
            "p":"This church choir sings to the masses as they sing joyous songs from the book at a church.",
            "he":"The church is filled with song.",
            "hn":"The church has cracks in the ceiling.",
            "hc":"A choir singing at a baseball game.",
            "en": [
                "Not all churches have cracks in the ceiling",
                "There is no indication that there are cracks in the ceiling of the church.",
                "Not all churches have cracks in the ceiling."
            ],
            "ee": [
                '"Filled with song" is a rephrasing of the "choir sings to the masses."',
                'hearing song brings joyous in the church.',
                'If the church choir sings then the church is filled with song.'
            ],
            "ec": [
                'A choir sing some other songs other than book at church during the base play. they cannot see book and play base ball same time.',
                'The choir is at a chruch not a baseball game.',
                'A baseball game isn’t played at a church.'
            ]
        }
        return None

    def configure_product(self) -> None:
        # populate product with configs
        i = 0
        for label in ["entailment","contradiction"]:        
            # argument_mask specifies which part of argument will be dropped in source text:
            for argument_mask in [[1,1,1],[0,1,1],[1,0,1],[1,1,0]]:
                # distractor_mask specifies which distractors will be dropped in source text:
                for distractor_mask in [[1,1],[1,0],[0,1],[0,0]]:
                    config = self.configurations[label][ i % len(self.configurations[label]) ]
                    metadata = {"config":config, "argument_mask":argument_mask, "distractor_mask":distractor_mask, "label":label}
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
                "premise": self._input["p"],
                "hypothesis": self._input["he"],
                "premise_cond": self._input["ee"][i%3], # used in source text
                "distractors": [self._input["hn"],self._input["ec"][i%3]], # used in source text
            }
        else: # label == contradiction
            data = {
                "premise": self._input["p"],
                "hypothesis": self._input["hc"],
                "premise_cond": self._input["ec"][i%3], # used in source text
                "distractors": [self._input["hn"],self._input["ee"][i%3]], # used in source text
            }


        ### Step 1: construct argdown 
        # argument list
        argument_list = [self._env.from_string(t).render(data) for t in config.nl_scheme]
        # argdown
        argdown_template = self._env.get_template(config.argdown_template_path)
        record.argdown_reconstruction = (argdown_template.render(premise1=argument_list[0],premise2=argument_list[1],conclusion=argument_list[-1],scheme=config.scheme_name))
        # erroneous argdown
        # TODO
        

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





    def postprocess_da2item(self) -> None:
        pass

    def add_metadata_da2item(self) -> None:
        pass
