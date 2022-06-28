# Documentation of `debatelab/deepa2`

The following pages outline how the scripts provided by this repository are used to import existing NLP datasets into the DeepA2 framework:

* [e-SNLI](esnli.md)
* [AIFdb](aifdb.md)
* [Entailment Bank](enbank.md)
* [IBM-ArgQ-Rank](argq.md)
* [IBM-KPA](argkp.md)


## Usage options for `deepa2 serve`

```
% deepa2 serve --help
Usage: deepa2 serve [OPTIONS]

  Prepares 🎂 DeepA2 datasets for text-2-text training, and serves a single 🍰
  t2t dataset.

Options:
  --path TEXT                path to DeepA2 dataset
  --revision TEXT            version of the dataset (script) to load
  --export-path TEXT         local directory to which t2t dataset is saved
  --export-format TEXT       format in t2t dataset is saved (parquet, csv,
                             jsonl), will use parquet if left blank
  --input-column-name TEXT   name of input column of t2t dataset  [default:
                             text]
  --target-column-name TEXT  name of target column of t2t dataset  [default:
                             target]
  --configfile TEXT          path to yml configuration while; commandline
                             options overwrite config file; using a config
                             file allows for serving multiple deepa2 datasets
                             as a single t2t dataset; generative modes covered
                             canalso be specified in config file.
  --help                     Show this message and exit.

```


A `configuration file` may look like:

```yml
sources:
-   path: "prjct/data"  # a local dir or a dataset at hf hub 
    data_files:         # all files relative to path 
        train:          # these files go in train split
        -   "enbank/task_1/train/train.parquet"
        -   "moral-maze/train/train.parquet"
        validation:     # these files go in validation split
        -   "enbank/task_1/validation/validation.parquet"
        -   "moral-maze/validation/validation.parquet"
        test:           # these files go in test split
        -   "enbank/task_1/test/test.parquet"
        -   "moral-maze/test/test.parquet"
export_format: "csv"
export_path: "data/tmp" 
generative_modes:       # list of modes used to construct t2t items 
-   name: "s+r => a"    # fully specified mode:
    target: "argdown_reconstruction"
    input:
    -   "argument_source"
    -   "reasons"
    weight: 1           # controls frequency of mode in t2t dataset
-   name: "s => c"      # mode will be inferred from keys in name
```

Keys for specifying generative modes:

```bash
% deepa2 keys
{'a': 'argdown_reconstruction',
 'c': 'conclusion',
 'e': 'erroneous_argdown',
 'fc': 'conclusion_formalized',
 'fi': 'intermediary_conclusions_formalized',
 'fp': 'premises_formalized',
 'g': 'gist',
 'h': 'source_paraphrase',
 'i': 'intermediary_conclusions',
 'j': 'conjectures',
 'k': 'plchd_substitutions',
 'p': 'premises',
 'r': 'reasons',
 's': 'source_text',
 't': 'title',
 'x': 'context'}

```


## Extensions of the Original Framework

We're continously extending the original `deepa2` framework and have, so far, added the following additional features (dataset dimensions):

```yml
erroneous_argdown:  # flawed reconstruction, e.g.
  "(1) God exists.
   ----
   (2) God exists."
gist:               # the argument's main point, e.g.
  "Being perfect entails real existence."
source_paraphrase:  # a maximally clear re-rendition of source_text
  "God is a being than which no greater can be conceived. If such a 
  being fails to exist, then a greater being — namely, a being than 
  which no greater can be conceived, and which exists — can be 
  conceived. But this would be absurd: nothing can be greater than 
  a being than which no greater can be conceived. So God exists."
title:              # a telling title of the argument, e.g.
  "The ontological argument"
context:             # the context, e.g.
  "Does God exist? This is a famous proof from St. Anselm's 
  Proslogion"
```
