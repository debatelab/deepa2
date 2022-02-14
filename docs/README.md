# Documentation of `debatelab/deepa2`

The following pages outline how the scripts provided by this repository are used to import existing NLP datasets into the DeepA2 framework:

* [e-SNLI](esnli.md)
* [AIFdb](aifdb.md)
* [Entailment Bank](enbank.md)

## Usage options for `deepa2 bake`

```
% deepa2 bake --help
Usage: deepa2 bake [OPTIONS]

  Builds a new 🎂 DeepA2 dataset from `path` using a Builder that fits
  `source_type`.

Options:
  --source-type TEXT    type of the source dataset, used tochoose a compatible
                        Builder; currently supported source types:`esnli`,
                        `aifdb`, `enbank`.
  --name TEXT           name of preconfigured dataset(s) to load given
                        `source_type`; see documentation of Builders for more
                        info.
  --path TEXT           path to the input dataset
  --export-path TEXT    local directory to which built DeepA2 dataset is
                        saved.
  --debug-size INTEGER  number of items to process for debugging
  --configfile TEXT     path to yml file that contains a configuration for
                        `deepa2 bake`. The configfile will typically set
                        builder-specific parameters, see documentation of
                        Builders for more info.
  --help                Show this message and exit.
```


A `configuration file` may look like:

```yml
sources:                # we load two datasets:
-   path: "./data/processed/esnli"
    revision: "main"    # tag used to pick version from HF hub
-   path: "./data/processed/enbank/task_2"
    revision: "main"
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
