# Documentation of `debatelab/deepa2`

The following pages outline how the scripts provided by this repository are used to import existing NLP datasets into the DeepA2 framework:

* [e-SNLI](esnli.md)
* [AIFdb](aifdb.md)
* [Entailment Bank](enbank.md)

## Usage options for `deepa2 bake`

```
% deepa2 serve --help
Usage: deepa2 serve [OPTIONS]

  Prepares 🎂 DeepA2 datasets for text-2-text training, and serves a single 🍰
  t2t dataset.

Options:
  --path TEXT                path to DeepA2 dataset
  --revision TEXT            version of the dataset (script) to load
  --export-path TEXT         local directory to which t2t dataset is saved
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