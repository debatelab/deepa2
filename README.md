# deepa2-datasets

Resources for Creating, Importing and Managing DeepA2 Argument Analysis Framework Datasets

## Getting Started

Install [poetry](https://python-poetry.org/docs/#installation). 

Clone the repository:
```bash
git clone https://github.com/debatelab/deepa2-datasets.git
```

Install this package from with the repo's root folder:
```bash
poetry install
```

Run a script, e.g.:
```bash
poetry run build-nli esnli --debug-mode --export-path ./data/processed
```

## Contribute a DeepA2Builder for another Dataset

We welcome contributions to this repository, especially scripts that port existing datasets to the DeepA2 Framework. Within this repo, code that transforms data into the DeepA2 format contains

1. [**required**] a builder class that implements the abstract `builder.Builder` interface (such as, e.g., `nli_builder.eSNLIBuilder`);
2. [**required**] a script that defines a pipeline for transforming the original data as a typer app / command  (such as, e.g., `scripts/build_nli`) using the concrete builder (item 1);
3. [**recommended**] dataclasses which describe raw and preprocessed examples and extend the abstract classes `builder.RawExample` and `builder.PreprocessedExample`;
4. [**recommended**] a documentation of the piepline (as for example in `docs/esnli.md`).