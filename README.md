# deepa2-datasets

Resources for Creating, Importing and Managing DeepA2 Argument Analysis Framework Datasets.

* [Documentation](docs/)
* DeepA2 Datasets

## Getting Started

Install [poetry](https://python-poetry.org/docs/#installation). 

Clone the repository:
```bash
git clone https://github.com/debatelab/deepa2-datasets.git
```

Install this package from within the repo's root folder:
```bash
poetry install
```

Run a script, e.g.:
```bash
poetry run build-nli esnli --debug-size 300 --export-path ./data/processed
```

## Contribute a DeepA2Builder for another Dataset

We welcome contributions to this repository, especially scripts that port existing datasets to the DeepA2 Framework. Within this repo, code that transforms data into the DeepA2 format contains

1. a Builder class that implements the abstract `core.Builder` interface (such as, e.g., `nli_builder.eSNLIBuilder`);
2. a DataLoader which provides a method for loading the raw data as a HF Dataset object (such as, for example, `aifdb_builder.AIFDBLoader`) -- you may use `core.DataLoader` as is in case the data is available in a way compatible with HF Dataset;
3. dataclasses which describe the features of the raw data and the preprocessed data, and which extend the dummy classes `core.RawExample` and `core.PreprocessedExample`;
4. a script that defines a typer app / command for transforming the original data, using the concrete builder (such as, e.g., `scripts/build_nli.py`);
5. a documentation of the pipeline (as for example in `docs/esnli.md`).

Consider **suggesting** to collaboratively construct such a pipeline by opening a [new issue](https://github.com/debatelab/deepa2-datasets/issues).

## Citation

This repository builds on and extends the DeepA2 Framework originally presented in:

```bibtex
@misc{betz2021deepa2,
      title={DeepA2: A Modular Framework for Deep Argument Analysis with Pretrained Neural Text2Text Language Models}, 
      author={Gregor Betz and Kyle Richardson},
      year={2021},
      eprint={2110.01509},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```