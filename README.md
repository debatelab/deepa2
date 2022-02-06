![unit tests](https://github.com/debatelab/deepa2-datasets/actions/workflows/run_pytest.yml/badge.svg?branch=main) ![code quality](https://github.com/debatelab/deepa2-datasets/actions/workflows/code_quality_checks.yml/badge.svg?branch=main)

# Deep Argument Analysis (deepa2)

Resources for creating, importing, managing, and using DeepA2 datasets (Deep Argument Analysis Framework).

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
poetry run deepa2 build \\
  --name esnli \\
  --debug-size 100 \\
  --export-path ./data/processed    
```

## Contribute a DeepA2Builder for another Dataset

We welcome contributions to this repository, especially scripts that port existing datasets to the DeepA2 Framework. Within this repo, a code module that transforms data into the DeepA2 format contains

1. a Builder class that describes how DeepA2 examples will be constructed and that implements the abstract `core.Builder` interface (such as, e.g., `nli_builder.ESNLIBuilder`);
2. a DataLoader which provides a method for loading the raw data as a HF Dataset object (such as, for example, `aifdb_builder.AIFDBLoader`) -- you may use `core.DataLoader` as is in case the data is available in a way compatible with HF Dataset;
3. dataclasses which describe the features of the raw data and the preprocessed data, and which extend the dummy classes `core.RawExample` and `core.PreprocessedExample`;
4. a collection of unit tests that check the concrete Builder's methods (such as, e.g., `tests/test_esnli.py`);
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