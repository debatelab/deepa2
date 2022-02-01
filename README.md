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
poetry run build-nli esnli --debug-mode --export-path ./data/processed
```

## Contribute a DeepA2Builder for another Dataset

We welcome contributions to this repository, especially scripts that port existing datasets to the DeepA2 Framework. Within this repo, code that transforms data into the DeepA2 format contains

1. [**required**] a builder class that implements the abstract `builder.Builder` interface (such as, e.g., `nli_builder.eSNLIBuilder`);
2. [**required**] a script that defines a pipeline for transforming the original data as a typer app / command, using the concrete builder (such as, e.g., `scripts/build_nli.py`);
3. [**recommended**] dataclasses which describe raw and preprocessed examples and extend the dummy classes `builder.RawExample` and `builder.PreprocessedExample`;
4. [**recommended**] a documentation of the piepline (as for example in `docs/esnli.md`).

Consider **suggesting** to collaboratively construct such pipeline by opening a [new issue](./issues).

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