<p align="center">
    <a href="https://github.com/debatelab/deepa2/actions/workflows/run_pytest.yml">
        <img alt="unit tests" src="https://github.com/debatelab/deepa2-datasets/actions/workflows/run_pytest.yml/badge.svg?branch=main">
    </a>
    <a href="https://github.com/debatelab/deepa2/actions/workflows/code_quality_checks.yml">
        <img alt="code quality" src="https://github.com/debatelab/deepa2-datasets/actions/workflows/code_quality_checks.yml/badge.svg?branch=main">
    </a>
</p>

<h1 align="center">
    <p>Deep Argument Analysis (deepa2)</p>
</h1>

Resources for creating, importing, managing, and using DeepA2 datasets (Deep Argument Analysis Framework).

* [Documentation](docs/)
* DeepA2 Datasets


## Integrating `deepa2` into Your Training Pipeline

1. Install `deepa2` into your ML project's virtual environment, e.g.:

```bash
source my-projects-venv/bin/activate 
python -m pip install git+https://github.com/debatelab/deepa2.git
```

2. Add `deepa2` preprocessor to your training pipeline. Your training script may look like, for example:

```sh
#!/bin/bash

# configure and activate environment
...

# download deepa2 datasets and 
# prepare for text2text training
deepa2 preptrain \
    --path some-deepa2-dataset \
    --export_path tmp/t2t-deepa2 \  # <<< 🤝

# run default training script, 
# e.g., with 🤗 Transformers
python examples/pytorch/summarization/run_summarization.py \
    --dataset_name tmp/t2t-deepa2 \ # <<< 🤝
    --text_column "text" \
    --summary_column "target" \
    --...

# clean-up
rm -r tmp/t2t-deepa2
```

3. That's it.


## Create DeepA2 datasets with `deepa2` from existing NLP data

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
2. a DataLoader which provides a method for loading the raw data as a 🤗 Dataset object (such as, for example, `aifdb_builder.AIFDBLoader`) -- you may use `core.DataLoader` as is in case the data is available in a way compatible with 🤗 Dataset;
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